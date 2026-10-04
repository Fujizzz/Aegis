"""Training option validation shared by run preparation and the trainer."""
import copy
import math


DIMENSIONS = ('emotion', 'exaggeration', 'bias', 'commonsense')
COLLABORATIVE = ('cons', 'pred', 'div', 'gate')
FOUR_AGENT_VARIANTS = ('full', 'no_comm', 'no_gate', 'no_dim', 'no_cons', 'no_pred',
                       'no_div', 'no_gate_loss', 'no_collab', 'custom')
NEURAL_VARIANTS = FOUR_AGENT_VARIANTS + ('single_agent',)
NON_NEURAL_VARIANTS = ('tfidf_nb', 'teacher_direct')
VARIANTS = NEURAL_VARIANTS + NON_NEURAL_VARIANTS


def resolve_variant(config):
    """Apply a single named ablation to the supplied baseline before archiving.

    Untargeted settings remain unchanged. `custom` supports combined switches.
    This function is idempotent and does not modify the requested config.
    """
    cfg = copy.deepcopy(config)
    stop = cfg.get('runtime', {}).get('stop_after_epoch')
    if stop is not None and (type(stop) is not int or stop < 1 or stop > sum(epoch_counts(cfg.get('training', {})))):
        raise ValueError('stop_after_epoch must be a valid complete epoch boundary')
    variant = cfg.get('experiment', {}).get('variant', 'custom')
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    model, data = cfg.setdefault('model', {}), cfg.setdefault('data', {})
    weights = cfg.setdefault('training', {}).setdefault('loss_weights', {})
    if variant in ('single_agent', 'tfidf_nb', 'teacher_direct'):
        data['use_pseudo_labels'] = False
        model.update(comm_rounds=0, use_gate=False, num_agents=1)
        weights.update({key: 0.0 for key in (*DIMENSIONS, *COLLABORATIVE)})
        weights['final'] = 1.0
    if variant == 'no_comm':
        model['comm_rounds'] = 0
    if variant == 'no_gate':
        model['use_gate'] = False
    if variant == 'no_dim':
        data['use_pseudo_labels'] = False
    if variant in ('no_cons', 'no_pred', 'no_div', 'no_gate_loss'):
        weights[{'no_gate_loss': 'gate'}.get(variant, variant[3:])] = 0.0
    if variant == 'no_collab':
        weights.update({key: 0.0 for key in COLLABORATIVE})
    rounds = model.setdefault('comm_rounds', 2)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise ValueError('model.comm_rounds must be a nonnegative integer')
    for section, key in ((model, 'use_gate'), (data, 'use_pseudo_labels')):
        if not isinstance(section.setdefault(key, True), bool):
            raise ValueError(f'{key} must be a boolean')
    if model.get('fuse_mode', 'mean') not in ('mean', 'attn'):
        raise ValueError('model.fuse_mode must be mean or attn')
    if rounds == 0:
        model['fuse_mode'] = 'mean'
    if rounds == 0 or not model['use_gate']:
        weights['gate'] = 0.0
    if not data['use_pseudo_labels']:
        weights.update({key: 0.0 for key in DIMENSIONS})
    for key, value in weights.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0):
            raise ValueError(f'loss_weights.{key} must be finite and nonnegative')
    return cfg


def grad_accum_steps(training):
    value = training.get("grad_accum_steps", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("training.grad_accum_steps must be a positive integer")
    return value


def epoch_counts(training):
    warmup = int(training.get("warmup_epochs", 1))
    collab = int(training.get("collab_epochs", training.get("epochs", 3) - warmup))
    if warmup < 0 or collab < 0 or warmup + collab < 1:
        raise ValueError("Training requires nonnegative phases and at least one epoch")
    return warmup, collab
