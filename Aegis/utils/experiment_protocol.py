"""Shared event/fold contract for archives, predictions and result verification."""
from utils.freeze_pheme import EVENTS, STEMS

IN_EVENT = 'in_event_v1'
CROSS_EVENT = 'loeo_source_membership_v1'


def scope(config):
    value = config['experiment'].get('evaluation_scope', IN_EVENT)
    if value not in (IN_EVENT, CROSS_EVENT):
        raise ValueError('Unknown evaluation scope')
    return value


def allowed_events(config, split):
    exp = config['experiment']
    event = exp['event']
    if scope(config) == IN_EVENT:
        return {event}
    if event not in EVENTS or exp.get('held_out_event') != event:
        raise ValueError('Invalid held-out event binding')
    expected = set(EVENTS) - {event}
    if set(exp.get('training_events', [])) != expected:
        raise ValueError('Training events must exclude exactly the held-out event')
    return {event} if split == 'test' else expected


def split_relative(config, split):
    event = config['experiment']['event']
    allowed_events(config, split)
    folder = 'folds/' + event if scope(config) == CROSS_EVENT else 'split_' + STEMS[event]
    return folder + '/' + split + '.csv'
