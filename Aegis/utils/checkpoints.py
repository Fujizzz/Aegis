"""Checkpoint layout, provenance and inference compatibility checks."""
import json
from pathlib import Path
from utils.run_archive import sha256
from training.options import resolve_variant
from utils.text_input import INPUT_FORMAT

COMPONENTS = ('agent_emotion', 'agent_exaggeration', 'agent_bias', 'agent_commonsense',
              'communication_module', 'final_classifier', 'team_layer', 'pred_heads')


def checkpoint_paths(directory):
    directory = Path(directory).resolve()
    paths = {name: str(directory / f'{name}.pth') for name in COMPONENTS}
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return paths


def checkpoint_config(directory):
    directory = Path(directory).resolve()
    path = directory / 'config.resolved.json'
    manifest = json.loads((directory / 'checkpoint_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('input_format') != INPUT_FORMAT:
        raise ValueError('Checkpoint lacks the verified news input contract; explicit migration required')
    if sha256(path) != manifest['sha256'].get(path.name):
        raise ValueError('Checkpoint config hash mismatch')
    schema = directory / 'label_schema.json'
    if sha256(schema) != manifest['sha256'].get(schema.name):
        raise ValueError('Checkpoint label schema hash mismatch')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_paths(paths, config):
    parents = {Path(path).resolve().parent for path in paths.values()}
    if len(parents) != 1:
        raise ValueError('Checkpoint components must belong to the same directory')
    directory = next(iter(parents))
    manifest = json.loads((directory / 'checkpoint_manifest.json').read_text(encoding='utf-8'))
    for path in paths.values():
        if sha256(path) != manifest['sha256'].get(Path(path).name):
            raise ValueError('Checkpoint content hash mismatch')
    saved = resolve_variant(checkpoint_config(directory))
    actual = resolve_variant(config)
    # Device placement is portable, architecture and input meaning are not.
    for section in ('model', 'ablation'):
        if saved.get(section) != actual.get(section):
            raise ValueError(f'Checkpoint {section} config mismatch')
    for key, default in (('max_length', 512), ('label_schema', None)):
        if saved['data'].get(key, default) != actual['data'].get(key, default):
            raise ValueError(f'Checkpoint data.{key} mismatch')
