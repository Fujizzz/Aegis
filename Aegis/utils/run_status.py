"""Explicit run state, including failures when Trainer is called without the CLI."""
from functools import wraps
from pathlib import Path
from utils.run_archive import write_json


def guarded(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException as error:
            if hasattr(self, 'run_state'):
                phase = self.run_state.get('phase', 'initializing')
                self.run_state.update(status='failed', error_type=type(error).__name__, failed_phase=phase)
                if phase in ('validation', 'test'):
                    key = 'val' if phase == 'validation' else 'test'
                    self.run_state[key].update(status='failed', error_type=type(error).__name__,
                                               epoch=getattr(self, 'current_epoch', None))
                if phase == 'training':
                    self.run_state['training'].update(status='failed')
                write_json(Path(self.save_dir)/'status.json', self.run_state)
            raise
    return call


def update_status(trainer, **fields):
    trainer.run_state.update(fields)
    write_json(Path(trainer.save_dir)/'status.json', trainer.run_state)
