"""Record exactly what produced a run: command, config, seed, code version and library versions."""
import json
import os
import platform
import subprocess
import sys

import torch
import transformers


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def collect_run_info(config):
    status = _git('status', '--porcelain', '--untracked-files=no')
    plain = (str, int, float, bool, list, type(None))
    return {
        'command': ' '.join(sys.argv),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(status) if status is not None else None,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # optimizer/activation are classes -> store their names; skip big objects like word2id
        'config': {k: getattr(v, '__name__', v) for k, v in vars(config).items()
                   if isinstance(getattr(v, '__name__', v), plain)},
    }


def save_run_info(config, path):
    info = collect_run_info(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    print('RUN INFO', json.dumps(info, indent=2))
    return info