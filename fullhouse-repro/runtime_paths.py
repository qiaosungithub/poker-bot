"""Keep this project's environments, caches and temporary files on work storage."""
import os
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent.parent / '.runtime'
ENVIRONMENTS = RUNTIME / 'envs'
TRAIN_PY = str(ENVIRONMENTS / 'poker-fullhouse-train312/bin/python')
EVAL_PY = str(ENVIRONMENTS / 'poker-hu-validation/bin/python')
for name, relative in {
    'TMPDIR': 'tmp',
    'TMP': 'tmp',
    'TEMP': 'tmp',
    'TPU_LOG_DIR': 'logs/tpu',
    'XDG_CACHE_HOME': 'cache',
    'PIP_CACHE_DIR': 'cache/pip',
    'UV_CACHE_DIR': 'cache/uv',
    'UV_PYTHON_INSTALL_DIR': 'python',
    'TORCH_HOME': 'cache/torch',
    'HF_HOME': 'cache/huggingface',
    'JAX_COMPILATION_CACHE_DIR': 'cache/jax',
    'NUMBA_CACHE_DIR': 'cache/numba',
}.items():
    path = RUNTIME / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
