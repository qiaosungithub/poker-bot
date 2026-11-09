"""Keep this project's environments, caches and temporary files on work storage.

Imported by the evaluation, ONNX export and latency scripts. The TPU/JAX entries
(TPU_LOG_DIR, JAX_COMPILATION_CACHE_DIR, TRAIN_PY) of the original training run
were removed with its launcher on the gpu-training branch; see main @ 7356b42.
"""
import os
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent.parent / '.runtime'
ENVIRONMENTS = RUNTIME / 'envs'
EVAL_PY = str(ENVIRONMENTS / 'poker-hu-validation/bin/python')
for name, relative in {
    'TMPDIR': 'tmp',
    'TMP': 'tmp',
    'TEMP': 'tmp',
    'XDG_CACHE_HOME': 'cache',
    'PIP_CACHE_DIR': 'cache/pip',
    'UV_CACHE_DIR': 'cache/uv',
    'UV_PYTHON_INSTALL_DIR': 'python',
    'TORCH_HOME': 'cache/torch',
    'HF_HOME': 'cache/huggingface',
    'NUMBA_CACHE_DIR': 'cache/numba',
}.items():
    path = RUNTIME / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
