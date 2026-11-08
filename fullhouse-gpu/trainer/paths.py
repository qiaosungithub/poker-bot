"""Import paths and cache locations shared by the trainer, tests and eval tools.

Import this module FIRST: it points numba / inductor / triton caches into the
package's runtime directory (so nothing is written into the read-only upstream
tree) and puts the sibling packages on ``sys.path``:

* ``../traversal`` -- level-synchronous batched GPU traversal (``gpu_traverse``)
* ``../fit``       -- GPU reservoir buffers + fast CUDA-graph fitting (``gpu_fit``);
  its CUDA extension builds once into ``../fit/.runtime/torch_ext``
* upstream fullhouse-bot @ e504793d (+ ``fullhouse-repro/training_changes.patch``)

Locations (environment variables, with defaults):

* ``FULLHOUSE_SOURCE``      -- upstream source tree, default
  ``<repo>/fullhouse-repro/source`` (reconstruct it with docs/repository.md)
* ``FULLHOUSE_GPU_RUNTIME`` -- numba / inductor / triton caches, default
  ``<repo>/fullhouse-gpu/.runtime``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # fullhouse-gpu/trainer
PACKAGE = HERE.parent                           # fullhouse-gpu
REPO = PACKAGE.parent                           # repository root
TRAVERSAL = PACKAGE / "traversal"
FIT_DIR = PACKAGE / "fit"
REPRO = REPO / "fullhouse-repro"
SOURCE = Path(os.environ.get("FULLHOUSE_SOURCE", str(REPRO / "source"))).resolve()
RUNTIME = Path(os.environ.get("FULLHOUSE_GPU_RUNTIME", str(PACKAGE / ".runtime"))).resolve()

for _name, _sub in (("NUMBA_CACHE_DIR", "numba"),
                    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                    ("TRITON_CACHE_DIR", "triton")):
    _p = RUNTIME / _sub
    _p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_name, str(_p))

os.environ.setdefault("FULLHOUSE_SOURCE", str(SOURCE))
os.environ.setdefault("FULLHOUSE_GPU_RUNTIME", str(RUNTIME))

if not (SOURCE / "training" / "train.py").exists():
    raise RuntimeError(f"upstream fullhouse-bot source not found at {SOURCE}; reconstruct "
                       f"fullhouse-repro/source (docs/repository.md) or set FULLHOUSE_SOURCE")

for _p in (SOURCE, FIT_DIR, TRAVERSAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def fit_manifest() -> dict:
    """sha256 of every source file of the fitting code actually imported."""
    import hashlib
    out = {"fit_dir": str(FIT_DIR)}
    for f in sorted(FIT_DIR.glob("*")):
        if f.is_file() and f.suffix in (".py", ".cu", ".txt", ".md", ".json"):
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out
