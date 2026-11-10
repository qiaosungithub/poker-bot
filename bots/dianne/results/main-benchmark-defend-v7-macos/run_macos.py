"""macOS-only CPU-affinity portability adapter; frozen benchmark stays unchanged.

This is not Linux qualification or a production latency measurement.  The only
runtime substitutions are unavailable OS CPU-affinity functions.  Top-level
installation also executes when multiprocessing spawn imports this entrypoint.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import sys
from datetime import datetime, timezone

PATCHED = []
if sys.platform == "darwin":
    if not hasattr(os, "sched_getaffinity"):
        def _unrestricted_cpu_mask(pid):
            return set(range(os.cpu_count() or 1))
        os.sched_getaffinity = _unrestricted_cpu_mask
        PATCHED.append("os.sched_getaffinity: report all logical CPUs")
    if not hasattr(os, "sched_setaffinity"):
        def _ignore_cpu_pinning(pid, mask):
            return None
        os.sched_setaffinity = _ignore_cpu_pinning
        PATCHED.append("os.sched_setaffinity: no-op")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        # Preserve the frozen runner's required hash seed while retaining this
        # wrapper as the process entrypoint (the runner otherwise re-execs -m).
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                  *sys.argv[1:]], {**os.environ, "PYTHONHASHSEED": "0"})
    out = (Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
           if "--out" in sys.argv else None)
    adapter = Path(__file__).resolve()
    provenance = {
        "qualification": "macOS portability-adapted research run; not Linux qualification",
        "source_commit": "481d1f0",
        "adapter": str(adapter),
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "patches": PATCHED,
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "argv": sys.argv[1:],
        "started": datetime.now(timezone.utc).isoformat(),
        "frozen_files_modified": False,
        "game_rules_seed_or_score_changes": False,
        "cpu_affinity_enforced": False,
    }
    print(json.dumps({"platform_adapter": provenance}), flush=True)
    try:
        # alter_sys=False preserves this wrapper as multiprocessing's main file.
        # Its top-level OS shims therefore also run in every spawned worker.
        runpy.run_module("auction_flop.benchmark.run", run_name="__main__", alter_sys=False)
    finally:
        if out is not None and out.is_dir():
            provenance["finished"] = datetime.now(timezone.utc).isoformat()
            (out / "platform-adapter.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n")
