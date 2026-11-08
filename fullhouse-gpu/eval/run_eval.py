"""Six-player selection / confirmation protocol for a trainer run (mirrors
``evaluate`` + ``paired`` of the original run's ``fullhouse-repro/supervise.py``, on
``main``). It drives the repository's own, unmodified benchmark scripts:
``fullhouse-repro/export_validate.py`` and ``fullhouse-repro/evaluate.py`` (which use
``hu-benchmark/`` and ``latency-survey/``).

  selection    : 256 seed blocks from 11,300,000, every advantage NPZ of the run
  confirmation : 1,024 seed blocks from 12,300,000, for the best selection
                 checkpoint and the final one; reference on the same seeds
  reference    : evaluate.py without --onnx = latency-survey/fullhouse_fp32.onnx,
                 the FP32 ONNX export of the released deep_cfr_model.npz
Paired statistic: per seed block, candidate block mean minus reference block mean;
95% CI = mean +/- 1.96 * stdev / sqrt(blocks).  Success: lower bound > -2.

Environment: EVAL_PY = python of the pinned eval environment (default: this python);
FULLHOUSE_REPRO = the fullhouse-repro directory (default: <repo>/fullhouse-repro).

usage:
  python run_eval.py export   CKPT.npz [...]          # ONNX export + 1,024-state validation
  python run_eval.py arena    OUT_DIR --split selection|confirmation [--onnx X.onnx] [--tag T]
  python run_eval.py protocol RUN_ADV_DIR OUT_DIR [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent                     # fullhouse-gpu/eval
REPRO = Path(os.environ.get("FULLHOUSE_REPRO", str(KIT.parents[1] / "fullhouse-repro")))
PY = os.environ.get("EVAL_PY", sys.executable)
SPLITS = {"selection": (11_300_000, 256), "confirmation": (12_300_000, 1024)}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(argv: list[str], log: Path) -> None:
    with open(log, "a") as f:
        subprocess.run(argv, cwd=REPRO, stdout=f, stderr=subprocess.STDOUT, check=True)


def export(ckpt: Path, out_dir: Path) -> Path:
    onnx = out_dir / (ckpt.stem + ".onnx")
    if not Path(str(onnx) + ".validation.json").exists():
        run([PY, "export_validate.py", str(ckpt), str(onnx)], out_dir / (ckpt.stem + "_export.log"))
    return onnx


def arena(out: Path, split: str, onnx: Path | None, workers: int) -> dict:
    seed, blocks = SPLITS[split]
    if not out.exists():
        argv = [PY, "evaluate.py", "--output", str(out), "--seed", str(seed), "--blocks",
                str(blocks), "--workers", str(workers)]
        if onnx is not None:
            argv[2:2] = ["--onnx", str(onnx)]
        t0 = time.time()
        run(argv, out.with_suffix(".log"))
        print(f"  arena {out.name}: {time.time() - t0:.1f}s", flush=True)
    return json.loads(out.read_text())


def block_scores(out: Path) -> dict[int, float]:
    rows = map(json.loads, out.with_suffix(".jsonl").read_text().splitlines())
    return {r["seed"]: r["scores"]["fullhouse"] for r in rows}


def paired(candidate: Path, reference: Path) -> dict:
    c, r = block_scores(candidate), block_scores(reference)
    assert c.keys() == r.keys(), "candidate and reference seed blocks differ"
    diffs = [c[k] - r[k] for k in c]
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs) / len(diffs) ** 0.5
    return dict(mean=mean, se=se, ci95=[mean - 1.96 * se, mean + 1.96 * se], blocks=len(diffs))


def evaluate(ckpt: Path, split: str, out_dir: Path, workers: int, results: Path) -> dict:
    onnx = export(ckpt, out_dir)
    ref = out_dir / f"reference_{split}.json"
    arena(ref, split, None, workers)
    out = out_dir / f"{ckpt.stem}_{split}.json"
    data = arena(out, split, onnx, workers)
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), checkpoint=str(ckpt),
               checkpoint_sha256=sha256(ckpt), split=split, score=data["results"]["fullhouse"],
               reference=json.loads(ref.read_text())["results"]["fullhouse"],
               difference=paired(out, ref), hands=data["hands"], errors=data["errors"],
               elapsed_seconds=data["elapsed_seconds"], artifact=str(out))
    with open(results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    d = rec["difference"]
    print(f"{split:12s} {ckpt.stem}: {rec['score']['mean']:.3f} "
          f"[{rec['score']['ci95'][0]:.2f},{rec['score']['ci95'][1]:.2f}]  ref "
          f"{rec['reference']['mean']:.3f}  diff {d['mean']:+.3f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]",
          flush=True)
    return rec


def protocol(adv_dir: Path, out_dir: Path, workers: int, every: int = 10) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    ckpts = sorted(p for p in adv_dir.glob("iter_*.npz") if int(p.stem[5:]) % every == 0)
    assert ckpts, f"no checkpoints in {adv_dir}"
    final = max(ckpts, key=lambda p: int(p.stem[5:]))
    sel = [evaluate(c, "selection", out_dir, workers, results) for c in ckpts]
    best = max(sel, key=lambda r: r["score"]["mean"])
    conf = {}
    for label, path in (("selected", Path(best["checkpoint"])), ("final_budget", final)):
        conf[label] = evaluate(path, "confirmation", out_dir, workers, results)
    passed = conf["selected"]["difference"]["ci95"][0] > -2
    status = dict(state="complete" if passed else "criterion_failed", criterion_passed=passed,
                  selection=best, confirmation=conf,
                  selection_table=[dict(checkpoint=Path(r["checkpoint"]).stem,
                                        mean=r["score"]["mean"], ci95=r["score"]["ci95"],
                                        diff=r["difference"]) for r in sel])
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("ckpts", nargs="+"); e.add_argument("--out-dir", required=True)
    a = sub.add_parser("arena"); a.add_argument("out"); a.add_argument("--split", required=True)
    a.add_argument("--onnx"); a.add_argument("--workers", type=int, default=46)
    p = sub.add_parser("protocol"); p.add_argument("adv_dir"); p.add_argument("out_dir")
    p.add_argument("--workers", type=int, default=46)
    args = ap.parse_args()
    if args.cmd == "export":
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        for c in args.ckpts:
            print(export(Path(c).resolve(), out.resolve()))
    elif args.cmd == "arena":
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(arena(out, args.split, Path(args.onnx).resolve() if args.onnx else None,
                               args.workers)["results"]["fullhouse"]))
    else:
        st = protocol(Path(args.adv_dir).resolve(), Path(args.out_dir).resolve(), args.workers)
        print(json.dumps({k: st[k] for k in ("state", "criterion_passed")}))


if __name__ == "__main__":
    main()
