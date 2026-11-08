"""Verify a trainer run's checkpoints against the upstream formats and bot path.

usage: python check_run.py RUN_DIR [--states N]

Checks, per checkpoint iteration:
* ``<run>_adv/iter_X.npz`` has exactly the released ``deep_cfr_model.npz`` keys,
  shapes and dtypes, loads through upstream ``bot.deep_cfr_lookup.DeepCFRLookup``
  as an advantage model and yields finite probability vectors on real six-seat
  states (the retained latency corpus);
* ``<run>/iter_X.npz`` loads as an ``average_strategy`` model and also plays;
* ``<run>/iter_X.pt`` state dicts equal the NPZ arrays bit for bit;
* unit: ``params_from_state_dict`` == upstream ``_extract_dueling_weights`` order.
Exit code 1 on any failure (verdicts are collected, not raised one by one).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import paths

import numpy as np
import torch

RELEASED = paths.SOURCE / "data" / "deep_cfr_model.npz"
CORPUS = paths.REPO / "hu-benchmark" / "multiway_latency_states.jsonl.gz"  # retained latency corpus


def load_states(n: int) -> list[dict]:
    out = []
    with gzip.open(CORPUS, "rt") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--states", type=int, default=256)
    a = ap.parse_args()
    run = Path(a.run_dir)
    adv_dir = Path(str(run) + "_adv")
    from bot.deep_cfr_lookup import DeepCFRLookup, _load_model
    from ckpt import ADV_SD_TO_NPZ, params_from_state_dict
    from training.network import DuelingAdvantageNet
    import common

    fails: list[str] = []
    # unit: parameter mapping
    torch.manual_seed(0)
    net = DuelingAdvantageNet(51, 256, 128)
    p1, p2 = params_from_state_dict(net.state_dict()), common.params_from_torch(net)
    if list(p1) != list(p2) or any(not np.array_equal(p1[k], p2[k]) for k in p1):
        fails.append("params_from_state_dict != common.params_from_torch")

    ref = np.load(RELEASED)
    ref_spec = {k: (ref[k].shape, ref[k].dtype) for k in ref.files}
    states = load_states(a.states) if CORPUS.exists() else []
    if not states:
        fails.append(f"no states corpus at {CORPUS}")
    adv_files = sorted(adv_dir.glob("iter_*.npz"))
    if not adv_files:
        fails.append(f"no advantage NPZ in {adv_dir}")
    for f in adv_files:
        d = np.load(f)
        spec = {k: (d[k].shape, d[k].dtype) for k in d.files}
        if spec != ref_spec:
            fails.append(f"{f.name}: keys/shapes/dtypes differ from released: {spec}")
        if _load_model(f)["_model_type"] != "advantage":
            fails.append(f"{f.name}: not recognised as advantage model")
        look = DeepCFRLookup(f)
        probs = np.array([look.get_strategy(s) for s in states])
        if states and (not np.isfinite(probs).all() or np.abs(probs.sum(1) - 1).max() > 1e-5):
            fails.append(f"{f.name}: bad probabilities")
        it = f.stem
        pt = run / f"{it}.pt"
        if not pt.exists():
            fails.append(f"missing {pt}")
            continue
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        for k, name in ADV_SD_TO_NPZ.items():
            if not np.array_equal(ck["net"][k].numpy(), d[name]):
                fails.append(f"{pt.name}: net[{k}] != {f.name}[{name}]")
        snpz = run / f"{it}.npz"
        if ck.get("strategy_net") is not None:
            if not snpz.exists():
                fails.append(f"missing strategy npz {snpz}")
            else:
                sm = _load_model(snpz)
                if sm["_model_type"] != "average_strategy":
                    fails.append(f"{snpz.name}: not average_strategy")
                if not np.array_equal(ck["strategy_net"]["policy_head.weight"].numpy(),
                                      sm["strategy_out_w"]):
                    fails.append(f"{snpz.name}: policy head != pt")
                sp = np.array([DeepCFRLookup(snpz).get_strategy(s) for s in states[:32]])
                if states and (not np.isfinite(sp).all() or np.abs(sp.sum(1) - 1).max() > 1e-5):
                    fails.append(f"{snpz.name}: bad strategy probabilities")
        print(f"{f.name}: ok-so-far, iteration={ck['iteration']} buffer={ck['buffer_total']} "
              f"strat_fit={ck.get('strategy_fit')} mean max-prob={probs.max(1).mean():.3f}"
              if states else f"{f.name} checked (no states)")
    # negative control: a corrupted copy must be rejected by the key/shape check
    if adv_files:
        d = dict(np.load(adv_files[0]))
        d["trunk_w0"] = d["trunk_w0"][:, :50]
        bad = {k: (v.shape, v.dtype) for k, v in d.items()}
        if bad == ref_spec:
            fails.append("negative control not detected")
    print(json.dumps(dict(run=str(run), adv_npz=len(adv_files), states=len(states),
                          failures=fails), indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
