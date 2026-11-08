"""Audit raw tables and summarize seed-block estimates, not correlated rotations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, stdev


def independent_points(stacks, bust_order, ties, payouts):
    """Small exact ICM oracle, independent of the engine's implementation."""
    n = len(stacks)
    result = [0.0] * n
    groups = []
    for i, seat in enumerate(bust_order):
        if i in ties:
            groups[-1].append((i, seat))
        else:
            groups.append([(i, seat)])
    for group in groups:
        share = sum(payouts[n-1-i] if n-1-i < len(payouts) else 0 for i,_ in group) / len(group)
        for _,seat in group:
            result[seat] = share
    live = tuple(i for i,s in enumerate(stacks) if s > 0)

    def walk(remaining, place, probability):
        if not remaining or place >= len(payouts) or not any(payouts[place:len(live)]):
            return
        total = sum(stacks[i] for i in remaining)
        for seat in remaining:
            chance = probability * stacks[seat] / total
            result[seat] += chance * payouts[place]
            walk(tuple(i for i in remaining if i != seat), place+1, chance)
    walk(live, 0, 1.0)
    return result


def interval(values, z=1.959963984540054):
    mean = fmean(values)
    se = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"mean": mean, "standard_error": se,
            "interval": [mean-z*se, mean+z*se] if se is not None else [None,None]}


def analyze(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    raw = (folder / "tables.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"], "Raw hash mismatch"
    rows = [json.loads(line) for line in raw.splitlines()]
    names = manifest["config"]["candidates"]
    blocks = manifest["config"]["blocks"]
    assert manifest["completed_blocks"] == blocks
    assert len(rows) == blocks * len(names)
    seen, grouped, stats = set(), defaultdict(list), defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = row["block"],row["group"],row["rotation"]
        assert key not in seen
        seen.add(key)
        entries = row["entries"]
        assert len(entries) == 8 and len({e["name"] for e in entries}) == 8
        assert [e["seat"] for e in entries] == list(range(8))
        stacks = [e["final_stack"] for e in entries]
        oracle = independent_points(stacks,row["bust_order"],row["ties"],row["payouts"])
        assert sum(stacks) == 16000
        assert abs(sum(e["points"] for e in entries)-100) < 1e-7
        for entry, expected in zip(entries,oracle):
            assert abs(entry["points"]-expected) < 1e-7
            assert 0 <= entry["failures"] <= entry["decisions"]
            name = entry["name"]
            assert name in names
            grouped[name,row["block"]].append(entry["points"])
            stats[name]["tables"] += 1
            stats[name]["failures"] += entry["failures"]
            stats[name]["decisions"] += entry["decisions"]
            stats[name]["outright_wins"] += entry["outright_win"]
            stats[name]["positive_points"] += entry["points"] > 0
            stats[name][f"seat_{entry['seat']}"] += 1
            stats[name]["hand_cap_tables"] += row["hands"] >= manifest["config"]["hand_cap"]
    block_means = {}
    scores = []
    for name in names:
        assert all(len(grouped[name,b]) == 8 for b in range(blocks))
        assert all(stats[name][f"seat_{s}"] == blocks for s in range(8))
        values = [fmean(grouped[name,b]) for b in range(blocks)]
        block_means[name] = values
        estimate = interval(values)
        scores.append({"name":name,"source_group":manifest["candidates"][name]["source_group"],
                       "mean_points":estimate["mean"],"ci95":estimate["interval"],
                       "standard_error":estimate["standard_error"],"independent_blocks":blocks,
                       **dict(stats[name]),
                       "failure_rate":stats[name]["failures"]/max(1,stats[name]["decisions"])})
    scores.sort(key=lambda x:-x["mean_points"])
    pairs = []
    count = len(names)*(len(names)-1)//2
    simultaneous_z = NormalDist().inv_cdf(1-0.05/(2*count))
    for a,b in itertools.combinations([s["name"] for s in scores],2):
        difference = [x-y for x,y in zip(block_means[a],block_means[b])]
        estimate = interval(difference)
        adjusted = interval(difference,simultaneous_z)
        pairs.append({"a":a,"b":b,"mean_a_minus_b":estimate["mean"],
                      "ci95":estimate["interval"],"simultaneous_ci95":adjusted["interval"]})
    best = scores[0]["name"]
    unresolved = [p["b"] for p in pairs if p["a"]==best and
                  (p["simultaneous_ci95"][0] is None or p["simultaneous_ci95"][0] <= 0)]
    report = {"config":manifest["config"],"tables":len(rows),
              "hands":sum(r["hands"] for r in rows),"scores":scores,"pairs":pairs,
              "highest_sample_mean":best,"not_separated_from_leader":unresolved,
              "simultaneous_pair_count":count,"simultaneous_z":simultaneous_z,
              "interval_method":"Normal approximation on independent eight-rotation block means; paired block differences; Bonferroni over every candidate pair.",
              "all_raw_records_passed_audit":True,"raw_sha256":manifest["raw_sha256"],
              "elapsed_seconds":manifest["elapsed_seconds"]}
    (folder / "summary.json").write_text(json.dumps(report,indent=2)+"\n")
    with (folder/"table.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["bot","source_group","mean_points","ci95_low","ci95_high","tables","decisions","failures","failure_rate"])
        for s in scores:
            writer.writerow([s["name"],s["source_group"],s["mean_points"],*s["ci95"],s["tables"],s["decisions"],s["failures"],s["failure_rate"]])
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    args=parser.parse_args()
    report=analyze(args.folder)
    print(json.dumps({"tables":report["tables"],"highest_sample_mean":report["highest_sample_mean"],
        "scores":[{k:s[k] for k in ("name","mean_points","ci95","failures","decisions")} for s in report["scores"]],
        "not_separated_from_leader":report["not_separated_from_leader"]},indent=2))
