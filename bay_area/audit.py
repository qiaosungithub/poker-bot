"""Recheck retained benchmark scores, source versions, timings and links."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from bay_area.analyze import analyze

ROOT=Path(__file__).resolve().parents[1]


def check_hash(path, expected):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    args=parser.parse_args()
    registry=json.loads((ROOT/"bay_area/candidates.json").read_text())["candidates"]
    checks={}
    for stage in ["smoke","screen","confirmation"]:
        report=analyze(args.folder/stage)
        manifest=json.loads((args.folder/stage/"manifest.json").read_text())
        for name,meta in manifest["candidates"].items():
            assert meta["commit"]==registry[name]["commit"]
            for path,sha in meta["sha256"].items():assert check_hash(ROOT/path,sha),path
        for path,sha in manifest["runner_sha256"].items():assert check_hash(ROOT/path,sha),path
        checks[stage]={"tables":report["tables"],"independent_blocks":report["config"]["blocks"],
                       "raw_scores_recomputed":True,"source_hashes_match":True}
    timing=json.loads((args.folder/"latency/result.json").read_text())
    raw=args.folder/"latency/timings.npz"
    assert check_hash(raw,timing["raw_sha256"])
    assert not check_hash(raw,"0"*64),"Negative hash control failed"
    assert not check_hash(args.folder/"nonexistent-control",timing["raw_sha256"])
    samples=0
    with np.load(raw) as archive:
        for name,result in timing["results"].items():
            source=ROOT/registry[name]["path"]/"agent.py"
            assert check_hash(source,result["agent_sha256"])
            for kind in ["wall","cpu"]:
                values=archive[f"{name}__{kind}"]/1e6
                assert len(values)==result["samples"]
                expected=np.quantile(values,[.5,.99,.999,1])
                assert np.allclose(expected,[result[f"{kind}_ms"][k] for k in ["p50","p99","p999","max"]],rtol=0,atol=1e-12)
            assert int((archive[f"{name}__wall"]>1_000_000).sum())==result["over_1ms"]
            phases=archive[f"{name}__phase"]
            for i,label in enumerate(["preflop","flop","turn","river","select"]):
                assert int((phases==i).sum())==result["phase_counts"][label]
            samples+=result["samples"]
    checks["latency"]={"samples":samples,"raw_quantiles_recomputed":True,"source_hashes_match":True}
    repos={}
    for meta in registry.values():
        directory=ROOT/meta["path"]
        base=subprocess.check_output(["git","-C",str(directory),"rev-parse","--show-toplevel"],text=True).strip()
        actual=subprocess.check_output(["git","-C",base,"rev-parse","HEAD"],text=True).strip()
        assert actual==meta["commit"]
        assert not subprocess.check_output(["git","-C",base,"status","--porcelain"],text=True)
        repos[str(Path(base).relative_to(ROOT))]=actual
    checked_links=0
    audit_path=args.folder/"audit.json"
    link_targets=[]
    for path in [ROOT/"bay_area/README.md",args.folder/"REPORT.md",ROOT/"docs/research/bay-area-benchmark.md"]:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if "://" in link or link.startswith("#"):continue
            target=path.parent/link.split("#")[0]
            assert target.exists() or target.resolve()==audit_path.resolve(),(path,link)
            link_targets.append(target)
            checked_links+=1
    result={"checks":checks,"clean_pinned_repositories":repos,
            "negative_hash_controls_passed":True,"local_links_checked":checked_links,
            "verdict":"pass"}
    audit_path.write_text(json.dumps(result,indent=2)+"\n")
    assert all(path.exists() for path in link_targets)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
