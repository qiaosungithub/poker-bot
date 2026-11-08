"""Isolated CPU timing of unchanged finalist act functions on played game states."""
from __future__ import annotations

import argparse
from array import array
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from bay_area.environment import BayAreaEnv
from bay_area.seeded_guest import SeededAgent

ROOT = Path(__file__).resolve().parents[1]


class TimedDelegate:
    def __init__(self, agent):
        self.agent = agent
        self.wall = array("Q")
        self.cpu = array("Q")
        self.phase = array("B")

    def act(self, state, memory):
        cpu0 = time.thread_time_ns()
        wall0 = time.perf_counter_ns()
        try:
            return self.agent.act(state, memory)
        finally:
            self.wall.append(time.perf_counter_ns()-wall0)
            self.cpu.append(time.thread_time_ns()-cpu0)
            self.phase.append(4 if state.phase=="select" else {0:0,3:1,4:2,5:3}[len(state.board)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"bay_area/configs/confirmation.json")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--blocks",type=int,default=16)
    parser.add_argument("--cpu",type=int,default=32)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    registry=json.loads((ROOT/"bay_area/candidates.json").read_text())["candidates"]
    names=json.loads(args.config.read_text())["candidates"]
    assert len(names)==8
    args.out.mkdir(parents=True,exist_ok=False)
    modules={}
    for name in names:
        folder=ROOT/registry[name]["path"]
        spec=importlib.util.spec_from_file_location("latency_"+name,folder/"agent.py")
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        sys.path.insert(0,str(folder))
        try: spec.loader.exec_module(module)
        finally: sys.path.pop(0)
        modules[name]=module
    measured={n:{"wall":array("Q"),"cpu":array("Q"),"phase":array("B")} for n in names}
    failures={n:0 for n in names}
    gc.disable()
    start=time.perf_counter()
    for block in range(args.blocks+1):
        order=names.copy()
        random.Random(f"latency-order:{block}").shuffle(order)
        for rotation in range(8):
            lineup=order[rotation:]+order[:rotation]
            wrappers=[]
            for name in lineup:
                wrapper=SeededAgent(modules[name],ROOT/registry[name]["path"])
                wrapper.agent=TimedDelegate(wrapper.agent)
                wrappers.append(wrapper)
            env=BayAreaEnv(f"bay-area-latency-20261108:{block}")
            while env.state:
                state=env.state;seat=state.seat
                try:
                    result=wrappers[seat].act(state,env.memory[seat])
                    _,reason=env.step(result)
                except Exception:
                    _,reason=env.step(crashed=True)
                failures[lineup[seat]] += reason is not None
            if block:
                for name,wrapper in zip(lineup,wrappers):
                    for key in ("wall","cpu","phase"):
                        measured[name][key].extend(getattr(wrapper.agent,key))
        print(json.dumps({"block":block,"blocks":args.blocks,"seconds":round(time.perf_counter()-start,2)}),flush=True)
    arrays={f"{name}__{key}":np.asarray(values) for name,data in measured.items() for key,values in data.items()}
    np.savez_compressed(args.out/"timings.npz",**arrays)
    results={}
    for name,data in measured.items():
        wall=np.asarray(data["wall"])/1e6;cpu=np.asarray(data["cpu"])/1e6
        results[name]={"samples":len(wall),"wall_ms":dict(zip(["p50","p99","p999","max"],map(float,np.quantile(wall,[.5,.99,.999,1])))),
                       "cpu_ms":dict(zip(["p50","p99","p999","max"],map(float,np.quantile(cpu,[.5,.99,.999,1])))),
                       "over_1ms":int((wall>1).sum()),"failures":failures[name],
                       "phase_counts":{label:int((np.asarray(data["phase"])==i).sum()) for i,label in enumerate(["preflop","flop","turn","river","select"])},
                       "agent_sha256":registry[name]["agent_sha256"]}
    result={"cpu":args.cpu,"blocks":args.blocks,"tables_measured":args.blocks*8,
            "warmup_tables":8,"gc":"disabled","time_scope":"Unmodified source Agent.act only; PRNG seeding, SDK conversion, IPC and startup excluded.",
            "rules":"Bay Area defaults; no decision timeouts imposed while collecting timing samples.",
            "results":results,"elapsed_seconds":time.perf_counter()-start,
            "raw_sha256":hashlib.sha256((args.out/"timings.npz").read_bytes()).hexdigest()}
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
