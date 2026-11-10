"""Validate phase composition on real released-engine trajectories, without scoring.

This is an interface/equivalence regression, not an auction-flop-v1 strength run.
Only two donor files may differ from each target: independent_preflop.py and
distilled_prior.py.  Candidate and frozen benchmark sources are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = ROOT / ".runtime/main-benchmark-c62cdac/auction_flop"
PAIRS = (("sqa_prior_classic_v1", "classic_hybrid_v1"),
         ("sqa_prior_fragile_v1", "fragile_both_v1"))
DONOR = "practice_prior_full_v1"
VARIANT = "flop_auction_holdem"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(folder):
    return {p.name: digest(p) for p in sorted(folder.glob("*.py"))}


def load_isolated(folder, identity):
    """Match the benchmark's isolation of top-level local policy imports."""
    local = {p.stem for p in folder.glob("*.py")}
    old_modules = {n: sys.modules.pop(n) for n in local if n in sys.modules}
    old_path = sys.path[:]
    sys.path.insert(0, str(folder))
    name = "composition_validation_" + identity
    spec = importlib.util.spec_from_file_location(name, folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        helpers = {n: sys.modules[n] for n in local if n in sys.modules}
        return module.Agent, helpers
    finally:
        sys.path[:] = old_path
        for n in local:
            sys.modules.pop(n, None)
        sys.modules.update(old_modules)


def check_action(state, answer, reply, action_from_dict):
    raw, memory = reply(answer)
    action = action_from_dict(raw)
    assert isinstance(memory, bytes), "SDK memory must be bytes"
    assert len(memory) <= min(state.config.memory_bytes, 8192), "memory bound exceeded"
    legal = state.can(action.kind)
    assert legal is not None, ("missing legal action", action)
    if action.kind == "raise":
        assert type(action.amount) is int
        assert legal.min_amount <= action.amount <= legal.max_amount, (action, legal)
    if action.kind == "bid":
        assert len(action.cards) == 5
        assert all(type(x) is int and x >= 0 for x in action.cards)
        assert sum(action.cards) <= legal.max_amount
    return action, memory


def prior_gate(state, donor_helpers):
    """Classify the actual donor's narrow override; no invented state interface."""
    pre = donor_helpers["independent_preflop"]
    prior = donor_helpers["distilled_prior"]
    *_, raises, limpers, effective = pre._context(state)
    if effective <= 10:
        return "short_stack"
    if raises or limpers:
        return "raised_or_limped"
    if any(e.kind in ("raise", "call") or
           (e.seat == state.seat and e.kind in ("check", "fold"))
           for e in state.history):
        return "already_acted"
    if prior.runtime_context(state, effective) not in prior.OPEN_PROBABILITY:
        return "missing_prior_context"
    return "applicable"


def run(args):
    snapshot = args.snapshot.resolve()
    sdk = snapshot / "vendor/sdk"
    spec_file = snapshot / "benchmark/v1.json"
    assert digest(spec_file) == (snapshot / "benchmark/v1.sha256").read_text().strip()
    frozen = json.loads(spec_file.read_text())
    sdk_checked = 0
    for file, expected in frozen["sha256"].items():
        if file.startswith("vendor/sdk/"):
            assert digest(snapshot / file) == expected, ("changed frozen SDK", file)
            sdk_checked += 1
    assert sdk_checked >= 20
    for package in ("agent-protocol", "engine", "agent-sdk"):
        location = sdk / package / "src"
        if not location.is_dir():
            raise FileNotFoundError(f"Frozen SDK extraction is not ready: {location}")
        sys.path.insert(0, str(location))
    from poker_agent_protocol.game import Action, action_from_dict
    from poker_agent_sdk.guest import reply
    from poker_engine.table import SitAndGo, TableRules
    from poker_engine.variant import load_variant

    started = time.monotonic()
    folders = {name: ROOT / "bots/dianne" / name for name, _ in PAIRS}
    folders.update({name: snapshot / "candidates" / name for _, name in PAIRS})
    folders[DONOR] = snapshot / "candidates" / DONOR
    source_hashes = {name: sources(folder) for name, folder in folders.items()}
    allowed = {"independent_preflop.py", "distilled_prior.py"}
    for candidate, baseline in PAIRS:
        expected = dict(source_hashes[baseline])
        for file in allowed:
            expected[file] = source_hashes[DONOR][file]
        assert source_hashes[candidate] == expected, (
            "Candidate must be target plus exactly two donor files", candidate)
    loaded = {name: load_isolated(folder, name) for name, folder in folders.items()}
    donor_helpers = loaded[DONOR][1]
    counts = Counter()
    per_candidate = {name: Counter() for name, _ in PAIRS}
    examples = {}
    max_memory = {name: 0 for name, _ in PAIRS}
    configurations = []
    phase_names = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}

    def validate(state, name, memory):
        baseline = dict(PAIRS)[name]
        agent, helpers = loaded[name]
        candidate_answer = agent().act(state, memory)
        action, updated = check_action(state, candidate_answer, reply, action_from_dict)
        expected = loaded[baseline][0]().act(state, memory)
        check_action(state, expected, reply, action_from_dict)
        tally = per_candidate[name]
        phase = "auction" if state.phase == "auction" else phase_names[len(state.board)]
        tally[phase] += 1
        if phase != "preflop":
            assert candidate_answer == expected, ("nonpreflop changed", name, phase, state)
            tally["nonpreflop_exact_action_memory"] += 1
        else:
            donor_answer = loaded[DONOR][0]().act(state, memory)
            check_action(state, donor_answer, reply, action_from_dict)
            assert action == donor_answer[0], ("preflop donor mismatch", name, state)
            tally["preflop_exact_donor_action"] += 1
            gate = prior_gate(state, donor_helpers)
            tally["prior_" + gate] += 1
            if gate != "applicable":
                assert candidate_answer == expected, ("out-of-scope change", name, gate, state)
                tally["unaffected_preflop_exact_action_memory"] += 1
            elif action != expected[0]:
                tally["changed_opening_actions"] += 1
                examples.setdefault(name, {
                    "hand": state.hand_number, "players_remaining": state.players_remaining,
                    "seat": state.seat, "cards": list(state.cards),
                    "before": asdict(expected[0]), "after": asdict(action)})
        assert agent().act(state, memory) == candidate_answer, "same-input replay changed"
        _, prepared = helpers["public_features"].prepare(state, updated)
        assert helpers["public_features"].finish(state, action, prepared) == updated, (
            "memory recorded the same decision twice", name)
        if "classic_policy" in helpers:
            decoded = json.loads(updated)
            assert "classic" in decoded
            tally["classic_profile_preserved"] += 1
        max_memory[name] = max(max_memory[name], len(updated))
        return action, updated

    # These are legal engine configurations chosen for coverage, not rankings.
    # Short starts guarantee the prior's <=10BB guard is exercised immediately.
    for players in (2, 3, 8):
        for stack in (100, 300, 2000):
            for repetition in range(args.repetitions):
                seed = f"sqa-composition-regression:{players}:{stack}:{repetition}"
                rules = TableRules(variant=VARIANT, players=players, starting_stack=stack,
                                   hand_cap=80, decision_ms=60000, time_bank_ms=0)
                table = SitAndGo(rules=rules, variant=load_variant(VARIANT),
                                 table_id=seed, seed=seed)
                # Independent memories preserve both candidate schemas. The SDK
                # reconciles proposed actions against the actual public events.
                shadows = {name: [b""] * players for name, _ in PAIRS}
                first = True
                table_states = 0
                while (state := table.decision()) is not None:
                    candidates = {}
                    for name, _ in PAIRS:
                        memory = shadows[name][state.seat]
                        candidates[name] = validate(state, name, memory)
                        shadows[name][state.seat] = candidates[name][1]
                        if first:
                            fresh = loaded[name][0]().act(state, b"")
                            for bad in (b"not json", b"\xff", b"x" * 8193):
                                assert loaded[name][0]().act(state, bad) == fresh
                                counts["malformed_memory_resets"] += 1
                    first = False
                    # Mix real composed policies with legal passive and min-raise
                    # policies. Passive seats also create limped/checked histories.
                    mode = (state.seat + repetition) % 4
                    if mode < 2:
                        chosen = candidates[PAIRS[mode][0]][0]
                    elif state.phase == "auction":
                        chosen = Action("bid", cards=(0, 0, 0, 0, 0))
                    elif mode == 3 and state.can("raise") is not None:
                        chosen = Action.raise_to(state.can("raise").min_amount)
                    elif state.can("check") is not None:
                        chosen = Action.check()
                    elif state.can("call") is not None:
                        chosen = Action.call()
                    else:
                        chosen = Action.fold()
                    assert table.apply(chosen, answered=True, bank_ms=0.0), (
                        "engine rejected trajectory action", seed, state, chosen)
                    phase = "auction" if state.phase == "auction" else phase_names[len(state.board)]
                    counts["states"] += 1
                    counts[phase] += 1
                    counts[f"players_remaining_{state.players_remaining}"] += 1
                    table_states += 1
                assert sum(table.result((0,) * players).final_stacks) == players * stack
                configurations.append({"players": players, "starting_stack": stack,
                                       "seed": seed, "states": table_states})
    for phase in ("preflop", "auction", "flop", "turn", "river"):
        assert counts[phase] > 0, ("missing phase coverage", phase)
    assert counts["states"] >= 1000, "insufficient state coverage"
    for name, _ in PAIRS:
        for key in ("prior_short_stack", "prior_raised_or_limped", "prior_missing_prior_context",
                    "changed_opening_actions", "nonpreflop_exact_action_memory"):
            assert per_candidate[name][key] > 0, ("missing boundary coverage", name, key)
    return {"status": "passed", "kind": "real-engine composition regression, not benchmark scoring",
            "snapshot": str(snapshot), "snapshot_commit": "c62cdac",
            "variant": VARIANT, "tables": len(configurations),
            "frozen_sdk_files_verified": sdk_checked,
            "coverage": dict(counts), "per_candidate": {k: dict(v) for k, v in per_candidate.items()},
            "maximum_memory_bytes": max_memory, "changed_action_examples": examples,
            "mismatches": 0, "illegal_actions": 0, "source_hashes": source_hashes,
            "configurations": configurations, "seconds": time.monotonic() - started,
            "limitations": ["Finite trajectory coverage, not proof for every possible state.",
                            "Custom legal engine configurations cover boundary cases; no strength score is produced.",
                            "Direct SDK serialization checks do not qualify production IPC latency.",
                            "Nonpreflop equivalence is conditional on identical input state and memory; changed openings change future trajectories."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/sqa-composition-validation.json")
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_hashes", "configurations")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
