"""Audit participant logs using only information visible before each action."""

import argparse
from collections import Counter
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT.parent / "starter-kit"
sys.path.insert(0, str(ROOT))
from poker_agent_protocol.game import Action, BlindLevel, Event, GameState, LegalAction, Seat, TableConfig
from poker_engine.table import DEFAULT_BLINDS


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(source):
    reader = load("official_reader", KIT / "hand-history/read.py")
    names = ("icm_v2", "icm_price_v4", "icm_postonly_v4", "icm_texture_v4")
    modules = {name: load(name, ROOT / "bots/dianne" / name / "agent.py") for name in names}
    counters, cases, losses = Counter(), {"cheap_folds": [], "pocket_pair_calls": [], "large_paired_underpair_calls": [], "board_quads": [], "shared_nuts": [], "settlement_mismatches": [], "baseline_mismatches": []}, []
    changes = {name: Counter() for name in names[1:]}
    matches = Counter()
    for header, table, private in reader.read_history(source):
        hero = private["seat"]
        if table["fillers"][hero]:
            counters["filler_tables_skipped"] += 1
            continue
        counters["tables"] += 1
        counters["automatic_decisions"] += len(private["automatic"])
        players = len(table["seats"])
        for hand, hole in zip(table["hands"], private["cards"]):
            if not hole:
                continue
            counters["hands"] += 1
            stacks, bets = list(hand["stacks_before"]), [0] * players
            statuses = ["active" if stack else "busted" for stack in stacks]
            shown = [()] * players
            board, auction, history, decisions = (), (), [Event("hand", amount=hand["hand"])], []
            pot, last_raise = 0, hand["blinds"][1]
            acted = [None] * players
            own_raised = False
            config = TableConfig(str(table["iteration"]) + ":" + str(table["table"]), hero, players,
                                 "boston", 2000, DEFAULT_BLINDS, 20, 250,
                                 tuple(table["payouts"]), 1, 65536, 100)

            def close_street():
                nonlocal pot, bets, last_raise, acted
                ordered = sorted(bets, reverse=True)
                refund = ordered[0] - ordered[1]
                if refund:
                    seat = bets.index(ordered[0])
                    stacks[seat] += refund
                    pot -= refund
                    if statuses[seat] == "all_in":
                        statuses[seat] = "active"
                bets = [0] * players
                acted = [None] * players
                last_raise = hand["blinds"][1]

            for index, raw in enumerate(hand["events"]):
                kind, seat, amount = raw["kind"], raw.get("seat"), raw.get("amount", 0)
                if seat == hero and kind in ("call", "fold", "raise", "check"):
                    highest = max(bets)
                    to_call = highest - bets[hero]
                    legal = [LegalAction("fold"), LegalAction("call")] if to_call else [LegalAction("check")]
                    other_active = any(i != hero and statuses[i] == "active" for i in range(players))
                    reopened = acted[hero] is None or highest - acted[hero] >= last_raise
                    maximum = bets[hero] + stacks[hero]
                    if other_active and reopened and maximum > highest:
                        minimum = min(maximum, max(highest + last_raise, hand["blinds"][1]))
                        legal.append(LegalAction("raise", minimum, maximum))
                    balances = hand.get("flop_chips_after" if board else "flop_chips_before", [1000] * players)
                    state = GameState(tuple(history), hand["hand"], hero, tuple(hole), board, pot, to_call,
                                      tuple(legal), tuple(history),
                                      tuple(Seat(i, stacks[i], bets[i], statuses[i], shown[i]) for i in range(players)),
                                      hand["button"], max(0, (hand["hand"]-1)//20), BlindLevel(*hand["blinds"]),
                                      20-(hand["hand"]-1)%20, sum(value > 0 for value in hand["stacks_before"]), (), config,
                                      {"auction_cards": list(auction), "flop_chips": balances[hero],
                                       "flop_chips_by_seat": balances}, "betting", 100)
                    action_amount = amount if kind == "raise" else 0
                    predicted = {}
                    for name, module in modules.items():
                        try:
                            answer = module.Agent().act(state, b"")[0]
                            predicted[name] = {"kind": answer.kind, "amount": answer.amount}
                        except Exception as error:
                            predicted[name] = {"error": repr(error)}
                    baseline_match = predicted["icm_v2"] == {"kind": kind, "amount": action_amount}
                    counters["decisions"] += 1
                    counters["baseline_exact_matches"] += baseline_match
                    matches[f"{len(board)}:{kind}:{'match' if baseline_match else 'mismatch'}"] += 1
                    cost = min(to_call, stacks[hero])
                    cap = bets[hero] + cost
                    eligible = max(0, pot - sum(max(0, bet - cap) for bet in bets))
                    odds = cost / max(1, eligible + cost)
                    record = {"iteration": table["iteration"], "table": table["table"], "hand": hand["hand"],
                              "event_index": index, "cards": hole, "board": board, "auction_cards": auction,
                              "pot": pot, "stack": stacks[hero], "bet": bets[hero], "to_call": to_call,
                              "call_cost": cost, "eligible_pot": eligible, "pot_odds": odds,
                              "actual": {"kind": kind, "amount": action_amount}, "paid_call": amount if kind == "call" else 0,
                              "baseline_reproduced": baseline_match,
                              "candidates": predicted}
                    decisions.append(record)
                    if not baseline_match and len(cases["baseline_mismatches"]) < 5:
                        cases["baseline_mismatches"].append(record)
                    if baseline_match:
                        for name in names[1:]:
                            if predicted[name] != predicted["icm_v2"]:
                                changes[name][f"{len(board)}:{kind}->{predicted[name].get('kind','error')}"] += 1
                    categories = []
                    if kind == "fold" and own_raised and 0 < odds <= 0.10:
                        categories.append("cheap_folds")
                    if board:
                        quality, category = modules["icm_v2"].hand_quality(tuple(hole), board)
                        record.update(heuristic_quality=quality, hand_category=category)
                        if kind == "call" and hole[0][0] == hole[1][0] and category <= 2:
                            categories.append("pocket_pair_calls")
                        ranks = [card[0] for card in board]
                        if (kind == "call" and category == 2 and hole[0][0] == hole[1][0]
                                and any(modules["icm_v2"].RANKS[rank] > modules["icm_v2"].RANKS[hole[0][0]] for rank in ranks)
                                and cost >= stacks[hero] * 0.30):
                            categories.append("large_paired_underpair_calls")
                        if any(ranks.count(rank) == 4 for rank in ranks):
                            categories.append("board_quads")
                        if len(board) == 5 and set(ranks) == set("TJQKA"):
                            suits = [card[1] for card in board]
                            if len(set(suits)) == 1 or max(suits.count(s) for s in suits) < 3:
                                categories.append("shared_nuts")
                    for category in categories:
                        counters[category] += 1
                        if len(cases[category]) < 20:
                            cases[category].append(record)
                if kind in ("auction", "board"):
                    close_street()
                    if kind == "auction":
                        auction = tuple(raw["cards"])
                    else:
                        board += tuple(raw["cards"])
                elif kind in ("ante", "blind", "call", "raise"):
                    delta = amount - bets[seat] if kind == "raise" else amount
                    if kind == "raise":
                        increase = amount - max(bets)
                        if increase >= last_raise:
                            last_raise = increase
                    stacks[seat] -= delta
                    pot += delta
                    if kind != "ante":
                        bets[seat] += delta
                    if stacks[seat] == 0:
                        statuses[seat] = "all_in"
                    if kind in ("call", "raise"):
                        acted[seat] = max(bets)
                    own_raised |= seat == hero and kind == "raise"
                elif kind == "check":
                    acted[seat] = max(bets)
                elif kind == "fold":
                    statuses[seat] = "folded"
                elif kind == "show":
                    shown[seat] = tuple(raw.get("cards", ()))
                elif kind == "win":
                    # Uncontested awards include the winner's unmatched bet.
                    # Showdown awards can instead follow an implicit refund.
                    if sum(status in ("active", "all_in") for status in statuses) >= 2:
                        close_street()
                    stacks[seat] += amount
                elif kind == "stack":
                    if stacks[seat] != amount and len(cases["settlement_mismatches"]) < 5:
                        cases["settlement_mismatches"].append({"predicted": stacks[seat], "recorded": amount, "seat": seat,
                            "stacks_before": hand["stacks_before"], "events": hand["events"]})
                    counters["settlement_stack_mismatches"] += stacks[seat] != amount
                    stacks[seat] = amount
                history.append(Event(kind, seat, amount, tuple(raw.get("cards", ()))))
            loss = hand["stacks_before"][hero] - hand["stacks_after"][hero]
            if loss > 0 and decisions:
                item = {"iteration": table["iteration"], "table": table["table"], "hand": hand["hand"],
                        "chip_loss": loss, "icm_loss": hand["equity_before"][hero]-hand["equity_after"][hero],
                        "decisions": decisions}
                key = (loss, table["iteration"], table["table"], hand["hand"], item)
                if len(losses) < 20:
                    heapq.heappush(losses, key)
                elif key[:4] > losses[0][:4]:
                    heapq.heapreplace(losses, key)
    return {"source": str(source), "counts": counters, "baseline_match_detail": matches,
            "candidate_changes_on_reproduced_states": changes, "cases": cases,
            "largest_chip_losses": [item[-1] for item in sorted(losses, reverse=True)],
            "source_hashes": {name: hashlib.sha256((ROOT / "bots/dianne" / name / "agent.py").read_bytes()).hexdigest()
                              for name in names},
            "method": "Stream kit reader. Reconstruct public bets/stacks, cap uncalled street excess, use private own cards only. Outcomes rank cases after all decisions; never feed outcomes/reveals backward.",
            "limitations": "History omits actual decision time bank and full table configuration. Use kit defaults for omitted config. Counterfactual changes counted only where uploaded icm_v2 exactly reproduces recorded action. Range quality is heuristic, and a loss alone does not establish an error."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "candidate_changes_on_reproduced_states")}, indent=2))
