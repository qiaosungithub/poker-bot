"""Inspect public decisions and settled hand records in full Boston tables."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import statistics
import sys

from benchmark import BayAreaEnv, VARIANT, load_agent


def snapshot(state, action):
    own = state.seats[state.seat]
    cost = min(state.to_call, own.stack)
    committed = cost if action.kind == "call" else max(0, action.amount - own.bet)
    return {
        "hand": state.hand_number, "phase": state.phase,
        "cards": state.cards, "board": state.board, "pot": state.pot,
        "stack": own.stack, "bet": own.bet, "to_call": state.to_call,
        "players_remaining": state.players_remaining,
        "big_blind": state.blinds.big_blind,
        "action": {"kind": action.kind, "amount": action.amount, "cards": action.cards},
        "commit_fraction": committed / max(1, own.stack),
        "legal_actions": [[item.kind, item.min_amount, item.max_amount] for item in state.legal_actions],
        "seats": [[seat.seat, seat.stack, seat.bet, seat.status, seat.shown_cards] for seat in state.seats],
        "auction_cards": state.variant.get("auction_cards", []),
        "flop_chips": state.variant.get("flop_chips", 1000),
        "recent_public_events": [[event.kind, event.seat, event.amount, event.cards] for event in state.history[-8:]],
    }


def diagnose_block(task):
    folders, focus, block, seed = task
    constructors = {folder.name: load_agent(folder) for folder in folders}
    order = list(constructors)
    random.Random(f"{seed}:seating:{block}").shuffle(order)
    records = []
    for rotation in range(len(order)):
        names = order[rotation:] + order[:rotation]
        hero = names.index(focus)
        agents = [constructors[name]() for name in names]
        env = BayAreaEnv(f"{seed}:{block}", players=8, hand_cap=250, variant=VARIANT)
        action_counts, danger_counts = Counter(), Counter()
        own_hands, dangerous, cheap_folds = {}, [], []
        bid_totals, errors = [], []
        while env.state is not None:
            state = env.state
            try:
                answer = agents[state.seat].act(state, env.memory[state.seat])
            except Exception as error:
                errors.append({"bot": names[state.seat], "hand": state.hand_number, "error": repr(error)})
                env.step(crashed=True)
                continue
            action = answer[0]
            if state.seat == hero:
                phase = "auction" if state.phase == "auction" else ("preflop", "flop", "turn", "river")[(0, 3, 4, 5).index(len(state.board))]
                action_counts[f"{phase}:{action.kind}"] += 1
                observed = snapshot(state, action)
                own_hands.setdefault(state.hand_number, []).append(observed)
                if action.kind == "bid":
                    bid_totals.append(sum(action.cards))
                elif action.kind in ("call", "raise") and observed["commit_fraction"] >= 0.33:
                    danger_counts[f"{phase}:{action.kind}"] += 1
                    dangerous.append(observed)
                if action.kind == "fold" and state.can("call"):
                    call_cost = min(state.to_call, state.seats[hero].stack)
                    call_total = state.seats[hero].bet + call_cost
                    excess = sum(max(0, seat.bet - call_total) for seat in state.seats)
                    price = call_cost / max(1, state.pot - excess + call_cost)
                    previously_raised = any(event.kind == "raise" and event.seat == hero for event in state.history)
                    if not state.board and previously_raised and price < 0.12:
                        observed["contestable_pot_odds"] = price
                        cheap_folds.append(observed)
            _, reason = env.step(answer)
            if reason:
                errors.append({"bot": names[state.seat], "hand": state.hand_number, "error": reason})

        result, points = env.result()
        lost_hands, spend, spend_then_fold = [], [], []
        for hand in env.table.log:
            before, after = hand["stacks_before"][hero], hand["stacks_after"][hero]
            paid = hand["flop_chips_before"][hero] - hand["flop_chips_after"][hero]
            spend.append(paid)
            observed = own_hands.get(hand["hand"], [])
            if paid > 0 and any(item["action"]["kind"] == "fold" for item in observed):
                spend_then_fold.append(paid)
            if before > after:
                lost_hands.append({
                    "block": block, "rotation": rotation, "seat": hero,
                    "hand": hand["hand"], "chip_loss": before - after,
                    "icm_loss": hand["equity_before"][hero] - hand["equity_after"][hero],
                    "stack_before": before, "stack_after": after,
                    "flop_chips_paid": paid,
                    "public_decisions": observed,
                    # This evidence is collected only after the complete table ends.
                    "settled_cards": hand["cards"], "settled_events": hand["events"],
                })
        last = env.table.log[-1]
        eliminated = result.final_stacks[hero] == 0
        final_balance = last["flop_chips_after"][hero]
        biggest_chips = sorted(lost_hands, key=lambda hand: hand["chip_loss"], reverse=True)[:12]
        biggest_icm = sorted(lost_hands, key=lambda hand: hand["icm_loss"], reverse=True)[:12]
        retained_losses = {hand["hand"]: hand for hand in biggest_chips + biggest_icm}
        records.append({
            "block": block, "rotation": rotation, "seat": hero,
            "points": points[hero], "hands": result.hands_played,
            "eliminated": eliminated, "remaining_flop_chips": final_balance,
            "action_counts": dict(action_counts), "danger_counts": dict(danger_counts),
            "bids": bid_totals, "paid_per_hand": spend,
            "paid_then_folded": spend_then_fold,
            "dangerous_samples": dangerous[:2],
            "cheap_folds_after_raising": cheap_folds,
            "highest_losses": list(retained_losses.values()),
            "errors": errors,
        })
    return records


def distribution(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p90": values[min(len(values) - 1, int(len(values) * 0.9))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bots", nargs=8, type=Path)
    parser.add_argument("--focus", default="selected")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", default="boston-diagnostics-20261109")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folders = [folder.resolve() for folder in args.bots]
    if len({folder.name for folder in folders}) != 8 or args.focus not in {folder.name for folder in folders}:
        parser.error("Eight distinct folder names must include the focus agent")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {folder.name: hashlib.sha256((folder / "agent.py").read_bytes()).hexdigest() for folder in folders}
    records = []
    tasks = [(folders, args.focus, block, args.seed) for block in range(args.blocks)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for block, rows in enumerate(pool.map(diagnose_block, tasks), 1):
            records.extend(rows)
            print(f"Completed {block}/{args.blocks} blocks", flush=True)
    actions, dangers = Counter(), Counter()
    for record in records:
        actions.update(record["action_counts"])
        dangers.update(record["danger_counts"])
    bids = [bid for record in records for bid in record["bids"]]
    paid = [paid for record in records for paid in record["paid_per_hand"] if paid]
    folded = [paid for record in records for paid in record["paid_then_folded"]]
    eliminated = [record["remaining_flop_chips"] for record in records if record["eliminated"]]
    losses = [hand for record in records for hand in record["highest_losses"]]
    summary = {
        "focus": args.focus, "tables": len(records), "seed": args.seed, "sources": sources,
        "execution": "Official engine, direct calls, no deadlines imposed. Bots receive only public GameState.",
        "mean_points": statistics.mean(record["points"] for record in records),
        "action_counts": dict(actions), "high_commitment_actions": dict(dangers),
        "high_commitment_definition": "A call or raise spends at least one third of the remaining stack; this is a diagnostic flag, not an error.",
        "bids": distribution(bids), "positive_payments": distribution(paid),
        "paid_then_folded": distribution(folded),
        "eliminated_remaining_flop_chips": distribution(eliminated),
        "eliminated_with_over_500": sum(value > 500 for value in eliminated),
        "all_table_end_flop_chips": distribution([record["remaining_flop_chips"] for record in records]),
        "errors": [error for record in records for error in record["errors"]],
        "cheap_folds_after_raising": [{"block": record["block"], "rotation": record["rotation"], "decision": decision}
                                     for record in records for decision in record["cheap_folds_after_raising"]],
        "highest_chip_losses": sorted(losses, key=lambda hand: hand["chip_loss"], reverse=True)[:12],
        "highest_icm_losses": sorted(losses, key=lambda hand: hand["icm_loss"], reverse=True)[:12],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    compact = [{key: value for key, value in record.items() if key not in ("highest_losses", "dangerous_samples", "paid_per_hand", "bids")} for record in records]
    (args.out / "tables.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("highest_chip_losses", "highest_icm_losses", "sources", "cheap_folds_after_raising")}, indent=2))


if __name__ == "__main__":
    main()
