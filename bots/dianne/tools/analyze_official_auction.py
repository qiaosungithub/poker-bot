"""Aggregate Boston auction history without exporting private cards or raw bids."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def distribution(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p90": values[min(len(values)-1, int(len(values)*0.9))],
            "max": values[-1], "zero_count": values.count(0)}


def bucket(value, boundaries):
    for boundary in boundaries:
        if value <= boundary:
            return f"at_most_{boundary}"
    return f"over_{boundaries[-1]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--reader", type=Path, default=Path(__file__).resolve().parents[4] / "starter-kit/hand-history/read.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agent", type=Path, help="Optional frozen agent supplying hand_quality for descriptive zero-bid diagnostics")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_history_reader", args.reader)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    agent = None
    if args.agent:
        agent_spec = importlib.util.spec_from_file_location("frozen_auction_policy", args.agent)
        agent = importlib.util.module_from_spec(agent_spec)
        agent_spec.loader.exec_module(agent)
    counts, selection, bid_efficiency = Counter(), Counter(), Counter()
    own_spend, other_spend, own_balances, eliminated_balances = [], [], [], []
    paid_then_folded, elimination_hand, auctions_per_table, payments_per_table = [], [], [], []
    by_stack, by_alive, by_bidder, elim_by_hand = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    zero_bid_quality_drops = []
    target_selection_by_bid = defaultdict(Counter)
    cap_presence = Counter()
    for header, table, private in reader.read_history(args.history):
        hero = private["seat"]
        if table["fillers"][hero]:
            counts["filler_tables_skipped"] += 1
            continue
        counts["scored_tables"] += 1
        detail = defaultdict(list)
        for event in private.get("private_events", []):
            if event["kind"] == "bid":
                detail[event["hand"]].append(event)
        own_auctions = own_payments = 0
        own_cards = {hand["hand"]: tuple(cards) for hand, cards in zip(table["hands"], private["cards"])}
        for hand in table["hands"]:
            if "flop_chips_before" not in hand or "flop_chips_after" not in hand:
                counts["hands_missing_balances"] += 1
                continue
            before, after = hand["flop_chips_before"], hand["flop_chips_after"]
            alive = sum(stack > 0 for stack in hand["stacks_before"])
            events = hand["events"]
            bidders = {event["seat"] for event in events if event["kind"] == "bid"}
            for seat in bidders:
                if seat != hero:
                    other_spend.append(before[seat] - after[seat])
            if hand["stacks_before"][hero] > 0 and hand["stacks_after"][hero] == 0:
                eliminated_balances.append(after[hero])
                elimination_hand.append(hand["hand"])
                elim_by_hand[bucket(hand["hand"], (40, 80, 120, 160, 200))].append(after[hero])
                counts["eliminations_with_over_500"] += after[hero] > 500
                counts["eliminations_with_1000"] += after[hero] == 1000
            if hero not in bidders:
                continue
            own_auctions += 1
            counts["own_auctions"] += 1
            paid = before[hero] - after[hero]
            own_payments += paid > 0
            own_spend.append(paid)
            stack_bb = hand["stacks_before"][hero] / max(1, hand["blinds"][1])
            stack_bucket = bucket(stack_bb, (5, 10, 20, 40))
            alive_bucket = str(alive)
            by_stack[stack_bucket].append(paid)
            by_alive[alive_bucket].append(paid)
            by_bidder[str(len(bidders))].append(paid)
            if paid > 0 and any(event["kind"] == "fold" and event.get("seat") == hero for event in events):
                paid_then_folded.append(paid)
            auction_index = next(index for index, event in enumerate(events) if event["kind"] == "auction")
            auction_cards = events[auction_index]["cards"]
            flop = next(event["cards"] for event in events[auction_index+1:] if event["kind"] == "board")
            selected = set(flop)
            for index, card in enumerate(auction_cards):
                selection[f"index_{index}_selected"] += card in selected
            selection["default_first_three"] += set(auction_cards[:3]) == selected
            known_bids = {event["cards"][0]: event.get("amount", 0) for event in detail[hand["hand"]]}
            if "private_events" not in private:
                counts["auctions_without_private_events"] += 1
                continue
            total_bid = sum(known_bids.values())
            predicted_payment = sum(amount for card, amount in known_bids.items() if card in selected)
            counts["own_payment_reconstruction_mismatches"] += predicted_payment != paid
            bid_efficiency["total_bid"] += total_bid
            bid_efficiency["paid_bid"] += predicted_payment
            bid_efficiency["refunded_dead_card_bid"] += total_bid - predicted_payment
            positive = {card for card, amount in known_bids.items() if amount > 0}
            bid_efficiency["positive_card_bids"] += len(positive)
            bid_efficiency["positive_card_bids_selected"] += len(positive & selected)
            if positive:
                bid_efficiency["positive_bid_auctions"] += 1
                bid_efficiency["all_positive_cards_selected"] += positive <= selected
                tier = target_selection_by_bid[bucket(total_bid, (15, 45, 90, 180))]
                tier["auctions"] += 1
                tier["all_positive_cards_selected"] += positive <= selected
            else:
                selection["zero_bid_auctions"] += 1
                selection["zero_bid_default_first_three"] += set(auction_cards[:3]) == selected
                if agent:
                    hole = own_cards[hand["hand"]]
                    baseline_quality = agent.hand_quality(hole, tuple(auction_cards[:3]))[0]
                    actual_quality = agent.hand_quality(hole, tuple(flop))[0]
                    drop = baseline_quality - actual_quality
                    zero_bid_quality_drops.append(drop)
                    counts["zero_bid_quality_drop_over_010"] += drop > 0.10
                    counts["zero_bid_quality_drop_over_025"] += drop > 0.25
                    counts["zero_bid_good_default_lost"] += baseline_quality >= 0.70 and drop > 0.10
            if before[hero] > 0:
                fraction = paid / before[hero]
                cap_presence["paid_at_least_20pct"] += fraction >= 0.20
                cap_presence["paid_at_least_50pct"] += fraction >= 0.50
        auctions_per_table.append(own_auctions)
        payments_per_table.append(own_payments)
        if table["hands"] and "flop_chips_after" in table["hands"][-1]:
            own_balances.append(table["hands"][-1]["flop_chips_after"][hero])
    output = {
        "source_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
        "source_filename": args.history.name,
        "scope": "Downloaded scored-table subset; public balance changes and own private bid amounts only. No opponent per-card bids are observed or inferred.",
        "counts": dict(counts), "own_auction_payments": distribution(own_spend),
        "other_player_auction_payments": distribution(other_spend),
        "own_auctions_per_table": distribution(auctions_per_table),
        "own_positive_payments_per_table": distribution(payments_per_table),
        "own_paid_then_folded": distribution(paid_then_folded),
        "own_table_end_balances": distribution(own_balances),
        "own_elimination_balances": distribution(eliminated_balances),
        "own_elimination_hand": distribution(elimination_hand),
        "payments_by_start_of_hand_stack_bb": {key: distribution(values) for key, values in by_stack.items()},
        "payments_by_players_alive_at_hand_start": {key: distribution(values) for key, values in by_alive.items()},
        "payments_by_auction_bidders": {key: distribution(values) for key, values in by_bidder.items()},
        "elimination_balances_by_hand": {key: distribution(values) for key, values in elim_by_hand.items()},
        "flop_selection": dict(selection), "known_own_bid_efficiency": dict(bid_efficiency),
        "large_payment_counts": dict(cap_presence),
        "target_selection_by_total_own_bid": {key: dict(value) for key, value in target_selection_by_bid.items()},
        "zero_bid_default_minus_actual_heuristic_quality": distribution(zero_bid_quality_drops),
        "heuristic_agent_sha256": hashlib.sha256(args.agent.read_bytes()).hexdigest() if args.agent else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
