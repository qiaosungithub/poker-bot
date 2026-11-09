"""Generate controlled auction swaps with exactly the selected betting policy."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOT_ROOT = ROOT / "bots/dianne"
DESTINATION = ROOT / ".runtime/boston/phase-ablation-candidates"


def source_node(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.get_source_segment(source, node)
    raise ValueError(name)


def main():
    base = (BOT_ROOT / "selected/agent.py").read_text()
    old_auction = source_node(base, "auction_bid")
    shadow = (BOT_ROOT / "auction_shadow_price/agent.py").read_text()
    push = (BOT_ROOT / "push_fold/agent.py").read_text()
    zero = '''def auction_bid(state):
    return Action("bid", cards=(0, 0, 0, 0, 0))'''
    double = '''def auction_bid(state):
    action = base_auction_bid(state)
    budget = state.can("bid").max_amount
    total = sum(action.cards)
    if total == 0:
        return action
    multiplier = min(2.0, budget / total)
    return Action("bid", cards=tuple(int(bid * multiplier) for bid in action.cards))'''
    push_auction = source_node(push, "auction_action").replace("def auction_action(", "def auction_bid(")
    variants = {
        "phase_base": base,
        "phase_zero": base.replace(old_auction, zero),
        "phase_shadow": base.replace(old_auction, source_node(shadow, "auction_bid")),
        "phase_push": base.replace(old_auction, "\n\n".join([
            source_node(push, "STRAIGHTS"), source_node(push, "board_strength"), push_auction])),
        "phase_double": base.replace(old_auction, old_auction.replace("def auction_bid(", "def base_auction_bid(") + "\n\n" + double),
    }
    base_betting = source_node(base, "contestable_pot")
    base_class = next(node for node in ast.parse(base).body if isinstance(node, ast.ClassDef))
    for name, source in variants.items():
        folder = DESTINATION / name
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "agent.py").write_text(source)
        candidate_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
        assert ast.dump(candidate_class) == ast.dump(base_class)
        assert source_node(source, "contestable_pot") == base_betting
        print(folder)


if __name__ == "__main__":
    main()
