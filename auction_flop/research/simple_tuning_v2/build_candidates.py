"""Generate isolated numeric variants without editing the measured v1 source."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "candidates/simple_classic_v1"
DEFAULT = dict(BASE_BID=20.0, AUCTION_HORIZON=0, OPEN_LOOSER=False,
               CALL_SCALE=1.0, SHOVE_BB=12, BET_SCALE=1.0,
               BLUFF_FREQUENCY=0.5, EQUITY_SAMPLES=32)
ARMS = {
    "bid_zero": dict(BASE_BID=0.0), "bid_60": dict(BASE_BID=60.0),
    "bid_120": dict(BASE_BID=120.0),
    "duration_10": dict(AUCTION_HORIZON=10),
    "duration_25": dict(AUCTION_HORIZON=25),
    "loose_open": dict(OPEN_LOOSER=True),
    "call_wider": dict(CALL_SCALE=1.7),
    "shove_later": dict(SHOVE_BB=8),
    "bet_larger": dict(BET_SCALE=1.3),
    "no_air_bluff": dict(BLUFF_FREQUENCY=0.0),
    "samples_16": dict(EQUITY_SAMPLES=16),
}


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def write_config(path, changes):
    values = {**DEFAULT, **changes}
    assert set(values) == set(DEFAULT)
    path.write_text('"""Explicit tuning parameters; no environment-dependent settings."""\n\n' +
                    "\n".join(f"{key} = {value!r}" for key, value in values.items()) + "\n")


def make_template():
    target = ROOT / "candidates/simple_tunable_v2"
    shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns("__pycache__"))
    write_config(target / "config.py", {})
    path = target / "simple_auction.py"
    code = path.read_text().replace("from config import BASE_BID", "from config import BASE_BID, AUCTION_HORIZON")
    code = replace_once(code,
        "budget = min(balance, max(0, round(BASE_BID * improvement * 2 / players)))",
        "base = balance / AUCTION_HORIZON if AUCTION_HORIZON > 0 else BASE_BID\n"
        "    budget = min(balance, max(0, round(base * improvement * 2 / players)))")
    path.write_text(code)
    path = target / "simple_preflop.py"
    code = path.read_text().replace("RANKS =", "from config import OPEN_LOOSER, CALL_SCALE, SHOVE_BB\n\nRANKS =", 1)
    code = replace_once(code, "pair_min, hands = OPEN[pos]",
        'chart = {"early": "middle", "middle": "late", "late": "late"}[pos] if OPEN_LOOSER else pos\n'
        "    pair_min, hands = OPEN[chart]")
    code = replace_once(code, "stack <= 12 * bb", "stack <= SHOVE_BB * bb")
    code = replace_once(code, "cost <= 4 * bb and cost <= .15 * stack",
                        "cost <= 4 * CALL_SCALE * bb and cost <= .15 * CALL_SCALE * stack")
    path.write_text(code)
    path = target / "classic_postflop.py"
    code = path.read_text().replace('"""Unchanged postflop functions', '"""Parameterized postflop functions', 1)
    code = code.replace("Only unused preflop/profile code and its table import were removed.",
                        "Bet size, flop air-stab frequency and sampling budget are configurable.", 1)
    code = replace_once(code, "SAMPLES = 32", "from config import EQUITY_SAMPLES, BET_SCALE, BLUFF_FREQUENCY\nSAMPLES = EQUITY_SAMPLES")
    for fraction in (".75", ".55", ".5", ".8"):
        code = code.replace(f"round({fraction} * state.pot)", f"round({fraction} * BET_SCALE * state.pot)")
    code = replace_once(code, "stream.random() < .5", "stream.random() < BLUFF_FREQUENCY")
    path.write_text(code)
    return target


def main():
    template = make_template()
    arms = {"control": str(SOURCE.relative_to(ROOT))}
    for name, config in ARMS.items():
        target = HERE / "candidates" / name
        shutil.copytree(template, target)
        write_config(target / "config.py", config)
        arms[name] = str(target.relative_to(ROOT))
    (HERE / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    print(json.dumps({"template": str(template), "screen_arms": len(arms)}))


if __name__ == "__main__":
    main()
