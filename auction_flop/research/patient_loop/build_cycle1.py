"""Create four immutable, isolated ablations from the patient v1 source."""
import json
from pathlib import Path
import shutil

from auction_flop.benchmark import run


def edit(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, path
    path.write_text(text.replace(old, new))


def main():
    parent = run.ROOT / 'candidates'
    names = ['patient_price_v2', 'patient_shortcall_v2',
             'patient_loose_v2', 'patient_huopen_v2']
    for name in names:
        shutil.copytree(parent / 'patient_value_v1', parent / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    edit(parent / names[0] / 'value_postflop.py',
         '    # Marginal hands and draws continue only for a modest price with a margin.\n',
         '''    # A strong single pair may be committed when the pot is already large.
    # Keep a margin because uniform equity ignores the bettor's stronger range.
    if (top_pair and opponents == 1 and price.threshold <= .35
            and me.stack + me.bet <= 2 * price.eligible_pot_before_call
            and eq >= max(.65, price.threshold + .20)):
        return call(state)
    # Marginal hands and draws continue only for a modest price with a margin.
''')
    short = parent / names[1] / 'simple_preflop.py'
    edit(short, 'def preflop(state):', 'def chart_action(state):')
    edit(short, 'from cards import passive, call, raise_to',
         'from cards import passive, call, raise_to\nfrom preflop_table import EQUITY\nfrom pot_accounting import eligible_call')
    with short.open('a') as f:
        f.write('''

def preflop(state):
    action = chart_action(state)
    if action.kind != "fold" or state.players_remaining > 3 or not state.can("call"):
        return action
    me = state.seats[state.seat]
    if me.stack + me.bet > 12 * max(1, state.blinds.big_blind):
        return action
    # Short-handed, do not surrender a very cheap call just because it is all-in.
    # This small reference table is uniform-opponent equity, not a solved chart.
    price = eligible_call(state)
    opponents = sum(s.status in ("active", "all_in") for s in state.seats) - 1
    raises = sum(e.kind == "raise" for e in state.history)
    margin = .18 if raises >= 2 else .12
    equity = EQUITY[hand_key(state.cards)] ** max(1, opponents)
    if price.threshold <= .40 and equity >= price.threshold + margin:
        return call(state)
    return action
''')
    shutil.copy2(parent / 'classic_txnujk32_v1/preflop_table.py', short.parent / 'preflop_table.py')
    config = parent / names[2] / 'config.py'
    edit(config, 'OPEN_LOOSER = False', 'OPEN_LOOSER = True')
    edit(config, 'CALL_SCALE = 1.0', 'CALL_SCALE = 1.7')
    edit(parent / names[3] / 'simple_preflop.py',
         '    if short_handed:\n        opening = opening or short_range',
         '''    if state.players_remaining == 2:
        short_range = (short_range or high == 13
                       or key.endswith('s') and high in (11, 12)
                       or high == 12 and low >= 8
                       or high == 11 and low >= 9
                       or high == 10 and low >= 8
                       or high == 9 and low >= 7)
    if short_handed:
        opening = opening or short_range''')
    arms = {'patient_value_v1': 'candidates/patient_value_v1'}
    arms.update({n: 'candidates/' + n for n in names})
    (Path(__file__).parent / 'cycle1-arms.json').write_text(json.dumps(arms, indent=2) + '\n')
    print(json.dumps(arms, indent=2))


if __name__ == '__main__':
    main()
