"""Build a compact, reproducible diagnosis from complete checkpoint outputs."""
import argparse
import ast
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from .diagnose import Moments, bucket


def build(inputs, candidate):
    reports = {}
    for name in ('P4','S1','P5'):
        source = inputs/name/'summary.json'
        d=json.loads(source.read_text())
        groups={(r['checkpoint'],r['stratum']):r for r in d['checkpoint_groups']}
        combined=Moments();payout=Moments();inventory=defaultdict(Moments)
        with gzip.open(inputs/name/'table-checkpoints.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line); stages=row['accounting_by_players_before_hand'];o=row['outcome']
                delta=lambda n:stages.get(str(n),{}).get('points_change',0)
                combined.add({'players_8_to_5':sum(delta(n) for n in (8,7,6,5)),
                              'players_4':delta(4),'players_3':delta(3),'players_2':delta(2),
                              'players_3_and_2':delta(3)+delta(2)})
                pieces={'top3_component':20*(o['top3']-.375),
                        'top2_component':10*((o['final_rank']<=2)-.25),
                        'win_component':20*(o['win']-.125)}
                pieces['cap_or_tie_adjustment']=o['final_points']-12.5-sum(pieces.values())
                payout.add(pieces)
                cp=row['checkpoints'].get('first_2_or_fewer')
                if cp:
                    # Exploratory cross-strata, published including sparse cells.
                    share=bucket(cp['chip_share'],(.35,.65),('<35%','35-65%','65%+'))
                    depth='<=20' if cp['effective_bb']<=20 else '>20'
                    budget=bucket(cp['auction_chips'],(50,200),('<50','50-199','200+'))
                    inventory[(share,depth,budget)].add({'win':o['win'],'chip_share':cp['chip_share'],
                        'win_minus_chip_share':o['win']-cp['chip_share'], 'points_minus_icm':o['final_points']-cp['icm_points']})
        initial=groups[('hand_1','all')]
        hu=groups[('first_2_or_fewer','all')]
        ranks=[groups[(f'first_{n}_or_fewer',f'players={n};rank={r}')] for n in (4,3) for r in range(1,n+1)]
        assert [d['finish_rank_counts'].get(str(i),0) for i in range(1,9)] == d['official_result']['places']
        assert abs(sum(v['mean'] for v in payout.report()['metrics'].values())-(d['overall']['metrics']['final_points']['mean']-12.5)) < 1e-8
        reports[name]={'tournament_id':d['tournament_id'],'submission_id':d['submission_id'],
            'source_version':'old hybrid 7e7aa88' if name=='P4' else 'corrected loose v2 2873210',
            'tables':d['tables'],'public_hands':d['public_hands'],'history_sha256':d['history_sha256'],
            'full_summary':str(source.resolve()),'full_summary_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'table_checkpoints':str((inputs/name/'table-checkpoints.jsonl.gz').resolve()),
            'table_checkpoints_sha256':d['checkpoint_rows_sha256'],
            'official_points_and_all_eight_place_counts_match':True,'overall':d['overall'],
            'stage_delta_points_per_initial_table':combined.report(),
            'payout_decomposition_vs_symmetric_start':payout.report(),
            'fixed_hand_checkpoints':[groups[(f'hand_{n}','all')] for n in (1,21,41,61,81,101,121,161,201)],
            'checkpoint_coverage':d['checkpoint_coverage'],
            'entry_checkpoints':[groups[(f'first_{n}_or_fewer','all')] for n in (6,4,3,2)],
            'exact_3_and_4_player_rank_strata':ranks,
            'heads_up_share_and_depth_strata':[v for (cp,st),v in groups.items() if cp=='first_2_or_fewer' and (st.startswith('share=') or st.startswith('players=2;effective_bb='))],
            'heads_up_arithmetic':{
                'reach_rate':hu['n']/d['tables'],
                'point_gain_if_matching_chip_share_reference':-hu['metrics']['points_minus_icm']['mean']*hu['n']/d['tables'],
                'absolute_hu_win_rate_increase_needed_for_2_points_at_unchanged_reach':2/(20*hu['n']/d['tables']),
                'warning':'Reference matching is not a causal treatment or headroom bound; unchanged reach is an accounting assumption.'},
            'exploratory_hu_inventory_cross_strata':[{'share':s,'effective_bb':b,'auction_chips':a,**v.report()} for (s,b,a),v in sorted(inventory.items())],
            'inference_limits':d['definitions']}
    icm_calls=[]
    hashes={}
    for file in sorted(candidate.glob('*.py')):
        hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node,ast.Call) and ((isinstance(node.func,ast.Name) and node.func.id=='icm') or
                    (isinstance(node.func,ast.Attribute) and node.func.attr=='icm')):
                icm_calls.append({'file':file.name,'line':node.lineno})
    audit={'candidate':'hybrid_both','source_revision':'0c60de2e0aa5624844e3d0ae5d38e73ded25623c',
           'package_sha256':'3d5e6595a3c1f73f5a8ed051a25200609f19a3132487185739661b52a48520ef',
           'candidate_path':str(candidate.resolve()),'source_sha256':hashes,'explicit_icm_call_sites':icm_calls,
           'actual_decision_paths':[
               'agent.py:13 auction -> TournamentAgent.act; tournament_agent.py:14-21 returns auction_bids before betting risk_premium code.',
               'agent.py:16 preflop -> choose_preflop_loose; independent_preflop.py:93-97 only counts paid places and adds 0.035 to selected bubble call thresholds.',
               'agent.py:16 postflop -> classic_policy.postflop(price_veto=True, contribution_veto=True); no payout or ICM action-value calculation.',
               'classic_policy.equity samples uniformly from unknown opponent hole cards, not action-conditioned ranges; postflop uses pot odds and fixed hand-class/commitment rules.'],
           'verdict':'The benchmark rewards final tournament points and uses ICM at the hand cap. The deployed policy is selected for that objective but does not explicitly maximize expected ICM value per action. Existing payout-awareness is a limited preflop heuristic.'}
    assert not icm_calls
    hypotheses=[
      {'priority':1,'name':'Range-conditioned action-value policy across all stages',
       'evidence':'S1 and P5 repeat HU conversion deficits concentrated at 10-40 effective BB and below 65% entry share; early and middle checkpoints also show falling conversion. Current Hboth equity source samples uniformly, ignoring the information in bets.',
       'proposal':'Infer bounded opponent ranges from position, prior actions and bet sizes; compare fold/call/raise values with a shared value representation through later streets. Use a small action set and immutable precomputed tables, with per-table public updates.',
       'scope_for_two_points':'Extend beyond HU. Closing the observed HU chip-share-reference gap accounts for only 0.239/0.279 points per starting table in S1/P5.',
       'falsification':'A separately frozen field and fresh seeds must show at least +2 over exact Hboth, with reported interval, zero execution failures, and no collapse in stage-specific utility.'},
      {'priority':2,'name':'Tournament-point value at decision leaves, not blanket risk aversion',
       'evidence':'The deployed source does not compute ICM alternatives. S1/P5 exhibit losses of 0.344/0.394 points per initial table during 3-player plus HU play; medium-stack three-player residuals are negative. Overall top3 rates exceed symmetry while wins do not, so maximizing wins alone would be the wrong objective.',
       'proposal':'Evaluate fold, win, lose and split stack vectors in payout units; make risk depend on the actual payout increments, coverage and players left. For HU, ICM is linear in stack share, so extra bubble risk margins cannot solve the observed HU deficit.',
       'causal_limit':'ICM residuals are descriptive null-model discrepancies and may reflect opponent skill/ranges. They do not prove that adding ICM will recover these points.'},
      {'priority':3,'name':'Joint auction and continuation value with a normal-chip value for auction budget',
       'evidence':'The auction path optimizes a separate hand-quality/budget heuristic; subsequent betting uses another value model. Low auction inventory at HU entry correlates with poorer conversion in broad strata, but matched cross-strata are sparse and do not establish a causal budget effect.',
       'proposal':'Value each candidate flop and bid by the same downstream range-aware tournament-point model, including the future value of saved auction chips and the current effective stack/pot. Keep ordinary chips and auction currency separate.',
       'confidence':'Architectural hypothesis with weaker direct empirical support than priority 1; do not infer that simply saving or spending more auction chips improves outcomes.'}
    ]
    return {'schema':1,'rounds':reports,'current_policy_objective_audit':audit,'top_three_macro_hypotheses':hypotheses,
            'not_a_current_policy_live_evaluation':'P4 is old hybrid; S1 and P5 are corrected loose. None evaluates the newly deployed Hboth.',
            'no_score_promise':'These observations motivate experiments; no +2 improvement has been established.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True);result=build(a.inputs,a.candidate);a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.out)
