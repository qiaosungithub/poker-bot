"""Negative controls for score, schedule, source-lock and paired statistics."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import run


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = run.verify_lock()
        cls.config = {'blocks': 2, 'seed': 'fixed-benchmark-test', 'hand_cap': 250,
                      'entries': [run.entry(str(i), 'checkfold', run.ROOT/'strategies')
                                  for i in range(8)]}
        run.arena.initialize(cls.config)
        cls.rows = []
        for block in range(2):
            cls.rows += run.arena.block((block, f'fixed-benchmark-test:{block}',
                                         [str(i) for i in range(8)]))
        for row in cls.rows:
            row['panel'] = 0

    def test_valid_scores_and_schedule(self):
        self.assertTrue(run.audit_rows(self.rows, [self.config], 2))

    def test_identical_policies_have_exact_equal_rotation_scores(self):
        for block in range(2):
            for name in map(str, range(8)):
                points = [e['points'] for row in self.rows if row['block'] == block
                          for e in row['entries'] if e['name'] == name]
                self.assertAlmostEqual(sum(points) / len(points), 12.5)

    def test_corrupt_score_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['entries'][0]['points'] += 1
        with self.assertRaisesRegex(ValueError, 'score audit'):
            run.audit_rows(rows, [self.config], 2)

    def test_nonfinite_score_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['entries'][0]['points'] = math.nan
        with self.assertRaisesRegex(ValueError, 'score audit'):
            run.audit_rows(rows, [self.config], 2)

    def test_missing_rotation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            run.audit_rows(self.rows[:-1], [self.config], 2)

    def test_duplicate_rotation_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            run.audit_rows(rows, [self.config], 2)

    def test_wrong_seed_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['seed'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Seed'):
            run.audit_rows(rows, [self.config], 2)

    def test_wrong_seating_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['entries'][0]['name'], rows[0]['entries'][1]['name'] = (
            rows[0]['entries'][1]['name'], rows[0]['entries'][0]['name'])
        with self.assertRaisesRegex(ValueError, 'rotation'):
            run.audit_rows(rows, [self.config], 2)

    def test_currency_corruption_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['entries'][0]['remaining_flopchips'] -= 1
        with self.assertRaisesRegex(ValueError, 'currency'):
            run.audit_rows(rows, [self.config], 2)

    def test_frozen_source_change_rejected(self):
        real = run.digest
        def corrupted(path):
            return 'broken' if Path(path).name == 'arena.py' else real(path)
        with patch.object(run, 'digest', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                run.verify_lock()

    def test_block_rerun_deterministic(self):
        rerun = run.arena.block((0, 'fixed-benchmark-test:0', [str(i) for i in range(8)]))
        for a, b in zip(self.rows[:8], rerun):
            self.assertEqual({k:v for k,v in a.items() if k not in ('seconds', 'panel')},
                             {k:v for k,v in b.items() if k != 'seconds'})

    def test_equal_panel_weights(self):
        result = run.estimate({(0, 0): 10, (0, 1): 14, (1, 0): 20, (1, 1): 24})
        self.assertEqual(result['mean_points'], 17)
        self.assertAlmostEqual(result['ci95'][1], 17 + 1.96 * math.sqrt(2))

    def test_zero_paired_difference(self):
        result = run.estimate({(p, b): 0.0 for p in range(4) for b in range(4)})
        self.assertEqual(result['ci95'], [0, 0])


if __name__ == '__main__':
    unittest.main()
