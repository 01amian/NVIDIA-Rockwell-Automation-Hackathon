"""Unit and integration tests for vial inspection system."""

import unittest
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(HERE))

from src.geometry import TrayGeometry, compute_center
from src.rules import summarize, CLASSES, DEFECTS
from src.pipeline import VialInspectionPipeline
from src.evaluate import evaluate_predictions


class TestVialInspection(unittest.TestCase):

    def setUp(self):
        self.calib_path = HERE / "calibration.json"
        self.pipeline = VialInspectionPipeline(str(self.calib_path))

    def test_geometry_generates_60_slots(self):
        geom = TrayGeometry(str(self.calib_path))
        slots = geom.get_slot_definitions()
        self.assertEqual(len(slots), 60)
        
        good = [s for s in slots if s['area'] == 'good']
        scrap = [s for s in slots if s['area'] == 'scrap']
        self.assertEqual(len(good), 30)
        self.assertEqual(len(scrap), 30)

        # Verify ID formatting
        self.assertEqual(good[0]['id'], 'G01')
        self.assertEqual(good[29]['id'], 'G30')
        self.assertEqual(scrap[0]['id'], 'S01')
        self.assertEqual(scrap[29]['id'], 'S30')

    def test_rules_complete_order(self):
        slots = [{'position': i, 'presence': True, 'class': 'pass'} for i in range(1, 31)]
        res = summarize('good', slots)
        self.assertTrue(res['findings']['complete_order'])
        self.assertFalse(res['findings']['missing_vial'])
        self.assertFalse(res['findings']['bad_misclassified_as_good'])

    def test_rules_detects_bad_in_good_tray(self):
        slots = [{'position': i, 'presence': True, 'class': 'pass'} for i in range(1, 31)]
        slots[5]['class'] = 'crooked_cap'
        res = summarize('good', slots)
        self.assertFalse(res['findings']['complete_order'])
        self.assertTrue(res['findings']['bad_misclassified_as_good'])
        self.assertIn(6, res['visible_defect_positions'])

    def test_evaluation_metric_penalty(self):
        gt = [{'id': f"G{i:02d}", 'class': 'pass'} for i in range(1, 10)]
        gt.append({'id': 'G10', 'class': 'missing_cap'}) # 1 true defect

        pred_perfect = list(gt)
        eval_perfect = evaluate_predictions(gt, pred_perfect)
        self.assertEqual(eval_perfect['penalty_score'], 0)

        # Escaped defect (false negative) should incur high penalty
        pred_missed = [{'id': f"G{i:02d}", 'class': 'pass'} for i in range(1, 11)]
        eval_missed = evaluate_predictions(gt, pred_missed, missed_defect_weight=10.0)
        self.assertEqual(eval_missed['escaped_defects'], 1)
        self.assertEqual(eval_missed['penalty_score'], 10.0)


if __name__ == '__main__':
    unittest.main()
