#!/usr/bin/env python3
"""CLI tool to evaluate inspection predictions against ground truth labels.

Usage:
    python evaluate_run.py --ground-truth my_scene_gt.json --predictions inspection_results/inspection_summary.json
"""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.evaluate import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate vial inspection predictions against ground truth")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON (e.g., from generate_test_scene.py)")
    parser.add_argument("--predictions", required=True, help="Path to predictions inspection_summary.json")
    parser.add_argument("--missed-penalty", type=float, default=10.0, help="Penalty multiplier for missed defects")
    parser.add_argument("--false-alarm-penalty", type=float, default=1.0, help="Penalty multiplier for false alarms")
    args = parser.parse_args()

    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(args.predictions, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    gt_slots = gt_data.get('slots', [])
    pred_slots = pred_data.get('slots', [])

    report = evaluate_predictions(
        gt_slots,
        pred_slots,
        missed_defect_weight=args.missed_penalty,
        false_alarm_weight=args.false_alarm_penalty
    )

    print("\n" + "=" * 60)
    print("           HACKATHON INSPECTION BENCHMARK           ")
    print("=" * 60)
    print(f"Total True Defects:      {report['total_defects']}")
    print(f"Escaped Defects (FN):    {report['escaped_defects']} (Rate: {report['escaped_defect_rate']*100:.1f}%)")
    print(f"Total True Passes:       {report['total_passes']}")
    print(f"False Alarms (FP):       {report['false_alarms']} (Rate: {report['false_alarm_rate']*100:.1f}%)")
    print(f"Penalty Score:           {report['penalty_score']:.1f}  (Lower is better; missed defect weight = {args.missed_penalty}x)")
    
    print("\n--- PER-CLASS PERFORMANCE ---")
    print(f"{'Class':<15} | {'Count':<6} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 55)
    for c, stats in report['per_class'].items():
        print(f"{c:<15} | {stats['count']:<6} | {stats['precision']:<9.2f} | {stats['recall']:<8.2f} | {stats['f1']:<8.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
