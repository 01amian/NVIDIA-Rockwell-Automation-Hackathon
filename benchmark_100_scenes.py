#!/usr/bin/env python3
"""Large-Scale 100-Scene Automated Benchmark and Verification Engine.

Loops through 100 distinct scenes (3,000 total slot inspections),
evaluates each slot prediction against ground truth, logs real-time
verdicts cleanly to the terminal, and computes a comprehensive industrial audit
including Defect Recall, Escaped Defect Rate, False Alarm Rate, and Class Metrics.

Usage Examples:
    # Run 100-scene benchmark on Brev cloud VM:
    python3 benchmark_100_scenes.py --remote

    # Run on local scenes directory:
    python3 benchmark_100_scenes.py --data-dir path/to/synthetic_100 --count 100
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.classifier import DeepLearningVialClassifier

SLOT_COORDS = {
    'G01': (377, 162), 'G02': (378, 198), 'G03': (380, 233), 'G04': (380, 268), 'G05': (382, 305),
    'G06': (348, 163), 'G07': (349, 199), 'G08': (350, 234), 'G09': (351, 270), 'G10': (352, 306),
    'G11': (318, 164), 'G12': (319, 200), 'G13': (320, 236), 'G14': (322, 271), 'G15': (322, 307),
    'G16': (287, 164), 'G17': (288, 201), 'G18': (290, 237), 'G19': (291, 273), 'G20': (294, 309),
    'G21': (256, 166), 'G22': (258, 202), 'G23': (259, 239), 'G24': (261, 274), 'G25': (262, 311),
    'G26': (225, 167), 'G27': (227, 204), 'G28': (229, 240), 'G29': (230, 276), 'G30': (232, 311)
}


def run_remote_benchmark(host: str, count: int, threshold: float):
    print(f"\n[*] Connecting to {host} to benchmark {count} scenes...", flush=True)
    cmd = (
        f"ssh -o StrictHostKeyChecking=no {host} "
        f"\"PYTHONPATH=/home/ubuntu/team_repo ~/.venv/bin/python3 "
        f"/home/ubuntu/team_repo/benchmark_100_scenes.py "
        f"--data-dir /home/ubuntu/csi_digital_twin_v1/synthetic_100 "
        f"--model /home/ubuntu/team_repo/models/vial_mobilenet.pth "
        f"--count {count} --threshold {threshold}\""
    )
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\r\n")
        print(line)
    proc.wait()
    sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Large-scale 100-scene automated benchmark")
    parser.add_argument("--data-dir", default=str(HERE / "synthetic_100"), help="Directory containing rendered scenes")
    parser.add_argument("--model", default=str(HERE / "models" / "vial_mobilenet.pth"), help="Path to MobileNetV3 weights")
    parser.add_argument("--count", type=int, default=100, help="Number of scenes to evaluate (1-100)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold for defect alerts")
    parser.add_argument("--remote", action="store_true", help="Execute benchmark remotely on Brev GPU host")
    parser.add_argument("--remote-host", default="vial-gpu", help="SSH host alias for Brev GPU")
    args = parser.parse_args()

    if args.remote:
        run_remote_benchmark(args.remote_host, args.count, args.threshold)
        return

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[!] Data directory not found: {data_dir}")
        print("    If running locally, pass --remote to run against Brev VM scenes directly.")
        sys.exit(1)

    print("\n" + "=" * 90)
    print(f"🚀 MASSIVE 100-SCENE AUTOMATED BENCHMARK SUITE ({args.count} FIXTURES / {args.count*30} INSPECTIONS)")
    print(f"Data Source: {data_dir}")
    print(f"Model Path:  {args.model}")
    print("=" * 90)

    classifier = DeepLearningVialClassifier(args.model, defect_threshold=args.threshold)
    half = 16

    # Metrics counters
    total_slots = 0
    total_matches = 0
    total_defects_injected = 0
    total_defects_caught = 0
    total_escaped_defects = 0
    total_false_alarms = 0

    class_counts = defaultdict(lambda: {"total": 0, "correct": 0})
    discrepancies = []

    print(f"\n{'Scene ID':<11} | {'Verdict':<8} | {'Match Rate':<12} | {'Defects Caught':<16} | {'Escaped':<8} | {'False Alarms'}")
    print("-" * 90)

    for idx in range(1, args.count + 1):
        scene_name = f"scene_{idx:03d}"
        img_path = data_dir / f"{scene_name}.png"
        gt_path = data_dir / f"{scene_name}_gt.json"

        if not img_path.exists() or not gt_path.exists():
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            import numpy as np
            arr = np.asarray(img)
            if arr.mean() < 5.0:
                print(f"{scene_name:<11} | ⚠️ SKIP  | 0/30 (Black Frame / Unrendered)")
                continue

            gt_data = json.load(open(gt_path))
            gt_map = {item['slot_id']: item['class'] for item in gt_data.get('slots', [])}
        except Exception as e:
            print(f"{scene_name:<11} | ⚠️ ERROR | {e}")
            continue

        scene_matches = 0
        scene_defects = 0
        scene_caught = 0
        scene_escaped = 0
        scene_false_alarm = 0

        for slot_id, (cx, cy) in sorted(SLOT_COORDS.items()):
            crop = img.crop((cx - half, cy - half, cx + half, cy + half))
            pred_class, conf, feat = classifier.classify_crop(crop, {'slot_id': slot_id})
            expected = gt_map.get(slot_id, 'pass')

            total_slots += 1
            class_counts[expected]["total"] += 1

            if expected != 'pass':
                total_defects_injected += 1
                scene_defects += 1

            if pred_class == expected:
                total_matches += 1
                scene_matches += 1
                class_counts[expected]["correct"] += 1
                if expected != 'pass':
                    total_defects_caught += 1
                    scene_caught += 1
            else:
                if expected != 'pass' and pred_class == 'pass':
                    total_escaped_defects += 1
                    scene_escaped += 1
                    reason = "Escaped Defect (Defective vial classified as pass)"
                elif expected == 'pass' and pred_class != 'pass':
                    total_false_alarms += 1
                    scene_false_alarm += 1
                    reason = f"False Alarm (Pass vial flagged as {pred_class})"
                else:
                    reason = f"Defect Type Discrepancy ({expected} caught as {pred_class})"

                discrepancies.append({
                    "scene": scene_name,
                    "slot": slot_id,
                    "expected": expected,
                    "predicted": pred_class,
                    "conf": conf,
                    "reason": reason
                })

        status_icon = "✅ PASS" if scene_matches == 30 else "⚠️ FAIL"
        caught_str = f"{scene_caught}/{scene_defects}" if scene_defects > 0 else "0/0 (Clean)"
        print(f"{scene_name:<11} | {status_icon:<8} | {scene_matches:2d}/30 ({scene_matches/30:5.1%}) | {caught_str:<16} | {scene_escaped:<8} | {scene_false_alarm}")

    print("=" * 90)
    print("📈 FINAL COMPREHENSIVE BENCHMARK AUDIT")
    print("=" * 90)

    acc = (total_matches / total_slots) if total_slots > 0 else 0
    recall = (total_defects_caught / total_defects_injected) if total_defects_injected > 0 else 1.0
    escape_rate = (total_escaped_defects / total_defects_injected) if total_defects_injected > 0 else 0.0

    print(f"Total Fixtures Evaluated:      {args.count}")
    print(f"Total Slot Positions Tested:   {total_slots:,}")
    print(f"Total Injected Defects:        {total_defects_injected:,}")
    print(f"Defects Correctly Caught:      {total_defects_caught:,}/{total_defects_injected:,} ({recall:.2%})")
    print(f"Escaped Defects (Missed):      {total_escaped_defects:,} ({escape_rate:.2%})")
    print(f"False Alarm Alerts:            {total_false_alarms:,}")
    print(f"Overall System Accuracy:       {acc:.2%}")
    print("-" * 90)
    print(f"{'Defect Class':<16} | {'Total Tested':<14} | {'Correct':<10} | {'Recall Rate'}")
    print("-" * 90)
    for c in ['pass', 'no_vial', 'missing_cap', 'crooked_cap', 'damaged_vial']:
        tot = class_counts[c]["total"]
        cor = class_counts[c]["correct"]
        r = (cor / tot) if tot > 0 else 1.0
        print(f"{c:<16} | {tot:<14} | {cor:<10} | {r:.2%}")
    print("=" * 90)

    # Detailed Failure & Mismatch Breakdown
    print("\n" + "=" * 90)
    print("🚨 DETAILED FAILURE & MISMATCH BREAKDOWN")
    print("=" * 90)
    if not discrepancies:
        print("🎉 ZERO DISCREPANCIES! 100% of all slots matched ground truth perfectly.")
    else:
        print(f"Total Discrepancies Recorded: {len(discrepancies)} out of {total_slots:,} slots inspected\n")
        print(f"{'Scene':<10} | {'Slot':<6} | {'Expected (GT)':<14} | {'Predicted (AI)':<15} | {'Confidence':<10} | {'Diagnostic Reason'}")
        print("-" * 90)
        for d in discrepancies:
            print(f"{d['scene']:<10} | {d['slot']:<6} | {d['expected']:<14} | {d['predicted']:<15} | {d['conf']:<9.1%} | {d['reason']}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
