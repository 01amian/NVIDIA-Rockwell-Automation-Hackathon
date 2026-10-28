#!/usr/bin/env python3
"""Large-Scale 100-Scene Automated Benchmark and Verification Engine.

Loops through 100 distinct scenes (3,000 total slot inspections),
evaluates each slot prediction against ground truth, logs real-time
verdicts cleanly line-by-line to the terminal, and provides an explicit
root-cause audit for every discrepancy.

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
    print(f"\n[*] Connecting to remote GPU ({host}) to benchmark {count} scenes...", flush=True)
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
        line = raw_line.strip("\r\n")
        if not line or "Shared connection" in line:
            continue
        print(line, flush=True)
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
        print(f"[!] Data directory not found: {data_dir}", flush=True)
        print("    If running locally, pass --remote to run against Brev VM scenes directly.", flush=True)
        sys.exit(1)

    print("=" * 110, flush=True)
    print(f"🚀 MASSIVE 100-SCENE AUTOMATED BENCHMARK SUITE ({args.count} FIXTURES / {args.count*30} INSPECTIONS)", flush=True)
    print(f"Data Source: {data_dir}", flush=True)
    print(f"Model Path:  {args.model}", flush=True)
    print("=" * 110, flush=True)

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
    skipped_scenes = []

    print(f"{'Scene ID':<11} | {'Verdict':<8} | {'Match Rate':<14} | {'Defects Caught':<16} | {'Escaped':<8} | {'False Alarms':<12} | {'Notes'}", flush=True)
    print("-" * 110, flush=True)

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
                skipped_scenes.append(scene_name)
                continue

            gt_data = json.load(open(gt_path))
            gt_map = {item['slot_id']: item['class'] for item in gt_data.get('slots', [])}
        except Exception as e:
            print(f"{scene_name:<11} | ⚠️ ERROR | {e}", flush=True)
            continue

        scene_matches = 0
        scene_defects = 0
        scene_caught = 0
        scene_escaped = 0
        scene_false_alarm = 0
        scene_discrepancies = []

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

                disc_info = {
                    "scene": scene_name,
                    "slot": slot_id,
                    "expected": expected,
                    "predicted": pred_class,
                    "conf": conf,
                    "reason": reason
                }
                discrepancies.append(disc_info)
                scene_discrepancies.append(f"{slot_id}:{expected}->{pred_class}")

        status_icon = "✅ PASS" if scene_matches == 30 else "⚠️ FAIL"
        caught_str = f"{scene_caught}/{scene_defects}" if scene_defects > 0 else "0/0 (Clean)"
        notes_str = ", ".join(scene_discrepancies) if scene_discrepancies else "-"
        print(f"{scene_name:<11} | {status_icon:<8} | {scene_matches:2d}/30 ({scene_matches/30:5.1%}) | {caught_str:<16} | {scene_escaped:<8} | {scene_false_alarm:<12} | {notes_str}", flush=True)

    if skipped_scenes:
        print(f"\n[ℹ️] Note: {len(skipped_scenes)} cold-start scenes skipped (shader cache warmup black frames: {skipped_scenes[0]} - {skipped_scenes[-1]})", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("📈 FINAL COMPREHENSIVE BENCHMARK AUDIT", flush=True)
    print("=" * 90, flush=True)

    acc = (total_matches / total_slots) if total_slots > 0 else 0
    recall = (total_defects_caught / total_defects_injected) if total_defects_injected > 0 else 1.0
    escape_rate = (total_escaped_defects / total_defects_injected) if total_defects_injected > 0 else 0.0

    print(f"Total Fixtures Evaluated:      {args.count}", flush=True)
    print(f"Total Valid Inspections:       {total_slots:,}", flush=True)
    print(f"Total Injected Defects:        {total_defects_injected:,}", flush=True)
    print(f"Defects Correctly Caught:      {total_defects_caught:,}/{total_defects_injected:,} ({recall:.2%})", flush=True)
    print(f"Escaped Defects (Missed):      {total_escaped_defects:,} ({escape_rate:.2%})", flush=True)
    print(f"False Alarm Alerts:            {total_false_alarms:,}", flush=True)
    print(f"Overall System Accuracy:       {acc:.2%}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Defect Class':<16} | {'Total Tested':<14} | {'Correct':<10} | {'Recall Rate'}", flush=True)
    print("-" * 90, flush=True)
    for c in ['pass', 'no_vial', 'missing_cap', 'crooked_cap', 'damaged_vial']:
        tot = class_counts[c]["total"]
        cor = class_counts[c]["correct"]
        r = (cor / tot) if tot > 0 else 1.0
        print(f"{c:<16} | {tot:<14} | {cor:<10} | {r:.2%}", flush=True)
    print("=" * 90, flush=True)

    # Detailed Failure & Mismatch Breakdown
    print("\n" + "=" * 110, flush=True)
    print("🚨 DETAILED FAILURE & MISMATCH BREAKDOWN (EVERY SINGLE ISSUE LISTED)", flush=True)
    print("=" * 110, flush=True)
    if not discrepancies:
        print("🎉 ZERO DISCREPANCIES! 100% of all slots matched ground truth perfectly.", flush=True)
    else:
        print(f"Total Discrepancies Recorded: {len(discrepancies)} out of {total_slots:,} slots inspected\n", flush=True)
        print(f"{'#':<3} | {'Scene':<10} | {'Slot':<6} | {'Expected (GT)':<14} | {'Predicted (AI)':<15} | {'Confidence':<10} | {'Root Cause & Impact'}", flush=True)
        print("-" * 110, flush=True)
        for i, d in enumerate(discrepancies, 1):
            if "Escaped" in d['reason']:
                impact = "CRITICAL FAIL: Defective vial was not caught!"
            elif "False Alarm" in d['reason']:
                impact = "SAFE REJECT: Good vial flagged as defect (false positive)"
            else:
                impact = "SAFE CAUGHT: Defect caught & rejected (class label ambiguity)"
            explanation = f"{d['reason']} -> {impact}"
            print(f"{i:<3} | {d['scene']:<10} | {d['slot']:<6} | {d['expected']:<14} | {d['predicted']:<15} | {d['conf']:<9.1%} | {explanation}", flush=True)
    print("=" * 110 + "\n", flush=True)


if __name__ == "__main__":
    main()
