#!/usr/bin/env python3
"""Interactive Isaac Sim Custom Vial Defect Inspection Tool.

Allows you to specify any vial defect scenario, sends it to NVIDIA Isaac Sim
on the Brev cloud GPU to render the photorealistic digital twin scene,
transfers the image back locally, runs it through our AI inspection pipeline,
and generates an annotated overlay and diagnostic verdict report.

Usage Examples:
    # Custom scenario with multiple defects:
    python3 run_custom_inspection.py --name scenario_01 --defects G05:no_vial G12:crooked_cap G14:missing_cap

    # All-pass baseline scenario:
    python3 run_custom_inspection.py --name clean_tray

    # Damaged vial scenario:
    python3 run_custom_inspection.py --name damaged_test --defects G03:damaged_vial G20:missing_cap
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.classifier import DeepLearningVialClassifier

# Calibrated 2D projected slot centers in CapSealCam (640x480)
SLOT_COORDS = {
    'G01': (377, 162), 'G02': (378, 198), 'G03': (380, 233), 'G04': (380, 268), 'G05': (382, 305),
    'G06': (348, 163), 'G07': (349, 199), 'G08': (350, 234), 'G09': (351, 270), 'G10': (352, 306),
    'G11': (318, 164), 'G12': (319, 200), 'G13': (320, 236), 'G14': (322, 271), 'G15': (322, 307),
    'G16': (287, 164), 'G17': (288, 201), 'G18': (290, 237), 'G19': (291, 273), 'G20': (294, 309),
    'G21': (256, 166), 'G22': (258, 202), 'G23': (259, 239), 'G24': (261, 274), 'G25': (262, 311),
    'G26': (225, 167), 'G27': (227, 204), 'G28': (229, 240), 'G29': (230, 276), 'G30': (232, 311)
}

COLOR_MAP = {
    'pass': (32, 168, 117),         # Emerald Green
    'no_vial': (59, 130, 246),       # Electric Blue
    'missing_cap': (245, 158, 11),   # Amber/Orange
    'crooked_cap': (168, 85, 247),   # Purple
    'damaged_vial': (239, 68, 68)    # Crimson Red
}


def normalize_slot_id(slot_raw: str) -> str:
    s = slot_raw.strip().upper()
    if s.startswith('G') and len(s) == 2:
        return f"G0{s[1]}"
    if s.startswith('S') and len(s) == 2:
        return f"S0{s[1]}"
    return s


def run_command(cmd: str, desc: str = "") -> str:
    print(f"[*] {desc}...", flush=True)
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[!] Error executing: {cmd}\n{res.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
    return res.stdout


def main():
    parser = argparse.ArgumentParser(description="Render custom defect scenario in Isaac Sim and inspect locally")
    parser.add_argument("--defects", nargs="*", default=[],
                        help="Defects to inject as SLOT:CLASS (e.g. G05:no_vial G12:crooked_cap G14:missing_cap)")
    parser.add_argument("--name", default="custom_scene", help="Scenario output name")
    parser.add_argument("--remote-host", default="vial-gpu", help="SSH host alias for Brev GPU VM")
    parser.add_argument("--out-dir", default=None, help="Directory to store rendered images and analysis")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold for defect alerts")
    args = parser.parse_args()

    # Parse and validate injected defects
    parsed_defects = {}
    for item in args.defects:
        if ":" not in item:
            print(f"[!] Invalid defect format '{item}'. Use format SLOT:CLASS (e.g. G05:no_vial)")
            sys.exit(1)
        slot, cls_name = item.split(":", 1)
        slot_norm = normalize_slot_id(slot)
        cls_norm = cls_name.strip().lower()
        if cls_norm not in ['pass', 'no_vial', 'missing_cap', 'crooked_cap', 'damaged_vial']:
            print(f"[!] Unknown defect class '{cls_norm}'. Valid classes: pass, no_vial, missing_cap, crooked_cap, damaged_vial")
            sys.exit(1)
        parsed_defects[slot_norm] = cls_norm

    out_dir = Path(args.out_dir or (HERE / "output_scenes" / args.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"🚀 INITIATING CUSTOM DIGITAL TWIN INSPECTION CYCLE: {args.name}")
    print(f"Injected Defects: {parsed_defects if parsed_defects else 'None (All Pass)'}")
    print("=" * 70)

    # 1. RENDER IN DIGITAL TWIN VIA BREV GPU
    defect_cli_args = " ".join([f"{k}:{v}" for k, v in parsed_defects.items()])
    sim_docker_cmd = (
        f"ssh -o StrictHostKeyChecking=no {args.remote_host} "
        f"\"docker run --gpus all -e 'ACCEPT_EULA=Y' --rm "
        f"-v /home/ubuntu/csi_digital_twin_v1:/workspace "
        f"-v /home/ubuntu/shader_cache:/root/.cache "
        f"-v /home/ubuntu/kit_cache:/root/isaacsim/kit/cache "
        f"-w /workspace j3soon/runai-isaac-sim-ex:6.0.0-ros2-jazzy "
        f"/root/isaacsim/python.sh tools/generate_synthetic_defects.py --name {args.name} --defects {defect_cli_args}\""
    )

    print(f"\n[1/4] Rendering 3D ray-traced scene in Isaac Sim on {args.remote_host}...")
    run_command(sim_docker_cmd, "Dispatching render job to NVIDIA Isaac Sim")

    # 2. TRANSFER IMAGES & GROUND TRUTH LOCALLY
    local_img = out_dir / f"{args.name}.png"
    local_gt = out_dir / f"{args.name}_gt.json"

    print(f"\n[2/4] Downloading rendered scene and ground truth to {out_dir}...")
    scp_img_cmd = f"scp \"{args.remote_host}:/home/ubuntu/csi_digital_twin_v1/synthetic_dataset/{args.name}.png\" \"{local_img}\""
    scp_gt_cmd = f"scp \"{args.remote_host}:/home/ubuntu/csi_digital_twin_v1/synthetic_dataset/{args.name}_gt.json\" \"{local_gt}\""
    run_command(scp_img_cmd, "Transferring rendered scene image")
    run_command(scp_gt_cmd, "Transferring scene ground truth JSON")

    # 3. RUN LOCAL AI INSPECTION PIPELINE
    print("\n[3/4] Running automated AI inspection pipeline on rendered image...")
    model_path = HERE / "models" / "vial_mobilenet.pth"
    classifier = DeepLearningVialClassifier(str(model_path), defect_threshold=args.threshold)

    img = Image.open(local_img).convert('RGB')
    gt_data = json.load(open(local_gt))
    gt_map = {item['slot_id']: item['class'] for item in gt_data.get('slots', [])}

    inspection_results = []
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    half = 16

    escaped_defects = 0
    false_alarms = 0
    correct_count = 0
    injected_count = sum(1 for v in gt_map.values() if v != 'pass' and v.startswith('G') or v in parsed_defects.values())

    for slot_id in sorted(SLOT_COORDS.keys()):
        cx, cy = SLOT_COORDS[slot_id]
        crop = img.crop((cx - half, cy - half, cx + half, cy + half))
        pred_class, conf, feat = classifier.classify_crop(crop, {'slot_id': slot_id})

        expected = gt_map.get(slot_id, parsed_defects.get(slot_id, 'pass'))
        is_match = (pred_class == expected)

        if is_match:
            correct_count += 1
            status_badge = "✅ MATCH"
        elif expected != 'pass' and pred_class == 'pass':
            escaped_defects += 1
            status_badge = "🚨 MISSED DEFECT (Escaped)"
        elif expected == 'pass' and pred_class != 'pass':
            false_alarms += 1
            status_badge = "⚠️  FALSE ALARM"
        else:
            status_badge = "⚠️  CLASS MISMATCH"

        inspection_results.append({
            'slot_id': slot_id,
            'expected': expected,
            'predicted': pred_class,
            'confidence': conf,
            'match': is_match,
            'status': status_badge,
            'probabilities': feat.get('probabilities', {})
        })

        # Draw inspection annotations on overlay
        box_color = COLOR_MAP.get(pred_class, (170, 180, 196))
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Highlight discrepancies with high-contrast alert border
        if not is_match:
            draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(255, 0, 0), width=2)

        # Label slot ID and prediction
        label_text = f"{slot_id}:{pred_class[:4]}"
        draw.text((x1, y1 - 10), label_text, fill=box_color)

    annotated_path = out_dir / f"annotated_{args.name}.png"
    overlay.save(annotated_path)

    # 4. REPORT & SUMMARY
    print("\n" + "=" * 70)
    print("📊 INSPECTION VERDICT & DISCREPANCY AUDIT REPORT")
    print("=" * 70)
    print(f"{'Slot':<6} | {'Injected Defect (GT)':<20} | {'Detected (AI)':<15} | {'Confidence':<10} | {'Status'}")
    print("-" * 70)

    for res in inspection_results:
        # Show all injected defects or non-pass items or mismatches prominently
        if res['expected'] != 'pass' or res['predicted'] != 'pass' or not res['match']:
            print(f"{res['slot_id']:<6} | {res['expected']:<20} | {res['predicted']:<15} | {res['confidence']:<9.1%} | {res['status']}")

    print("-" * 70)
    total_slots = len(SLOT_COORDS)
    accuracy = correct_count / total_slots
    defects_injected = sum(1 for r in inspection_results if r['expected'] != 'pass')
    defects_caught = sum(1 for r in inspection_results if r['expected'] != 'pass' and r['predicted'] == r['expected'])

    print(f"Total Fixture Slots Inspected: {total_slots}")
    print(f"Total Injected Defects:        {defects_injected}")
    print(f"Defects Correctly Caught:      {defects_caught}/{defects_injected} ({(defects_caught/defects_injected if defects_injected > 0 else 1.0):.1%})")
    print(f"Escaped Defects (Missed):      {escaped_defects}")
    print(f"False Alarm Alerts:            {false_alarms}")
    print(f"Overall Fixture Accuracy:      {accuracy:.1%}")
    print("=" * 70)
    print(f"\n[+] Rendered Scene Image:     file://{local_img.resolve()}")
    print(f"[+] Annotated Overlay Image:  file://{annotated_path.resolve()}")
    print(f"[+] Ground Truth Manifest:    file://{local_gt.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
