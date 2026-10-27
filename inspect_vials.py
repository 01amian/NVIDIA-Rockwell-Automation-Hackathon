#!/usr/bin/env python3
"""CLI for Automated Final Vial Inspection.

Usage:
    python inspect_vials.py --image path/to/image.jpg --output-dir results/
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure src is importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.pipeline import VialInspectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Automated Final Vial Inspection")
    parser.add_argument("--image", required=True, help="Path to 1920x1080 fixture image")
    parser.add_argument("--output-dir", default="inspection_results", help="Directory to save outputs")
    parser.add_argument("--expected-good", type=int, default=None, help="Expected placement count in good tray (0-30)")
    parser.add_argument("--expected-scrap", type=int, default=None, help="Expected placement count in scrap tray (0-30)")
    parser.add_argument("--calibration", default=str(HERE / "calibration.json"), help="Path to calibration.json")
    args = parser.parse_args()

    pipeline = VialInspectionPipeline(calibration_path=args.calibration)
    print(f"[*] Running automated vial inspection on {args.image}...")

    results = pipeline.inspect_image(
        args.image,
        expected_through_good=args.expected_good,
        expected_through_scrap=args.expected_scrap
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON result
    json_path = out_dir / "inspection_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved structured inspection results: {json_path}")

    # 2. Render and save annotated image overlay
    overlay = pipeline.render_overlay(args.image, results)
    overlay_path = out_dir / f"annotated_{Path(args.image).name}"
    overlay.save(overlay_path, quality=95)
    print(f"[+] Saved visual inspection overlay: {overlay_path}")

    # 3. Print Console Report
    print("\n" + "=" * 60)
    print("           VIAL FIXTURE INSPECTION REPORT           ")
    print("=" * 60)
    print(f"Image: {results['filename']}")
    
    good = results['groups']['good']
    scrap = results['groups']['scrap']
    
    print("\n--- GOOD TRAY (G01 - G30) ---")
    print(f"  Occupied Slots: {good['occupied']}/30")
    print(f"  Missing Positions (internal gaps): {good['missing_positions']}")
    print(f"  Defect Positions: {good['visible_defect_positions']}")
    print(f"  Findings:")
    for k, v in good['findings'].items():
        print(f"    - {k}: {v}")
        
    print("\n--- SCRAP TRAY (S01 - S30) ---")
    print(f"  Occupied Slots: {scrap['occupied']}/30")
    print(f"  Defect Positions: {scrap['visible_defect_positions']}")
    print(f"  Passing Scrap Positions: {scrap.get('visually_passing_scrap_positions', [])}")
    print(f"  Findings:")
    for k, v in scrap['findings'].items():
        print(f"    - {k}: {v}")
        
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
