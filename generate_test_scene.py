#!/usr/bin/env python3
"""CLI tool to generate custom synthetic vial fixture scenes and ground-truth labels.

Usage Examples:
    # 1. Generate scene with specific defects:
    python generate_test_scene.py --defects G05:crooked_cap G12:missing_cap S02:damaged_vial --output my_scene.jpg

    # 2. Generate partially filled tray (e.g. 18 vials filled in Good tray, remaining empty):
    python generate_test_scene.py --fill-good 18 --defects G08:crooked_cap --output partial_tray.jpg
"""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.synthetic_generator import SyntheticSceneGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic vial scenes with injected defects")
    parser.add_argument("--template", default=str(HERE.parent / "vial-labeler-participants-v1/examples/tray_000001.jpg"),
                        help="Path to baseline reference image")
    parser.add_argument("--calibration", default=str(HERE / "calibration.json"), help="Path to calibration.json")
    parser.add_argument("--defects", nargs="*", default=[],
                        help="Defects to inject in format SLOT:CLASS (e.g. G05:crooked_cap G12:missing_cap S03:damaged_vial)")
    parser.add_argument("--fill-good", type=int, default=None,
                        help="Fill count for Good tray (1-30). Positions beyond this are empty ('no_vial')")
    parser.add_argument("--fill-scrap", type=int, default=None,
                        help="Fill count for Scrap tray (1-30). Positions beyond this are empty ('no_vial')")
    parser.add_argument("--jitter", type=float, default=0.05, help="Lighting jitter variance (0.0 to 0.2)")
    parser.add_argument("--output", default="synthetic_test_scene.jpg", help="Path to save generated 1920x1080 JPEG")
    args = parser.parse_args()

    # Parse defects
    defect_specs = {}
    for item in args.defects:
        if ":" not in item:
            print(f"[!] Invalid defect format '{item}'. Use format SLOT:CLASS (e.g. G05:crooked_cap)")
            sys.exit(1)
        slot, cls_name = item.split(":", 1)
        slot = slot.strip().upper()
        defect_specs[slot] = cls_name.strip()

    generator = SyntheticSceneGenerator(args.template, args.calibration)
    print(f"[*] Generating synthetic test scene with defects: {defect_specs or 'None (All Pass)'}...")

    scene, gt_manifest = generator.generate_scene(
        defect_specs=defect_specs,
        fill_count_good=args.fill_good,
        fill_count_scrap=args.fill_scrap,
        lighting_jitter=args.jitter
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.save(out_path, quality=95)
    print(f"[+] Saved synthetic image: {out_path}")

    gt_path = out_path.with_name(out_path.stem + "_gt.json")
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_manifest, f, indent=2)
    print(f"[+] Saved ground truth labels: {gt_path}")

    # Print summary of injected conditions
    print("\n" + "=" * 55)
    print("        SYNTHETIC SCENE SPECIFICATIONS        ")
    print("=" * 55)
    print(f"Image: {out_path.name}")
    print(f"Good Tray Fill: {args.fill_good if args.fill_good is not None else 30}/30")
    print(f"Scrap Tray Fill: {args.fill_scrap if args.fill_scrap is not None else 30}/30")
    print(f"Injected Defects: {len(defect_specs)}")
    for s, c in defect_specs.items():
        print(f"  - Slot {s}: {c}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
