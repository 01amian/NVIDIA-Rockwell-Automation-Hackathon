#!/usr/bin/env python3
"""Interactive All-In-One Vial Inspection Test Cycle.

Generates a custom synthetic fixture scene with chosen defects,
inspects it with the AI pipeline, benchmarks results against ground truth,
and displays a clear position-by-position report with color-coded badges.

Usage Example:
    python3 test_vial_system.py --defects G04:crooked_cap G09:missing_cap G22:no_vial S05:damaged_vial --fill-good 25
"""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.synthetic_generator import SyntheticSceneGenerator
from src.pipeline import VialInspectionPipeline
from src.evaluate import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="All-in-one generate, inspect, and benchmark test cycle")
    parser.add_argument("--defects", nargs="*", default=[],
                        help="Defects to inject as SLOT:CLASS (e.g. G04:crooked_cap G09:missing_cap G22:no_vial S05:damaged_vial)")
    parser.add_argument("--fill-good", type=int, default=None,
                        help="Number of filled vials in Good tray (1-30). Slots beyond this become 'no_vial'")
    parser.add_argument("--fill-scrap", type=int, default=None,
                        help="Number of filled vials in Scrap tray (1-30). Slots beyond this become 'no_vial'")
    parser.add_argument("--template", default=str(HERE.parent / "vial-labeler-participants-v1/examples/tray_000001.jpg"),
                        help="Path to baseline reference image")
    parser.add_argument("--calibration", default=str(HERE / "calibration.json"), help="Path to calibration.json")
    parser.add_argument("--name", default="my_test_run", help="Name prefix for generated test run files")
    parser.add_argument("--open", action="store_true", help="Automatically open browser review page upon completion")
    args = parser.parse_args()

    # Parse defects
    defect_specs = {}
    for item in args.defects:
        if ":" not in item:
            print(f"[!] Invalid defect spec '{item}'. Use format SLOT:CLASS (e.g. G04:crooked_cap)")
            sys.exit(1)
        slot, cls_name = item.split(":", 1)
        defect_specs[slot.strip().upper()] = cls_name.strip()

    run_dir = HERE / "test_runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    img_path = run_dir / f"{args.name}.jpg"
    gt_path = run_dir / f"{args.name}_gt.json"
    pred_dir = run_dir / "inspection_output"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1. GENERATE
    print(f"\n[1/3] Generating synthetic scene with defects: {defect_specs or 'None (All Pass)'}...")
    generator = SyntheticSceneGenerator(args.template, args.calibration)
    scene, gt_manifest = generator.generate_scene(
        defect_specs=defect_specs,
        fill_count_good=args.fill_good,
        fill_count_scrap=args.fill_scrap
    )
    scene.save(img_path, quality=95)
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_manifest, f, indent=2)
    print(f"      Saved test image: {img_path}")

    # 2. INSPECT
    print(f"\n[2/3] Running automated inspection pipeline on {img_path.name}...")
    pipeline = VialInspectionPipeline(args.calibration)
    pred_results = pipeline.inspect_image(
        str(img_path),
        expected_through_good=args.fill_good,
        expected_through_scrap=args.fill_scrap
    )

    pred_json_path = pred_dir / "inspection_summary.json"
    with open(pred_json_path, 'w', encoding='utf-8') as f:
        json.dump(pred_results, f, indent=2)

    # Render and save visual overlay
    overlay = pipeline.render_overlay(str(img_path), pred_results)
    overlay_path = pred_dir / f"annotated_{img_path.name}"
    overlay.save(overlay_path, quality=95)
    print(f"      Saved annotated image: {overlay_path}")

    # Prepare browser review page with pre-populated predictions
    labeler_dir = HERE.parent / "vial-labeler-participants-v1"
    browser_page_path = None
    if labeler_dir.exists():
        import shutil
        sys.path.insert(0, str(labeler_dir))
        import labeler
        from rules import summarize as orig_summarize

        browser_output = run_dir / "browser_review"
        if browser_output.exists():
            shutil.rmtree(browser_output)

        results_layout, sources = labeler.prepare([img_path], profile_id='view_001', offset=(0, 0))
        pred_map = {s['id']: s for s in pred_results['slots']}
        for slot in results_layout[0]['slots']:
            sid = slot['id']
            if sid in pred_map:
                p = pred_map[sid]
                slot['class'] = p['class']
                slot['presence'] = p['presence']
                slot['review_status'] = 'confirmed'
                slot['reason'] = f"AI Prediction ({p.get('confidence', 0.95)*100:.0f}%)"

        results_layout[0]['groups']['good'] = orig_summarize('good', [s for s in results_layout[0]['slots'] if s['area']=='good'])
        results_layout[0]['groups']['scrap'] = orig_summarize('scrap', [s for s in results_layout[0]['slots'] if s['area']=='scrap'])

        labeler.save(results_layout, sources, browser_output)
        browser_page_path = browser_output / "index.html"

    # 3. BENCHMARK & EVALUATION REPORT
    print(f"\n[3/3] Benchmarking inspection results against ground truth...")
    gt_slots = gt_manifest.get('slots', [])
    pred_slots = pred_results.get('slots', [])
    eval_report = evaluate_predictions(gt_slots, pred_slots)

    # Position-by-Position Detailed Table
    print("\n" + "=" * 78)
    print("                 POSITION-BY-POSITION INSPECTION REPORT                 ")
    print("=" * 78)
    print(f"{'Slot':<6} | {'Area':<6} | {'Ground Truth':<14} | {'AI Prediction':<14} | {'Conf':<6} | {'Status'}")
    print("-" * 78)

    gt_map = {s['id']: s['class'] for s in gt_slots}
    any_mismatch = False

    for s in pred_slots:
        slot_id = s['id']
        gt_cls = gt_map.get(slot_id, 'unknown')
        pred_cls = s['class']
        conf = f"{s.get('confidence', 0.95)*100:.0f}%"

        # Only highlight slots with defects, empty positions, or mismatches to keep terminal readable
        is_defect = gt_cls in ['missing_cap', 'crooked_cap', 'damaged_vial', 'no_vial'] or pred_cls != 'pass'

        if gt_cls == pred_cls:
            status = "✅ MATCH"
        elif gt_cls != 'pass' and pred_cls == 'pass':
            status = "🚨 ESCAPED DEFECT (FN)"
            any_mismatch = True
        elif gt_cls == 'pass' and pred_cls != 'pass':
            status = "⚠️  FALSE ALARM (FP)"
            any_mismatch = True
        else:
            status = f"🔄 MISCLASSIFIED ({gt_cls} -> {pred_cls})"
            any_mismatch = True

        if is_defect or any_mismatch:
            print(f"{slot_id:<6} | {s['area']:<6} | {gt_cls:<14} | {pred_cls:<14} | {conf:<6} | {status}")

    print("-" * 78)
    print("Note: Passing vials that matched correctly are omitted above for brevity.")

    # Tray Findings Summary
    good_findings = pred_results['groups']['good']['findings']
    scrap_findings = pred_results['groups']['scrap']['findings']

    print("\n" + "=" * 78)
    print("                    FACTORY RULES & FINDINGS SUMMARY                   ")
    print("=" * 78)
    print("--- GOOD TRAY (G01 - G30) ---")
    print(f"  Occupied Slots: {pred_results['groups']['good']['occupied']}/30")
    print(f"  Complete Order:            {good_findings.get('complete_order')}")
    print(f"  Missing Vial (Holes):      {good_findings.get('missing_vial')}")
    print(f"  Defect in Good Tray:       {good_findings.get('bad_misclassified_as_good')}")

    print("\n--- SCRAP TRAY (S01 - S30) ---")
    print(f"  Occupied Slots: {pred_results['groups']['scrap']['occupied']}/30")
    print(f"  No Problems in Scrap:      {scrap_findings.get('no_problems')}")
    print(f"  Good Routed to Scrap:      {scrap_findings.get('good_misclassified_as_bad')}")
    print(f"  Full Scrap Tray:           {scrap_findings.get('full')}")

    print("\n" + "=" * 78)
    print(f"SCORE SUMMARY: Penalty = {eval_report['penalty_score']:.1f} | Escaped Defects = {eval_report['escaped_defects']} | False Alarms = {eval_report['false_alarms']}")
    print(f"Annotated Image:      file://{overlay_path}")
    if browser_page_path:
        print(f"Browser Review Page:  file://{browser_page_path}")
        print(f"                      (Clickable link above opens the full interactive organizer UI)")
    print("=" * 78 + "\n")

    if args.open and browser_page_path and browser_page_path.exists():
        import subprocess
        subprocess.run(["open", str(browser_page_path)])


if __name__ == "__main__":
    main()
