"""End-to-End Automated Vial Inspection Pipeline.

Integrates:
1. Fixture geometric calibration & crop extraction
2. Defect classification per slot
3. Factory rule evaluation & tray-level findings
4. Exporting compliant labels.json and annotated visual inspection reports
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import TrayGeometry
from .classifier import BaselineVialClassifier
from .rules import summarize, CLASSES, DEFECTS


COLORS = {
    'pass': '#20a875',         # Green
    'missing_cap': '#f29a32',  # Orange
    'crooked_cap': '#bf78ec',  # Purple
    'damaged_vial': '#f15b65', # Red
    'no_vial': '#60a5fa',      # Blue
    None: '#aab4c4'            # Grey
}


class VialInspectionPipeline:
    """Full industrial automated inspection pipeline for vial fixtures."""

    def __init__(self, calibration_path: Optional[str] = None):
        if calibration_path is None:
            # Look in parent or local folder
            candidate = Path(__file__).resolve().parent.parent / "calibration.json"
            if not candidate.exists():
                candidate = Path(__file__).resolve().parent.parent.parent / "vial-labeler-participants-v1/calibration.json"
            calibration_path = str(candidate)

        self.geometry = TrayGeometry(calibration_path)
        self.classifier = BaselineVialClassifier()

    def inspect_image(
        self,
        image_path: str,
        expected_through_good: Optional[int] = None,
        expected_through_scrap: Optional[int] = None,
        upstream_verdicts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run full automated inspection on a 1920x1080 scene.
        """
        img_path = Path(image_path)
        with Image.open(img_path) as im:
            image_rgb = im.convert('RGB')

        if image_rgb.size != self.geometry.expected_size:
            raise ValueError(f"Expected {self.geometry.expected_size}, got {image_rgb.size}")

        slots = self.geometry.get_slot_definitions()
        crops = self.geometry.extract_crops(image_rgb, slots)

        # 1. Classify each of the 60 slots
        processed_slots = []
        for s in slots:
            slot_id = s['id']
            crop_pil = crops[slot_id]
            pred_class, conf, features = self.classifier.classify_crop(crop_pil, s)

            presence = False if pred_class == 'no_vial' else True
            slot_record = {
                'id': slot_id,
                'area': s['area'],
                'position': s['position'],
                'row': s['row'],
                'column_from_right': s['column_from_right'],
                'center': s['center'],
                'bbox': s['bbox'],
                'class': pred_class,
                'presence': presence,
                'confidence': conf,
                'review_status': 'confirmed',
                'features': features
            }
            processed_slots.append(slot_record)

        # 2. Evaluate Factory Rules for Good & Scrap trays
        good_slots = [s for s in processed_slots if s['area'] == 'good']
        scrap_slots = [s for s in processed_slots if s['area'] == 'scrap']

        good_summary = summarize('good', good_slots, expected_through=expected_through_good)
        scrap_summary = summarize('scrap', scrap_slots, expected_through=expected_through_scrap, upstream=upstream_verdicts)

        result = {
            'filename': img_path.name,
            'source_path': str(img_path),
            'image_size': list(self.geometry.expected_size),
            'slots': processed_slots,
            'groups': {
                'good': good_summary,
                'scrap': scrap_summary
            }
        }
        return result

    def render_overlay(self, image_path: str, inspection_result: Dict[str, Any]) -> Image.Image:
        """Render annotated visual inspection overlay with bounding rings, labels, and findings."""
        with Image.open(image_path) as im:
            base = im.convert('RGB')

        # Add top banner and bottom legend
        out = Image.new('RGB', (1920, 1220), '#101c2d')
        out.paste(base, (0, 100))
        d = ImageDraw.Draw(out)

        font_large = ImageFont.load_default()
        filename = inspection_result.get('filename', 'inspection')

        # Header info
        d.text((30, 15), f"AUTOMATED VIAL INSPECTION: {filename}", fill='white')
        good_findings = inspection_result['groups']['good']['findings']
        scrap_findings = inspection_result['groups']['scrap']['findings']

        g_status = "PASS" if good_findings.get('complete_order') else ("DEFECT" if good_findings.get('bad_misclassified_as_good') else "INCOMPLETE")
        s_status = "MISROUTE DETECTED" if scrap_findings.get('good_misclassified_as_bad') else "OK"

        d.text((30, 45), f"Good Tray: {g_status} | Scrap Tray: {s_status}", fill='#84ccff')

        # Draw rings on slots
        for s in inspection_result['slots']:
            cx, cy = s['center']
            cy += 100 # offset for banner
            cls = s['class']
            color = COLORS.get(cls, '#aab4c4')

            d.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), outline=color, width=3)
            d.text((cx - 15, cy - 8), s['id'], fill=color)

        # Legend at bottom
        x = 30
        for k, c in COLORS.items():
            if k is not None:
                d.rectangle((x, 1180, x + 16, 1196), fill=c)
                d.text((x + 24, 1182), k.replace('_', ' ').upper(), fill='white')
                x += 220

        return out
