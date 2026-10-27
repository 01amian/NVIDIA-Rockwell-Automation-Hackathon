"""Synthetic Scene and Defect Generator for Vial Inspection.

Allows generating photorealistic 1920x1080 inspection scenes with ground truth:
- Injects specified defects at any slot position:
    * crooked_cap (with controllable tilt angle and offset)
    * missing_cap (open mouth without cap)
    * damaged_vial (body deformation, cracks, and crushing)
    * no_vial (empty aperture)
    * pass (acceptable seated cap & body)
- Generates paired ground-truth labels and rules summary.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from .geometry import TrayGeometry
from .rules import summarize, CLASSES


class SyntheticSceneGenerator:
    """Generates synthetic vial inspection test scenes with ground-truth defect annotations."""

    def __init__(self, template_image_path: str, calibration_path: str):
        self.template_path = Path(template_image_path)
        self.geometry = TrayGeometry(calibration_path)
        with Image.open(self.template_path) as im:
            self.base_image = im.convert('RGB')
        self.slots = self.geometry.get_slot_definitions()
        self.slot_map = {s['id']: s for s in self.slots}

    def _create_crooked_cap(self, crop: Image.Image, angle: float = 22.0) -> Image.Image:
        """Tilts the yellow cap, exposing the dark aperture hole underneath."""
        arr = np.array(crop).copy()
        h, w, _ = arr.shape
        cx, cy = w // 2, int(h * 0.42)

        # 1. Reveal dark aperture hole underneath where cap is lifted
        cv2.circle(arr, (cx, cy + 4), int(min(w, h) * 0.28), (35, 42, 52), -1)
        cv2.circle(arr, (cx, cy + 4), int(min(w, h) * 0.30), (130, 140, 155), 2)

        # 2. Extract cap mask
        hsv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([18, 40, 70]), np.array([65, 255, 255]))
        
        # Isolate cap with alpha
        cap_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        cap_np = np.array(crop)
        for c in range(3):
            cap_np[:, :, c] = np.where(mask > 0, cap_np[:, :, c], 0)
        cap_rgba = Image.fromarray(np.dstack([cap_np, mask]))

        # Rotate cap
        rotated_cap = cap_rgba.rotate(angle, expand=False, resample=Image.BICUBIC)
        
        # Paste tilted cap over the hole
        base = Image.fromarray(arr).convert("RGBA")
        base.paste(rotated_cap, (0, -2), rotated_cap)
        return base.convert("RGB")

    def _create_missing_cap(self, crop: Image.Image) -> Image.Image:
        """Removes the yellow cap, exposing the clear glass/plastic vial neck and open mouth."""
        arr = np.array(crop).copy()
        h, w, _ = arr.shape
        cx, cy = w // 2, int(h * 0.42)

        # Replace yellow cap pixels with transparent glass neck reflections & background
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([18, 40, 70]), np.array([65, 255, 255]))
        arr[mask > 0] = np.array([180, 188, 198], dtype=np.uint8)

        # Draw clear glass body cylinder contours
        cv2.line(arr, (cx - 18, cy - 10), (cx - 18, h - 5), (230, 240, 250), 2)
        cv2.line(arr, (cx + 18, cy - 10), (cx + 18, h - 5), (230, 240, 250), 2)

        # Draw dark inner open mouth
        cv2.ellipse(arr, (cx, cy - 10), (16, 7), 0, 0, 360, (40, 48, 60), -1)
        # Specular glass rim
        cv2.ellipse(arr, (cx, cy - 10), (17, 8), 0, 0, 360, (245, 250, 255), 2)
        return Image.fromarray(arr)

    def _create_damaged_vial(self, crop: Image.Image) -> Image.Image:
        """Adds crushing, crack fractures, and body deformation."""
        arr = np.array(crop).copy()
        h, w, _ = arr.shape

        # Jagged high-contrast crack fractures
        pts = np.array([
            [int(w * 0.28), int(h * 0.35)],
            [int(w * 0.52), int(h * 0.52)],
            [int(w * 0.35), int(h * 0.70)],
            [int(w * 0.65), int(h * 0.88)]
        ], np.int32)
        cv2.polylines(arr, [pts], False, (255, 255, 255), 3)
        cv2.polylines(arr, [pts + 1], False, (15, 20, 28), 2)

        # Dark crushed deformation void
        cv2.ellipse(arr, (int(w * 0.5), int(h * 0.62)), (int(w * 0.30), int(h * 0.22)), 20, 0, 360, (30, 36, 48), -1)
        return Image.fromarray(arr)

    def _create_empty_hole(self, crop: Image.Image, area: str = 'good') -> Image.Image:
        """Simulates an authentic empty fixture aperture matching the real camera view."""
        w, h = crop.size
        cx, cy = w // 2, h // 2

        if area == 'good':
            # Light anodized aluminum plate background
            plate = np.full((h, w, 3), (198, 205, 218), dtype=np.uint8)
            noise = (np.random.randn(h, w, 3) * 3).astype(np.int16)
            plate = np.clip(plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Surrounding corner screws / holes
            for ox, oy in [(-36, -36), (36, -36), (-36, 38), (36, 38)]:
                cv2.circle(plate, (cx + ox, cy + oy), 14, (65, 72, 82), -1)

            # Outer aperture chamfer
            cv2.circle(plate, (cx, cy), 24, (135, 145, 158), 2)
            # Inner dark cavity
            cv2.circle(plate, (cx, cy), 22, (48, 54, 65), -1)
            cv2.circle(plate, (cx, cy), 19, (24, 28, 35), -1)
            # Subtle bevel reflection
            cv2.ellipse(plate, (cx, cy), (23, 23), 0, 45, 135, (235, 242, 252), 2)
            return Image.fromarray(plate)
        else:
            # Scrap tray: darker perforated fixture plate
            plate = np.full((h, w, 3), (155, 162, 172), dtype=np.uint8)
            cv2.circle(plate, (cx, cy), int(min(w, h) * 0.38), (35, 42, 52), -1)
            cv2.circle(plate, (cx, cy), int(min(w, h) * 0.40), (120, 130, 142), 2)
            return Image.fromarray(plate)

    def generate_scene(
        self,
        defect_specs: Dict[str, str],
        fill_count_good: Optional[int] = None,
        fill_count_scrap: Optional[int] = None,
        lighting_jitter: float = 0.0
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a 1920x1080 scene image with specified defects.

        defect_specs: dict of slot_id -> defect class name (e.g., {'G05': 'crooked_cap', 'G12': 'missing_cap'})
        fill_count_good: if specified (e.g. 20), slots > 20 are made empty ('no_vial')
        fill_count_scrap: if specified, slots > count are made empty ('no_vial')
        """
        scene = self.base_image.copy()
        ground_truth_slots = []

        for slot in self.slots:
            slot_id = slot['id']
            area = slot['area']
            pos = slot['position']
            bbox = slot['bbox']

            # Determine target class
            if slot_id in defect_specs:
                target_class = defect_specs[slot_id]
            elif area == 'good' and fill_count_good is not None and pos > fill_count_good:
                target_class = 'no_vial'
            elif area == 'scrap' and fill_count_scrap is not None and pos > fill_count_scrap:
                target_class = 'no_vial'
            else:
                target_class = 'pass'

            # Extract base crop
            crop = self.base_image.crop(bbox)

            # Apply defect transformation
            if target_class == 'crooked_cap':
                modified_crop = self._create_crooked_cap(crop, angle=24.0)
            elif target_class == 'missing_cap':
                modified_crop = self._create_missing_cap(crop)
            elif target_class == 'damaged_vial':
                modified_crop = self._create_damaged_vial(crop)
            elif target_class == 'no_vial':
                modified_crop = self._create_empty_hole(crop, area=area)
            else:
                modified_crop = crop

            # Paste modified crop back into scene only if changed
            if target_class != 'pass':
                scene.paste(modified_crop, (bbox[0], bbox[1]))

            presence = False if target_class == 'no_vial' else True
            ground_truth_slots.append({
                'id': slot_id,
                'area': area,
                'position': pos,
                'row': slot['row'],
                'column_from_right': slot['column_from_right'],
                'center': slot['center'],
                'bbox': bbox,
                'class': target_class,
                'presence': presence,
                'review_status': 'confirmed'
            })

        # Apply subtle lighting jitter if requested
        if lighting_jitter > 0:
            enhancer = ImageEnhance.Brightness(scene)
            scene = enhancer.enhance(1.0 + np.random.uniform(-lighting_jitter, lighting_jitter))

        # Compute ground truth rules summary
        good_slots = [s for s in ground_truth_slots if s['area'] == 'good']
        scrap_slots = [s for s in ground_truth_slots if s['area'] == 'scrap']
        good_summary = summarize('good', good_slots, expected_through=fill_count_good)
        scrap_summary = summarize('scrap', scrap_slots, expected_through=fill_count_scrap)

        gt_manifest = {
            'scene_type': 'synthetic',
            'slots': ground_truth_slots,
            'groups': {
                'good': good_summary,
                'scrap': scrap_summary
            }
        }

        return scene, gt_manifest
