"""Geometry, alignment, and coordinate transformations for vial trays."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def compute_center(grid, position):
    """
    Compute pixel center of a slot (1-30) given the 4 corner points of a tray grid.
    Columns fill from right to left (column 0 = rightmost, 5 = leftmost).
    Rows fill top to bottom (row 0 = top, 4 = bottom).
    """
    column, row = divmod(position - 1, 5)
    u, v = 1.0 - column / 5.0, row / 4.0
    tl, tr, bl, br = np.asarray(grid, dtype=float)
    x, y = (1.0 - v) * ((1.0 - u) * tl + u * tr) + v * ((1.0 - u) * bl + u * br)
    return float(x), float(y)


def extract_slot_bbox(center, crop_offsets):
    """
    Calculate bounding box [left, top, right, bottom] given center and offsets.
    crop_offsets: [dx_left, dy_top, dx_right, dy_bottom]
    """
    cx, cy = center
    left = int(round(cx + crop_offsets[0]))
    top = int(round(cy + crop_offsets[1]))
    right = int(round(cx + crop_offsets[2]))
    bottom = int(round(cy + crop_offsets[3]))
    return [left, top, right, bottom]


def anchor_descriptor(im, anchor_boxes, offset=(0, 0)):
    """Compute normalized anchor feature vector for alignment."""
    dx, dy = offset
    values = []
    for left, top, right, bottom in anchor_boxes:
        crop = im.crop((left + dx, top + dy, right + dx, bottom + dy)).convert('L').resize((48, 24))
        values.append(np.asarray(crop, dtype=float).ravel() / 255.0)
    values = np.concatenate(values)
    return (values - values.mean()) / max(values.std(), 0.05)


class TrayGeometry:
    """Manages geometry calibration, alignment, and crop extraction."""

    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        with open(self.calibration_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.expected_size = tuple(self.config.get('image_size', [1920, 1080]))
        self.profiles = {p['id']: p for p in self.config['profiles']}
        self.default_profile = self.config['profiles'][0]['id']

    def get_slot_definitions(self, profile_id=None, offset=(0, 0)):
        """Generate slot coordinate records for all 60 positions (G01-G30, S01-S30)."""
        pid = profile_id or self.default_profile
        profile = self.profiles[pid]
        dx, dy = offset
        slots = []

        for area, prefix in [('good', 'G'), ('scrap', 'S')]:
            area_cfg = profile['areas'][area]
            grid = area_cfg['grid']
            offsets = area_cfg['crop_offsets']

            for pos in range(1, 31):
                cx, cy = compute_center(grid, pos)
                cx += dx
                cy += dy
                col_from_right, row_0 = divmod(pos - 1, 5)
                bbox = extract_slot_bbox((cx, cy), offsets)

                slot_id = f"{prefix}{pos:02d}"
                slots.append({
                    'id': slot_id,
                    'area': area,
                    'position': pos,
                    'row': row_0 + 1,
                    'column_from_right': col_from_right + 1,
                    'center': [round(cx, 2), round(cy, 2)],
                    'bbox': bbox,
                })
        return slots

    def extract_crops(self, image_pil, slots):
        """Extract PIL Image crops for all slots."""
        crops = {}
        for s in slots:
            bbox = s['bbox']
            crop = image_pil.crop(bbox)
            crops[s['id']] = crop
        return crops
