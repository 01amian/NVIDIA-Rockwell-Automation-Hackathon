"""Computer Vision Defect Classifier for Vial Fixture Positions.

Evaluates each 80x90 or 170x130 crop into 5 target classes:
- pass: Visible cap and vial body appear acceptable.
- missing_cap: Vial is present, but mouth is open / cap is missing.
- crooked_cap: Cap is visibly tilted or improperly seated relative to the mouth.
- damaged_vial: Body crushed, cracked, broken, or deformed.
- no_vial: Aperture is empty.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image


class BaselineVialClassifier:
    """Feature-based computer vision classifier for vial slot crops."""

    def __init__(self, tilt_threshold_deg: float = 7.0):
        self.tilt_threshold_deg = tilt_threshold_deg

    def classify_crop(self, crop_pil: Image.Image, slot_info: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a single vial slot crop.
        
        Returns:
            (predicted_class, confidence_score, feature_details)
        """
        crop_np = np.array(crop_pil.convert('RGB'))
        h, w, _ = crop_np.shape

        # Convert color spaces
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)

        # Region partitions: Cap region (upper 45%) vs Body region (lower 55%)
        cap_h = int(h * 0.45)
        cap_roi = crop_np[:cap_h, :]
        body_roi = crop_np[cap_h:, :]

        # 1. Occupancy Analysis (Empty hole vs present vial)
        # Vial presence reflects light, specular reflections, or colored cap
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / float(h * w)

        # Cap color mask: Factory yellow/lemon caps have Hue ~18-65 in HSV
        lower_cap_color = np.array([18, 40, 70])
        upper_cap_color = np.array([65, 255, 255])
        cap_mask = cv2.inRange(hsv, lower_cap_color, upper_cap_color)
        cap_color_ratio = np.sum(cap_mask > 0) / float(h * w)

        gray_std = float(np.std(gray))
        body_gray = gray[cap_h:, :]
        body_std = float(np.std(body_gray))

        features = {
            'edge_density': round(float(edge_density), 4),
            'cap_color_ratio': round(float(cap_color_ratio), 4),
            'gray_std': round(float(gray_std), 4),
            'body_std': round(float(body_std), 4),
            'tilt_angle': 0.0,
        }

        # Decision 1: Occupancy & Cap Presence
        if cap_color_ratio < 0.05:
            # Distinguish empty aperture (no_vial) from uncapped vial (missing_cap)
            # Empty circular aperture has very low edge complexity (< 0.10)
            # A transparent vial body has multiple reflection edges & glass rim (> 0.10)
            if edge_density < 0.10:
                return 'no_vial', 0.98, features
            else:
                return 'missing_cap', 0.95, features

        # Decision 2: Cap Tilt / Crookedness Analysis
        ys, xs = np.where(cap_mask > 0)
        if len(xs) > 100:
            unique_xs = np.unique(xs)
            top_ys = [ys[xs == x].min() for x in unique_xs]
            if len(unique_xs) > 15:
                slope, _ = np.polyfit(unique_xs, top_ys, 1)
                slope_angle = abs(np.degrees(np.arctan(slope)))
                features['slope_angle'] = round(float(slope_angle), 2)

                if slope_angle > 9.0:
                    return 'crooked_cap', 0.92, features

        # Decision 3: Body Damage / Deformation
        body_edges = cv2.Canny(body_gray, 50, 150)
        body_edge_density = np.sum(body_edges > 0) / float(body_gray.size)
        features['body_edge_density'] = round(float(body_edge_density), 4)

        if edge_density > 0.13 and gray_std > 68.0:
            return 'damaged_vial', 0.85, features

        # Default: Passing vial
        return 'pass', 0.95, features


class DeepLearningVialClassifier:
    """MobileNetV3 deep learning classifier trained on digital twin crops."""

    def __init__(self, model_path: str = "models/vial_mobilenet.pth", device: str = None, defect_threshold: float = 0.70):
        import torch
        from torchvision import transforms, models
        from .rules import CLASSES

        self.classes = CLASSES
        self.defect_threshold = defect_threshold
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False

        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = torch.nn.Linear(in_features, len(self.classes))

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_crop(self, crop_pil: Image.Image, slot_info: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        import torch
        tensor = self.transform(crop_pil.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs))
            conf = float(probs[pred_idx])

        pred_class = self.classes[pred_idx]
        raw_pred = pred_class
        if pred_class != 'pass' and conf < self.defect_threshold:
            pred_class = 'pass'

        features = {
            'model': 'MobileNetV3',
            'raw_prediction': raw_pred,
            'confidence': round(conf, 4),
            'probabilities': {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))}
        }
        return pred_class, conf, features
