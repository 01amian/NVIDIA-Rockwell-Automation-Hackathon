"""Evaluation metrics for final vial inspection.

Calculates:
- False Negatives / Escaped Defects (the most critical hackathon metric)
- False Alarm Rate (false positives on good vials)
- Per-class Precision, Recall, and F1 Score
- Weighted Competition Penalty Score
"""

from typing import List, Dict, Any
import numpy as np

from .rules import CLASSES, DEFECTS


def evaluate_predictions(
    ground_truth_slots: List[Dict[str, Any]],
    predicted_slots: List[Dict[str, Any]],
    missed_defect_weight: float = 10.0,
    false_alarm_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate predicted slots against confirmed ground truth slots.
    """
    gt_map = {s['id']: s['class'] for s in ground_truth_slots}
    pred_map = {s['id']: s['class'] for s in predicted_slots}

    tp_defects = 0
    fn_defects = 0  # ESCAPED DEFECT: Ground truth is defect, predicted as pass or no_vial
    fp_defects = 0  # FALSE ALARM: Ground truth is pass, predicted as defect
    total_defects = 0
    total_passes = 0

    class_counts = {c: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for c in CLASSES}

    for slot_id, gt_class in gt_map.items():
        if gt_class is None:
            continue # Ignore unreviewed positions

        pred_class = pred_map.get(slot_id)
        class_counts[gt_class]['gt'] += 1

        if gt_class == pred_class:
            class_counts[gt_class]['tp'] += 1
        else:
            class_counts[gt_class]['fn'] += 1
            if pred_class in class_counts:
                class_counts[pred_class]['fp'] += 1

        # Defect-level scoring
        is_gt_defect = gt_class in DEFECTS
        is_pred_defect = pred_class in DEFECTS

        if is_gt_defect:
            total_defects += 1
            if is_pred_defect:
                tp_defects += 1
            else:
                fn_defects += 1
        elif gt_class == 'pass':
            total_passes += 1
            if is_pred_defect:
                fp_defects += 1

    escaped_defect_rate = (fn_defects / total_defects) if total_defects > 0 else 0.0
    false_alarm_rate = (fp_defects / total_passes) if total_passes > 0 else 0.0
    penalty_score = (fn_defects * missed_defect_weight) + (fp_defects * false_alarm_weight)

    per_class_metrics = {}
    for c in CLASSES:
        tp = class_counts[c]['tp']
        fp = class_counts[c]['fp']
        fn = class_counts[c]['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics[c] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'count': class_counts[c]['gt']
        }

    return {
        'total_defects': total_defects,
        'escaped_defects': fn_defects,
        'escaped_defect_rate': round(escaped_defect_rate, 4),
        'total_passes': total_passes,
        'false_alarms': fp_defects,
        'false_alarm_rate': round(false_alarm_rate, 4),
        'penalty_score': penalty_score,
        'per_class': per_class_metrics
    }
