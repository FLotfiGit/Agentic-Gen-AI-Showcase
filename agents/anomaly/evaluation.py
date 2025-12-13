from __future__ import annotations

from typing import List, Dict, Any


def confusion_matrix(pred_indices: List[int], labels: List[int]) -> Dict[str, int]:
    label_set = set(i for i, l in enumerate(labels) if l == 1)
    pred_set = set(pred_indices)
    tp = len(label_set & pred_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    tn = len(labels) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(pred_indices: List[int], labels: List[int]) -> Dict[str, float]:
    cm = confusion_matrix(pred_indices, labels)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
