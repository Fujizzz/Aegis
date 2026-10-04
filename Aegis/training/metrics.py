"""Evaluation protocol aegis_binary_v1; scores are fractions, positive target=1."""
import numpy as np

PROTOCOL = {
    "protocol_id": "aegis_binary_v1", "positive_target": 1,
    "threshold": 0.5, "threshold_policy": "fixed_before_test",
    "decision_rule": "p_positive >= threshold", "zero_division": 0,
    "class_macro_labels": [0, 1], "auc_input": "continuous_p_positive",
    "single_class_auc": None, "empty_input": "error", "scale": "fraction_0_1",
    "selection_metric": "binary_f1", "selection_split": "val", "tie_break": "first_epoch",
}
SCORES = ("accuracy", "positive_precision", "positive_recall", "binary_f1", "class_macro_f1", "roc_auc")


def confusion_metrics(tp, fp, tn, fn):
    counts = [tp, fp, tn, fn]
    if any(not np.isfinite(x) or int(x) != x or x < 0 for x in counts):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, tn, fn = map(int, counts)
    n = tp + fp + tn + fn
    if n == 0:
        raise ValueError("Cannot evaluate empty input")
    divide = lambda a, b: a / b if b else 0.0
    positive_f1 = divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n, "positive_precision": divide(tp, tp + fp),
            "positive_recall": divide(tp, tp + fn), "binary_f1": positive_f1,
            "negative_f1": negative_f1, "class_macro_f1": (positive_f1 + negative_f1) / 2}


def binary_metrics(y_true, y_prob, thr=0.5):
    y, p = np.asarray(y_true), np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Expected nonempty, equal-length 1D targets and probabilities")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be canonical 0/1 (convert source labels first)")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.isfinite(thr) or not 0 <= thr <= 1:
        raise ValueError("Threshold must be finite and in [0,1]")
    pred = p >= thr
    result = confusion_metrics(int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()),
                               int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    auc = None
    if n_pos and n_neg:
        order = np.argsort(p, kind="stable")
        ranks = np.empty(len(p), dtype=float)
        i = 0
        while i < len(p):
            j = i + 1
            while j < len(p) and p[order[j]] == p[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2
            i = j
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    result.update(roc_auc=auc, auc_defined=auc is not None, threshold=float(thr))
    # Compatibility aliases; new tables use explicit names.
    result.update(precision=result["positive_precision"], recall=result["positive_recall"],
                  f1=result["binary_f1"], auc=auc)
    return result


def event_metrics(events, targets, probabilities, threshold=0.5):
    if len(events) != len(targets) or not len(events):
        raise ValueError("Event labels and targets must have equal, nonzero lengths")
    y, p, e = np.asarray(targets), np.asarray(probabilities), np.asarray(events)
    pooled = binary_metrics(y, p, threshold)
    per_event = {str(name): binary_metrics(y[e == name], p[e == name], threshold) for name in sorted(set(events))}
    equal = {k: (None if any(m[k] is None for m in per_event.values()) else
                 float(np.mean([m[k] for m in per_event.values()]))) for k in SCORES}
    return {"per_event": per_event, "event_equal_mean": equal, "sample_pooled": pooled,
            "event_count": len(per_event), "auc_defined_event_count": sum(m["auc_defined"] for m in per_event.values())}
