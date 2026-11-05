"""Write evaluation outputs as plain CSV/JSON so metrics can be recomputed without retraining."""
import csv
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def emotion_metrics(true_bin, pred_bin):
    """Per-emotion metrics plus a MACRO row. true_bin, pred_bin: 0/1 int arrays of shape [N, 6].

    f1           : positive-class F1, the metric used throughout the dissertation
    weighted_acc : (TPR + TNR) / 2, the 'weighted accuracy' of Tong et al. (2017) used by Akhtar et al. (2019)
    weighted_f1  : support-weighted F1 over present/absent (sklearn average='weighted'), as in Akhtar et al.
    """
    rows = []
    for i, emo in enumerate(EMOTIONS):
        t, p = true_bin[:, i], pred_bin[:, i]
        rows.append({
            'emotion': emo,
            'support': int(t.sum()),
            'accuracy': accuracy_score(t, p),
            'precision': precision_score(t, p, zero_division=0),
            'recall': recall_score(t, p, zero_division=0),
            'f1': f1_score(t, p, zero_division=0),
            'weighted_acc': balanced_accuracy_score(t, p),
            'weighted_f1': f1_score(t, p, average='weighted', zero_division=0),
        })
    macro = {'emotion': 'MACRO', 'support': ''}
    for key in ('accuracy', 'precision', 'recall', 'f1', 'weighted_acc', 'weighted_f1'):
        macro[key] = float(np.mean([r[key] for r in rows]))
    return rows + [macro]


def prediction_rows(segment_ids, logits, labels):
    """One row per clip: raw logits (so any cut-off can be re-applied) and raw label intensities."""
    rows = []
    for sid, lg, lb in zip(segment_ids, logits, labels):
        row = {'segment_id': sid}
        row.update({f'logit_{e}': float(v) for e, v in zip(EMOTIONS, lg)})
        row.update({f'label_{e}': float(v) for e, v in zip(EMOTIONS, lb)})
        rows.append(row)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")
