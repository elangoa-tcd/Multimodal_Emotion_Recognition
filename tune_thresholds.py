""" per-emotion threshold tuning. No retraining."""
import argparse
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from jack_transformer_share import _BRANCH_COUNTS
from config import get_config
from data_loader import get_loader
import network
from utils.results_io import emotion_metrics, prediction_rows, write_csv, write_json

EMO = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def collect(model, loader):
    """Run the model over a loader; return (segment ids, logits [N,6], raw labels [N,6])."""
    ids, logits, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            v, a, y, l, bert_sent, bert_type, bert_mask, seg_ids = batch
            out = model(v.cuda(), a.cuda(), l,
                        bert_sent.cuda(), bert_type.cuda(), bert_mask.cuda())
            ids.extend(seg_ids)
            logits.append(out.detach().cpu().numpy())
            labels.append(y.numpy())
    return (ids,np.concatenate(logits, 0).reshape(-1, 6), np.concatenate(labels, 0).reshape(-1, 6))


def report(true_bin, pred_bin, title):
    print(f"\n===== {title} =====")
    print(f"{'emotion':10s} {'support':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
    f1s, recs = [], []
    for i, e in enumerate(EMO):
        t, p = true_bin[:, i], pred_bin[:, i]
        acc = accuracy_score(t, p)
        pr = precision_score(t, p, zero_division=0)
        rc = recall_score(t, p, zero_division=0)
        f1 = f1_score(t, p, zero_division=0)
        f1s.append(f1); recs.append(rc)
        print(f"{e:10s} {int(t.sum()):8d} {acc:7.3f} {pr:7.3f} {rc:7.3f} {f1:7.3f}")
    print(f"{'MACRO':10s} {'':8s} {'':7s} {'':7s} {np.mean(recs):7.3f} {np.mean(f1s):7.3f}")
    return float(np.mean(f1s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a .std model file")
    ap.add_argument("--ablate", default="none", choices=["none", "text", "audio", "only_text", "video",'shuffle_text', 'shuffle_audio', 'shuffle_video'])
    ap.add_argument("--grid", type=int, default=200, help="threshold candidates per emotion")
    ap.add_argument("--shuffle_eval", type=int, default=0,help="0 = fixed batch order. N>0 = reshuffle the TEST batches using N as the seed.")
    ap.add_argument("--run_name", default=None,help="name used for output files; default: taken from the checkpoint, model_<name>.std")
    ap.add_argument("--results_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

    args, _ = ap.parse_known_args()   # shared flags (--text_encoder, --pooling, --data_dir) go to get_config

    dev_cfg = get_config(parse=False, mode="dev")
    test_cfg = get_config(parse=False, mode="test")
    assert dev_cfg.task == 'emotion', "threshold tuning applies to the 6-emotion task only"
    dev_loader = get_loader(dev_cfg, shuffle=False)     # also sets visual_size/acoustic_size
    if args.shuffle_eval:
        torch.manual_seed(args.shuffle_eval)
    test_loader = get_loader(test_cfg, shuffle=bool(args.shuffle_eval))

    dev_cfg.ablate = args.ablate                        # read by network.alignment() (Part B)
    model = network.AOTN(dev_cfg).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state)
    print(f"loaded {args.checkpoint}  (ablate={args.ablate})")

    dev_ids, dev_logit, dev_lab = collect(model, dev_loader)
    test_ids, test_logit, test_lab = collect(model, test_loader)
    dev_true, test_true = (dev_lab > 0).astype(int), (test_lab > 0).astype(int)

    # baseline: the current fixed cut-off 
    base = report(test_true, (test_logit > 0).astype(int),
                  "TEST @ default cut-off (logit > 0)")

    # sweep each emotion's cut-off on DEV 
    thresholds = []
    for i in range(6):
        cands = np.percentile(dev_logit[:, i], np.linspace(0.5, 99.5, args.grid))
        scores = [f1_score(dev_true[:, i], (dev_logit[:, i] > c).astype(int),
                           zero_division=0) for c in cands]
        thresholds.append(float(cands[int(np.argmax(scores))]))
    print("\nchosen cut-offs (from DEV):")
    for e, t in zip(EMO, thresholds):
        print(f"  {e:10s} {t:+.4f}")

    tuned = report(test_true, (test_logit > np.array(thresholds)).astype(int),"TEST @ tuned cut-offs (chosen on DEV)")
    print(f"\nmacro-F1: {base:.4f} (default) -> {tuned:.4f} (tuned)   delta {tuned-base:+.4f}")
    print("\nBRANCH COUNTS:", _BRANCH_COUNTS)

    
    # ---- save everything needed to recompute these numbers without a GPU ----
    run_name = args.run_name or os.path.basename(args.checkpoint).replace('model_', '', 1).rsplit('.', 1)[0]
    tag = run_name if args.ablate == 'none' else f"{run_name}__ablate-{args.ablate}"
    R = args.results_dir
    print(f"\nsaving results as '{tag}'")
    write_csv(os.path.join(R, 'predictions', f'{tag}_dev.csv'), prediction_rows(dev_ids, dev_logit, dev_lab))
    write_csv(os.path.join(R, 'predictions', f'{tag}_test.csv'), prediction_rows(test_ids, test_logit, test_lab))
    write_csv(os.path.join(R, 'metrics', f'{tag}_test_default.csv'),
              emotion_metrics(test_true, (test_logit > 0).astype(int)))
    write_csv(os.path.join(R, 'metrics', f'{tag}_test_tuned.csv'),
              emotion_metrics(test_true, (test_logit > np.array(thresholds)).astype(int)))
    write_json(os.path.join(R, 'thresholds', f'{tag}.json'),
               {'checkpoint': args.checkpoint, 'ablate': args.ablate, 'grid': args.grid,
                'thresholds': dict(zip(EMO, thresholds)),
                'macro_f1_default': base, 'macro_f1_tuned': tuned})

