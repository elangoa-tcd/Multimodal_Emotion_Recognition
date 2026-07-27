"""Post-hoc per-emotion threshold tuning. No retraining.
Sweeps the decision cut-off on DEV to maximise positive-class F1,
then reports TEST metrics at those cut-offs vs the default (logit > 0).
Also supports test-time modality ablation via --ablate (see Part B)."""
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from jack_transformer_share import _BRANCH_COUNTS
from config import get_config
from data_loader import get_loader
import network

EMO = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def collect(model, loader):
    """Run the model over a loader; return (logits [N,6], raw labels [N,6])."""
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            v, a, y, l, bert_sent, bert_type, bert_mask = batch
            out = model(v.cuda(), a.cuda(), l,
                        bert_sent.cuda(), bert_type.cuda(), bert_mask.cuda())
            logits.append(out.detach().cpu().numpy())
            labels.append(y.numpy())
    return (np.concatenate(logits, 0).reshape(-1, 6),
            np.concatenate(labels, 0).reshape(-1, 6))


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
    args = ap.parse_args()

    dev_cfg = get_config(parse=False, mode="dev")
    test_cfg = get_config(parse=False, mode="test")
    dev_loader = get_loader(dev_cfg, shuffle=False)     # also sets visual_size/acoustic_size
    if args.shuffle_eval:
        torch.manual_seed(args.shuffle_eval)
    test_loader = get_loader(test_cfg, shuffle=bool(args.shuffle_eval))

    dev_cfg.ablate = args.ablate                        # read by network.alignment() (Part B)
    model = network.AOTN(dev_cfg).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state)
    print(f"loaded {args.checkpoint}  (ablate={args.ablate})")

    dev_logit, dev_lab = collect(model, dev_loader)
    test_logit, test_lab = collect(model, test_loader)
    dev_true, test_true = (dev_lab > 0).astype(int), (test_lab > 0).astype(int)

    # ---- baseline: the current fixed cut-off ----
    base = report(test_true, (test_logit > 0).astype(int),
                  "TEST @ default cut-off (logit > 0)")

    # ---- sweep each emotion's cut-off on DEV ----
    thresholds = []
    for i in range(6):
        cands = np.percentile(dev_logit[:, i], np.linspace(0.5, 99.5, args.grid))
        scores = [f1_score(dev_true[:, i], (dev_logit[:, i] > c).astype(int),
                           zero_division=0) for c in cands]
        thresholds.append(float(cands[int(np.argmax(scores))]))
    print("\nchosen cut-offs (from DEV):")
    for e, t in zip(EMO, thresholds):
        print(f"  {e:10s} {t:+.4f}")

    tuned = report(test_true,
                   (test_logit > np.array(thresholds)).astype(int),
                   "TEST @ tuned cut-offs (chosen on DEV)")
    print(f"\nmacro-F1: {base:.4f} (default) -> {tuned:.4f} (tuned)   delta {tuned-base:+.4f}")
    print("\nBRANCH COUNTS:", _BRANCH_COUNTS)