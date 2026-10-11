#!/usr/bin/env python3
"""
The ceiling on top of BERT-strength text.

Why this exists
---------------
probe_av_features.py measured how much audio+video add on top of TEXT — but its
text was TF-IDF, which reaches only ~+0.068 macro-F1 above a constant predictor.
Your model uses a fine-tuned BERT, which is stronger. Stronger text absorbs more
of what the non-verbal channels would otherwise contribute, so the ceiling
measured against TF-IDF (+0.0178) OVERSTATES what is available to your model.

This script measures the ceiling against the text representation the model
actually uses: the pooled BERT embedding from a trained checkpoint.

    BERT-emb                 vs
    BERT-emb + audio + video

The difference is the headroom a perfect fusion layer could still exploit.
Compare it against what your model already realises (the `only_text` ablation,
or the sum of the audio and video ablation deltas).

It does not modify network.py. The BERT pooling below replicates
network.py:175-182 exactly, using the fine-tuned weights from the checkpoint.

Audio/video features use the same pooling as probe_av_features.py, so the numbers
are directly comparable to that script's output.

Usage
-----
    source ~/tmbl_env/bin/activate
    python3 probe_bert_ceiling.py --checkpoint ~/TMBL/checkpoints/devBest/model_RUN1_dropout03.std
"""

import argparse
import sys
import os 
import numpy as np
import torch


try:
    from sklearn.linear_model import LogisticRegression      # noqa: F401  (used via fit_and_score)
    from sklearn.preprocessing import StandardScaler, normalize as l2norm
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")

from config import get_config
from data_loader import get_loader
import network
from utils.results_io import write_json

# Reuse the probe's pooling and scoring so the protocol is identical by construction
from probe_av_features import pool_clip, fit_and_score, trivial_baseline, print_row, EMOTIONS

SPLITS = ["train", "dev", "test"]


def extract(model, loader):
    """Return (bert_emb [N,768], av_pooled [N,5*Dv+5*Da], labels [N,6]).

    Everything is collected batch by batch, so the three arrays stay aligned even
    though collate_fn re-sorts each batch by utterance length.
    """
    bert_rows, av_rows, lab_rows = [], [], []

    with torch.no_grad():
        for batch in loader:
            v, a, y, l, bert_sent, bert_type, bert_mask = batch

            # ---- text: replicate network.py:175-182 (masked mean over BERT tokens) ----
            ids = bert_sent.cuda()
            mask = bert_mask.cuda()
            out = model.bertmodel(input_ids=ids, attention_mask=mask)[0]     # [B, L, 768]
            m = mask.unsqueeze(2).float()                                    # [B, L, 1]
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)              # [B, 768]
            bert_rows.append(pooled.cpu().numpy())

            # ---- audio / video: same pooling as probe_av_features.py ----
            # v: [T, B, Dv]   a: [T, B, Da]   l: [B] true lengths (padding excluded)
            v_np, a_np, l_np = v.numpy(), a.numpy(), l.numpy()
            for i in range(v_np.shape[1]):
                t = int(l_np[i])
                pv = pool_clip(v_np[:t, i, :])
                pa = pool_clip(a_np[:t, i, :])
                if pv is None or pa is None:                 # zero-length clip
                    pv = np.zeros(5 * v_np.shape[2])
                    pa = np.zeros(5 * a_np.shape[2])
                av_rows.append(np.concatenate([pv, pa]))

            lab_rows.append(y.numpy())

    bert = np.concatenate(bert_rows, 0)
    av = np.asarray(av_rows)
    lab = np.concatenate(lab_rows, 0).reshape(-1, len(EMOTIONS))
    assert bert.shape[0] == av.shape[0] == lab.shape[0], \
        f"misaligned: bert {bert.shape} av {av.shape} labels {lab.shape}"
    return bert, av, (lab > 0).astype(int)


def block(train_mat, mats):
    """Standardise on train, then L2-normalise rows.

    The L2 step matters for the combined model: BERT is 768-dimensional and the
    A/V block 545, so without it the larger block would dominate the regulariser
    purely through its row norm. Applied to every model so the single-block and
    combined numbers stay comparable.
    """
    sc = StandardScaler().fit(train_mat)
    return [l2norm(sc.transform(m)) for m in mats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a trained .std file")
    ap.add_argument("--grid", type=int, default=200, help="cut-off candidates per emotion")
    ap.add_argument("--results_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args, _ = ap.parse_known_args() 

    print("=" * 78)
    print("CEILING ON TOP OF BERT-STRENGTH TEXT")
    print("=" * 78)

    cfgs, loaders = {}, {}
    for s in SPLITS:
        cfgs[s] = get_config(parse=False, mode=s)
        loaders[s] = get_loader(cfgs[s], shuffle=False)      # shuffle=False => drop_last=False
    print(f"  data: {cfgs['train'].data_dir}")

    model = network.AOTN(cfgs["train"]).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state)
    print(f"  checkpoint: {args.checkpoint}")

    print("\n" + "-" * 78)
    print("EXTRACTING FEATURES  (one forward pass per split)")
    print("-" * 78)
    B, A, Y = {}, {}, {}
    for s in SPLITS:
        B[s], A[s], Y[s] = extract(model, loaders[s])
        print(f"  {s:6s}: bert {B[s].shape}   audio+video {A[s].shape}   labels {Y[s].shape}")

    Btr, Bdv, Bte = block(B["train"], [B["train"], B["dev"], B["test"]])
    Atr, Adv, Ate = block(A["train"], [A["train"], A["dev"], A["test"]])
    BAtr = np.hstack([Btr, Atr]); BAdv = np.hstack([Bdv, Adv]); BAte = np.hstack([Bte, Ate])

    print("\n" + "-" * 78)
    print("RESULTS  (fit on train, cut-offs chosen on dev, scored on test)")
    print("-" * 78)
    triv = trivial_baseline(Y["test"])
    header = f"  {'':28s}" + "".join(f"{e[:6]:>8s}" for e in EMOTIONS) + f"{'MACRO':>10s}"
    print(header)
    print_row("say-yes-to-everything", triv)

    rows = []
    rows.append(fit_and_score(Atr, Y["train"], Adv, Y["dev"], Ate, Y["test"],
                              "audio + video only", args.grid))
    rows.append(fit_and_score(Btr, Y["train"], Bdv, Y["dev"], Bte, Y["test"],
                              "BERT text only", args.grid))
    rows.append(fit_and_score(BAtr, Y["train"], BAdv, Y["dev"], BAte, Y["test"],
                              "BERT text + audio + video", args.grid))
    for label, _, f1_tuned in rows:
        print_row(label, f1_tuned)

    tb = float(np.mean(triv))
    av = float(np.mean(rows[0][2]))
    bt = float(np.mean(rows[1][2]))
    bav = float(np.mean(rows[2][2]))

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"  trivial baseline           : {tb:.4f}")
    print(f"  audio + video only         : {av:.4f}   ({av - tb:+.4f} vs trivial)")
    print(f"  BERT text only             : {bt:.4f}   ({bt - tb:+.4f} vs trivial)")
    print(f"  BERT text + audio + video  : {bav:.4f}   ({bav - bt:+.4f} vs BERT TEXT ALONE)")
    print()
    print(f"  >>> CEILING on top of BERT-strength text: {bav - bt:+.4f} macro-F1")
    print()
    print("  Compare against what the model already realises:")
    print("    - the `only_text` ablation delta, if wired, or")
    print("    - the sum of the audio and video ablation deltas (a floor, since")
    print("      single-modality ablations let the other modality compensate).")
    print()
    print("  If realised >= ceiling, the model is already extracting everything")
    print("  available and further fusion work cannot help. If realised is well")
    print("  below, that gap is what a fusion fix could recover.")
    print()
    run_name = os.path.basename(args.checkpoint).replace('model_', '', 1).rsplit('.', 1)[0]
    write_json(os.path.join(args.results_dir, 'metrics', f'ceiling_{run_name}.json'), {
        'checkpoint': args.checkpoint,
        'emotions': list(EMOTIONS),
        'per_emotion_f1': {'trivial': [float(x) for x in triv],
                           **{label: [float(x) for x in f1] for label, _, f1 in rows}},
        'macro_f1': {'trivial': tb, 'audio_video_only': av, 'text_only': bt, 'text_audio_video': bav},
        'ceiling_av_over_text': bav - bt,
    })



if __name__ == "__main__":
    main()
