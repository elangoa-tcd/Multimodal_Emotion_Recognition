# Multimodal Sentiment Analysis
This repository records the proposed Multimodal Sentiment Analysis on Pytorch Framework.

“TMBL: Transformer-based multimodal binding learning model for multimodal sentiment analysis”, paper has been accepted by Knowledge-Based System.
> 🏆 Official implementation of our paper accepted at **Knowleage-Based Systems**  
>
> 📄 DOI: [10.1016/j.knosys.2023.111346](https://doi.org/10.1016/j.knosys.2023.111346)


## Trackers
Here is a list of the improved trackers.

* MISA: Misa: Modality-invariant and-specific representations for multimodal sentiment analysis, ACMM, 2020. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394171.3413678)
* MMIM: Progressive image deraining networks: A better and simpler baseline, arXiv, 2021. [[Paper]](https://arxiv.org/abs/2109.00412)
* HyCon: Hybrid contrastive learning of tri-modal representation for multimodal sentiment analysis, IEEE Transactions on Affective Computing, 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9767560)  
* TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis, Pattern Recognition, 2023. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320322007385)
* AOBERT: All-modalities-in One BERT for multimodal sentiment analysis, 2023. [[Paper]](https://www.sciencedirect.com/science/article/pii/S1566253522002329)  

## Findings
* In this study, we develop a transformer model equipped with a modality-bound learning mechanism to extract unified affective information from multiple modalities. Our proposed structure and method hold potential for broader applications in other fields related to multimodal interaction and fusion, such as multimodal disease diagnosis.



* If you have any question, please set the issue and PR, we will process it as soon as possible~

---
---

# Running this code

This fork adapts TMBL from **sentiment regression** to **six-way emotion detection** on CMU-MOSEI,
and adds diagnostic tooling for measuring how much each modality actually contributes.

Full analysis of the results lives in [`MD_Files/RESULTS_CONSOLIDATED.md`](MD_Files/RESULTS_CONSOLIDATED.md).

> Paths below use `~/TMBL` for the code and `~/data/MOSEI` for the data. Substitute your own and
> update `config.py` and `solver.py` to match (see §3).

---

## 1. Environment

```bash
python3 -m venv ~/tmbl_env
source ~/tmbl_env/bin/activate

# PyTorch with CUDA — match the index-url to your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt --upgrade-strategy only-if-needed
pip install setuptools scikit-learn
```

Install the CMU MultimodalSDK from the bundled copy:

```bash
cd ~/TMBL/CMU-MultimodalSDK-main
pip install -e . --no-build-isolation
```

Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "from mmsdk import mmdatasdk; print('sdk ok')"
```

Both must succeed — `torch.cuda.is_available()` returning `False` means training will fall back to
CPU and take days.

**Remember to `source ~/tmbl_env/bin/activate` in every new shell.**

---

## 2. Data

### 2.1 Required files

The CMU servers are unreliable; download the `.csd` files separately (Kaggle mirrors exist) and
place them in `~/data/MOSEI/` under **exactly** these names:

```
CMU_MOSEI_TimestampedWords.csd
CMU_MOSEI_Labels.csd
CMU_MOSEI_COVAREP.csd
CMU_MOSEI_VisualOpenFace2.csd     # or CMU_MOSEI_VisualFacet42.csd — see §2.3
```

A misnamed file causes a silent failure rather than an error — see §7.

### 2.2 GloVe placeholder

`load_emb()` expects a GloVe text file, but the active model uses BERT and never consumes it. An
empty file is sufficient:

```bash
touch ~/data/MOSEI/glove_dummy.txt
```

### 2.3 Building feature variants (recommended)

`create_dataset.py` z-normalises each utterance against its **own** statistics
([`create_dataset.py:378-381`](create_dataset.py#L378-L381)). This forces every clip to mean 0 and
standard deviation 1, which removes the absolute level of the audio and visual signal — measurably
about **70% of the extractable non-verbal signal**.

`build_variant.py` rebuilds the dataset without that step. It writes to a new directory, never
downloads anything, refuses to overwrite existing `.pkl` files, and counts every dropped clip by
reason.

```bash
# corpus-level normalisation + FACET visual features (the configuration used for all final results)
python3 build_variant.py --out ~/data/MOSEI_C_both \
    --norm corpus --visual_field CMU_MOSEI_VisualFacet42

# reproduce the original preprocessing, for comparison
python3 build_variant.py --out ~/data/MOSEI_B_facet \
    --norm clip --visual_field CMU_MOSEI_VisualFacet42
```

| Flag | Values | Meaning |
|---|---|---|
| `--norm` | `clip` | original: each clip normalised against itself |
| | `corpus` | normalised against train-split statistics — **recommended** |
| | `none` | raw features |
| `--visual_field` | `CMU_MOSEI_VisualFacet42` | 35-d action-unit features — **recommended** |
| | `CMU_MOSEI_VisualOpenFace2` | 713-d; no better, and inflates the video branch enormously |
| `--common_with` | a directory | restrict to segment IDs shared with another variant |
| `--force` | — | overwrite existing output (avoid; delete the files instead) |

**After building, read the `FEATURE CHECK` block.** With `--norm corpus`, `avg|per-clip mean|` must
be clearly above zero. If it still reads ~0.000001, the change did not take effect.

> ⚠️ OpenFace2 needs roughly 30 GB of RAM to build (`.csd` files are float32 on disk and load as
> float64). FACET needs about 13 GB.

---

## 3. Configuration

| What | Where | Notes |
|---|---|---|
| Data directory | [`config.py:28`](config.py#L28) | point `'mosei'` at your variant directory |
| GloVe placeholder | [`config.py:15`](config.py#L15) | `word_emb_path` |
| SDK path | [`config.py:25`](config.py#L25) | `sdk_dir` |
| Text encoder | [`config.py:14`](config.py#L14) | `bert-base-uncased` or `roberta-large` |
| Checkpoint directory | [`solver.py:21`](solver.py#L21) | `checkpoint_path` — must contain `devBest/` |
| Log file | [`solver.py:25`](solver.py#L25) | `log_file` |

Create the checkpoint directory once:

```bash
mkdir -p ~/TMBL/checkpoints/devBest
```

### Switching between emotion and sentiment

The two tasks need **three** coordinated edits — changing only one produces a silent mismatch:

| | Emotion (current) | Sentiment |
|---|---|---|
| [`config.py:139`](config.py#L139) | `num_classes = 6` | `num_classes = 1` |
| [`data_loader.py:64`](data_loader.py#L64) | `sample[1][:, 1:7]` (6 emotion columns) | `sample[1][:, 0:1]` (sentiment column) |
| [`solver.py`](solver.py) `calc_metrics` | per-emotion detection metrics | the commented-out MAE / correlation / pos-neg version |

---

## 4. Training

```bash
cd ~/TMBL
source ~/tmbl_env/bin/activate
python3 train.py
```

For long runs, detach so the job survives an SSH disconnect:

```bash
nohup python3 -u train.py --modality_dropout 0.3 --ablate none \
  --name my_run --patience 15 --trials 5 \
  > ~/TMBL/test_results/my_run_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "PID $!"
tail -f ~/TMBL/test_results/my_run_*.log     # Ctrl+C stops watching, not the run
```

### Command-line options

| Flag | Default | Purpose |
|---|---|---|
| `--name` | timestamp | names the checkpoint `devBest/model_<name>.std` — **use a unique one per run** |
| `--modality_dropout` | `0.0` | probability of zeroing the text feature per sample during training, forcing the loss onto audio and video. `0.3` gave the best results |
| `--ablate` | `none` | train without a modality (see §6) |
| `--patience` | 6 | epochs without dev improvement before the LR is halved and the best model reloaded |
| `--trials` | 3 | how many times that may happen before stopping |
| `--n_epoch` | 300 | upper bound; early stopping normally fires around epoch 90 |
| `--dropout` | 0.5 | ordinary dropout (unrelated to `--modality_dropout`) |
| `--learning_rate` | 1e-4 | |


## 5. Evaluation and threshold tuning

The default decision boundary (raw output > 0) is arbitrary and badly placed for rare emotions.
Tuning it per emotion on dev is worth **+0.05 macro-F1** with no retraining.

```bash
python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_my_run.std
```

Prints test metrics at the default cut-off, the cut-offs chosen on dev, and test metrics at those
cut-offs. Always compare models at the **same** cut-off setting.

---

## 6. Ablation studies

**Test-time ablation** — takes one trained model and zeros a modality at inference. Minutes.
Answers *does the trained model rely on this?*

```bash
for M in none text audio video only_text; do
  echo "########## ablate = $M ##########"
  python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_my_run.std --ablate $M
done 2>&1 | tee ~/TMBL/test_results/ablation_my_run.log
```

### Ceiling probe

```bash
python3 probe_bert_ceiling.py --checkpoint checkpoints/devBest/model_my_run.std
```

Extracts the pooled BERT embedding from a trained checkpoint and measures how much audio and video
add **on top of it** — the headroom any fusion improvement could possibly recover. Compare against
what the model already achieves (`--ablate only_text`). If they match, the model is already
extracting everything available.

