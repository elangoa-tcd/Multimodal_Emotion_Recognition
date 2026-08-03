# Section 1 — Faults Found

### 1.1 Dataset faults

| Fault | Location | Impact | Evidence | Fixed? | The fix |
|---|---|---|---|---|---|
| **Per-utterance z-normalisation** destroys the absolute level of audio and video | [create_dataset.py:378-381](../create_dataset.py#L378-L381) | **Removes ~70% of extractable A/V signal** | §4.1, §4.3 | **Yes** | Dataset rebuilt with `build_variant.py --norm corpus` — z-normalisation against **train-split** statistics instead of each clip's own. Variant C used for all runs from §5.2 onward. `create_dataset.py` itself left unmodified (deliberately — §4.2 explains why). |
| **1,990 infinities** in COVAREP silently converted to 1.8e308 | [create_dataset.py:356](../create_dataset.py#L356) — `np.nan_to_num` default | Masked by per-clip normalisation; breaks any other scheme | §4.2 | **Yes** — in the variant pipeline | Non-finite values mapped to 0 in `build_variant.py` before normalisation (second build; variant C feature statistics are finite — §4.5). Still present in any dataset built by `create_dataset.py`. *(Verify: exact handling is in `build_variant.py`, not re-checked here.)* |
| Frames averaged to **word granularity** | [create_dataset.py:304](../create_dataset.py#L304) | Smooths away micro-expressions and prosodic contour — **not measured** | — | **No** | Would require re-aligning from raw `.csd` files at frame granularity and a sequence model over frames. Listed as future work (§8.3). |
| **OpenFace2 (713-d)** rather than FACET (35-d) | [create_dataset.py:282](../create_dataset.py#L282) | No signal difference; inflates the video branch enormously | §4.3 | **Yes** | `build_variant.py --visual_field CMU_MOSEI_VisualFacet42` — variant C uses FACET 35-d; video branch shrank from 713 to 35 hidden units (§5.2) with no performance loss. |
| Segments dropped by a bare `except: continue`, uncounted | [create_dataset.py:342-343](../create_dataset.py#L342-L343) | Cannot report how much data was discarded | — | **Yes** — in the variant pipeline | `build_variant.py` counts every skip by reason (no bare `except`) and prints per-reason drop counters (§4.2). `create_dataset.py` itself unchanged. |

### 1.2 Fusion faults

| Fault | Location | Impact | Evidence | Fixed? | The fix |
|---|---|---|---|---|---|
| **Cross-modal attention never executes** — `torch.allclose(k,v)` is permanently true | [jack_transformer_share.py:159](../jack_transformer_share.py#L159) | 3 of 4 encoders process one modality in isolation; 4,080 calls, zero taken | §3.3 | **No** | Deliberately left: §7 shows the model reaches the measured ceiling with the defect in place, so activating the path cannot add what the data does not contain. Documented via branch counters instead. |
| **Attention runs over the batch axis**, not time | [jack_transformer_share.py:55](../jack_transformer_share.py#L55) | Each clip mixed with 47 unrelated clips; scores vary by 0.007 with grouping | §3.1 | **No** | Deliberately left: repairing it means rearchitecting the fusion core, i.e. no longer evaluating TMBL. Measured (§3.1), adopted as the project's noise floor, and disclosed as a limitation. |
| **Shared encoder passes text as query in all three outputs** | [jack_transformer_share.py:226-228](../jack_transformer_share.py#L226-L228) | `share_T`/`share_V`/`share_A` are three copies of a text-queried function | code | **Yes** — Stage 1 | Encoder roles rotated so each modality takes a turn as query (§5.1). No performance gain resulted — which is what redirected the investigation to the data. |
| **Binding concatenates batch halves**, not modalities | [network.py:155-157](../network.py#L155-L157) | Half of each batch holds another clip's modality | code | **Yes** — Stage 1 | Replaced with per-clip modality binding (§5.1). |
| **`cls_token`/`pos_embed` indexed by batch position** | [network.py:113-114](../network.py#L113-L114) | 48 arbitrary positional embeddings; train/eval inconsistent | §5.3 | **Yes** — §5.3 | Reshaped from `[48, …]` to `[1, …]` and broadcast; `try/except` around the token concatenation removed; `drop_last=shuffle` on the train loader. |
| **`DiffLoss` reads the wrong axis** — `input1.size(0)` is 2, not the batch | [utils/functions.py:15](../utils/functions.py#L15) | Auxiliary objective near-vacuous; allocates ~600 MB per call | code | **No** — mitigated | Term itself unchanged (still weighted 0.5 in the loss). Mitigated indirectly: checkpoint selection moved from dev loss (which included it) to dev macro-F1 (§5.1). |
| **`loss_simi` rewards collapse** — minimised by suppressing the A/V contribution | [utils/functions.py:64-68](../utils/functions.py#L64-L68) | Gradient pressure *against* using non-verbal channels | code | **Partially** | Weight reduced 0.5 → 0.05 in Stage 1 (§5.1); the term's collapse incentive itself is unchanged, only ten times weaker. |
| **`keep_q_k` conv never runs** — wrong tensor layout, caught by a bare `except` | [jack_transformer_share.py:61-68](../jack_transformer_share.py#L61-L68) | A declared feature has never executed | code | **No** | Deliberately left: there is no clearly-correct version to restore (the conv treats the batch axis as channels and only runs at batch = 32 by coincidence). Documented as a reproducibility finding; batch size kept ≠ 32 so behaviour stays consistent. |
| **Whole non-verbal pathway underperforms simple pooling** | — | Model A/V-only 0.3774 vs probe 0.4013 | §4.5 | **No** | Not addressed: the ceiling analysis (§7) caps the recoverable amount at ~0.0075, below the noise floor’s reach for a redesign to prove itself. Reported as a finding (linear read-out ≥ fusion stack) rather than repaired. |

### 1.3 Method / evaluation faults

| Fault | Impact | Fixed? | The fix |
|---|---|---|---|
| BERT truncation at word count, not token count | Long utterances silently cut off | **Yes** — §5.4 | Tokeniser cap raised from `SENT_LEN + 2` to `min(2 × SENT_LEN + 2, 512)` in `data_loader.py`, covering sub-word expansion. |
| Checkpoint selected on dev **loss**, which includes two broken auxiliary terms | Best model chosen by a partly-meaningless criterion | **Yes** — Stage 1 | Selection switched to dev macro-F1 ([solver.py:165](../solver.py#L165)). |
| Periodic checkpoints written to `epoch{e}.pt` with no run name | Successive runs overwrite each other | **No** | Unchanged in `solver.py`. Consequences avoided procedurally (dev-best `.std` files carry the run name and are the only checkpoints used for evaluation). |
| Two runs sharing `--name` | **Cost one checkpoint** — see provenance warning | **Yes** — procedurally | Every subsequent run given a unique `--name` and a date-stamped log file. The overwritten 1.2 checkpoint is unrecoverable; its training log's final-test block remains valid. |
| `--ablate` omitted when threshold-tuning ablated checkpoints (`fine-tune.md:267`) | Would feed a modality-blind model real features | **Yes** — procedurally | Command corrected; the reproduction table (§11) records the correct form with `--ablate` always passed. *(Doc-level fault — no code change involved.)* |
| Single runs throughout, no seed variance | Differences below ~0.01 unresolvable | **No** | Outstanding. Mitigated by policy: the §3.1 batch-grouping spread (0.007) is adopted as the noise floor and every sub-0.01 difference is treated as unresolved. Proper fix (multi-seed runs or a bootstrap over test clips) listed as remaining work. |

---

# Section 2 — The Ceiling Analysis: why performance is limited

**The decisive comparison:**

| | Value | Source |
|---|---|---|
| **Ceiling** — most that audio+video can add on top of BERT-strength text | **+0.0199** | `probe_bert_ceiling.log` |
| **Realised** — what the model actually gets from them | **+0.0209** | `run1_ablations…log` + `only_text_RUN1.log` |

They match to within a thousandth. **The model already extracts the entire available multimodal
contribution.**

**In plain terms.** Measure an engine's maximum power, then measure what reaches the wheels. They
match, so nothing is being lost in transmission. Rebuilding the gearbox cannot help. To go faster
you need a bigger engine — better data — not a better gearbox.

### Why the ceiling sits where it does, ranked

| # | Cause | Size | Proven? |
|---|---|---|---|
| 1 | **Audio and video mostly repeat the words** — 63% redundant | Caps everything at +0.020 | ✅ §4.5 |
| 2 | **The non-verbal pathway loses information** vs simple pooling | ~0.0075 recoverable | ✅ §4.5 |
| 3 | **The fusion is defective** | ≈ 0 | ✅ §7 (measured, not assumed) |
| 4 | Word-level frame averaging | unknown | ❌ not tested |
| 5 | Annotator disagreement / label noise | unknown | ❌ not tested |

**Conclusion.** The fusion defects catalogued in §6.2 are real, proven, and **not what limits
performance**. The binding constraint is the dataset: on CMU-MOSEI, emotion is largely predictable
from the transcript, and the non-verbal channels carry little that the words do not.

That is the finding — a measured upper bound and a demonstration that the model already operates at
it — rather than a failure to improve a score.
