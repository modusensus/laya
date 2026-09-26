# A multilingual + typed-decisions fine-tune: recipe reproduced, 0.7875 on the official test split

Follow-up to #320 (a multilingual typed-decisions checkpoint) and the questions @jcastdev asked there.
Short version: I fine-tuned `laya-multilingual` (mmBERT-base encoder) on `LocalLLaMA/typed-decisions`
using the public Kaggle notebook, and the result scores **0.7875 accuracy on the full official 400-case
test split (2,000 decisions)** — versus **0.352** for `laya-multilingual` zero-shot on the same split,
and the published `laya-typed-decisions` number of 0.766.

## Setup

- Base checkpoint: `convaiinnovations/laya-multilingual` (mmBERT-base, 322M)
- Training: the public notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`), unmodified
  recipe — RLCD (proper-scoring-rule reward + noisy-logit policy gradient + soft CE), 4 epochs over the
  full ~30k-item train split, 2x T4, ~1.5 h wall clock
- Post-training: per-type temperature calibration on the notebook's held-out calibration slice
  (choice 1.11, score 1.05, noul 1.20)

## Results (official test split, 400 cases / 2,000 decisions, argmax vs gold label)

| model | choice | noul | score | total |
|---|---|---|---|---|
| laya-multilingual (zero-shot) | 0.295 | 0.497 | 0.286 | **0.352** |
| mine (mmBERT + typed-decisions fine-tune) | 0.752 | 0.862 | 0.759 | **0.7875** |

ECE as shipped (max-prob confidence, 10 bins): **0.159**. Temperatures were fitted on the notebook's
held-out-from-training calibration slice, but that slice is still in-distribution for the benchmark, so
treat the calibration number as optimistic.

## Early non-English observations (small spot check, not a benchmark)

9 hand-written cases in Chinese / Spanish / Japanese (data-erasure routing choice, plus
credential-leak and business-hours risk scoring):

- Choice routing is intact in all three languages — `delete_data` picked everywhere, though with less
  confidence than the base model (0.53–0.80 vs 0.98–1.00).
- Score judgments shifted with the fine-tune: the credential-leak case moves from ~1.1 toward
  1.34–1.46 (directionally right, still short of "high"); the trivial business-hours case improves in
  zh/es (~0.55–0.63) but got *worse* in ja (1.25).
- No sign of catastrophic forgetting on routing, but the score head's non-English calibration clearly
  moved. A proper per-language held-out evaluation is the obvious next step — per the maintainer's
  advice in #320, each language needs its own held-out set before any claim.

## What this means for #320

The combination the issue asks for (mmBERT encoder + typed-decisions heads) trains fine with the
public recipe as-is — no architecture changes needed, and the English-language benchmark result lands
at/above the published English-encoder checkpoint. To answer @jcastdev's question 3 from the issue:
freezing the encoder is *not* required for this path; the full-encoder RLCD recipe transfers to the
mmBERT backbone directly.

## Reproduce

```bash
# 1. run notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb on Kaggle T4 x2 (~1.5 h)
# 2. evaluate the output checkpoint:
python -c "
import laya
agent = laya.load('<output_dir>')
print(agent.predict(state, questions)['answers'])
"
```

Happy to share the evaluation harness and the per-decision logs. The checkpoint itself I can publish
to the HF Hub if there is interest — say the word and I'll put it up under my own account with a
model card mirroring this post.
