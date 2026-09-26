---
license: apache-2.0
library_name: transformers
tags:
- laya
- system-one
- typed-decisions
- rlcd
- multilingual
- fine-tuned
base_model: convaiinnovations/laya-multilingual
datasets:
- LocalLLaMA/typed-decisions
metrics:
- accuracy
model-index:
- name: laya-typed-decisions-multilingual
  results:
  - task:
      type: text-classification
      name: System One Decision Benchmark
    dataset:
      type: LocalLLaMA/typed-decisions
      name: Typed Decisions (all/test)
    metrics:
    - type: accuracy
      value: 0.7875
    - type: ece
      value: 0.159
---

# laya-typed-decisions-multilingual

A community fine-tune that combines what [#320](https://github.com/NandhaKishorM/laya/issues/320) asked
for: the **mmBERT-base multilingual encoder** (322M) with decision heads fine-tuned on the
typed-decisions workflows. Trained by [modusensus](https://github.com/modusensus) using the official
public recipe — the Kaggle 2xT4 notebook, unmodified.

## Results

Official test split, 400 cases / 2,000 decisions, argmax vs gold label:

| model | choice | noul | score | total |
|---|---|---|---|---|
| `laya-multilingual` (zero-shot) | 0.295 | 0.497 | 0.286 | **0.352** |
| **this checkpoint** | 0.752 | 0.862 | 0.759 | **0.7875** |
| `laya-typed-decisions` (published, English encoder) | — | — | — | 0.766 |

ECE as shipped (max-prob confidence, 10 bins): **0.159**. Temperatures were fitted on the notebook's
held-out-from-training calibration slice (choice 1.11, score 1.05, noul 1.20); that slice is still
in-distribution for the benchmark, so treat the calibration number as optimistic.

Full write-up and early non-English (zh/es/ja) spot-check observations:
[laya Discussions #482](https://github.com/NandhaKishorM/laya/discussions/482). Short version: choice
routing survives fine-tuning in all three languages tested; the score head's non-English calibration
moved (mostly directional improvements, one regression in ja). Per-language held-out evaluation is
still pending — no multilingual accuracy claims are made here.

## Training

- Base: `convaiinnovations/laya-multilingual` (mmBERT-base, 322M)
- Data: `LocalLLaMA/typed-decisions` train split (~30k items), full-encoder RLCD
  (proper-scoring-rule reward + noisy-logit policy gradient + soft CE), 4 epochs
- Hardware: Kaggle 2x T4 (DDP), ~1.5 h wall clock
- Post-training: per-type temperature calibration, `temperature_by_options` removed from the config

## Usage

```python
import laya  # pip install laya

agent = laya.load("Modusnsus/laya-typed-decisions-multilingual")
result = agent.predict(state, questions)  # same API as every Laya checkpoint
print(result["answers"])
```

## Reproduce

Run [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
with `convaiinnovations/laya-multilingual` as the base checkpoint, then evaluate on the
`all/test` split of `LocalLLaMA/typed-decisions`.

## License

Apache 2.0, matching the base checkpoint. Original model by
[Convai Innovations](https://huggingface.co/convaiinnovations).
