---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
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

A 322M "System One" decision head: hand it an agent trace (the `state`) and typed questions,
and it returns calibrated decisions in millisecond-class time on CPU — what to do with the
trace, whether a human should review it, how the task ended. Built by [modusensus](https://github.com/modusensus)
as a community fine-tune combining what [#320](https://github.com/NandhaKishorM/laya/issues/320)
asked for: the **mmBERT-base multilingual encoder** with decision heads trained on the
typed-decisions workflows, using the official public recipe — the Kaggle 2xT4 notebook, unmodified.

## The three question types

| type | answers | returns |
|---|---|---|
| `choice` | pick one option from `criteria` | argmax label + full probability vector |
| `noul` | yes/no — does the statement hold | `noul` = P(yes), calibrated |
| `score` | rate on an ordered `criteria` scale | expected score + per-level probabilities |

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

## Usage

Run verbatim (output below is from this exact snippet):

```python
import laya  # pip install laya

agent = laya.load("Modusnsus/laya-typed-decisions-multilingual")
state = {"task": "Rotate the expired TLS certificate on the staging load balancer.",
         "constraints": ["Do not exceed a $50 spend on cloud resources"],
         "trace_summary": {"steps": 11, "duration_s": 32.5, "tool_errors": 0,
                            "constraint_violations": 0, "irreversible_actions": 0}}
questions = {"action": {"type": "choice",
                        "instructions": "What should the observability system do with this trace?",
                        "criteria": {"continue": "Let the agent proceed without interruption.",
                                     "human_review": "Queue this trace for a human to review.",
                                     "observe": "Keep running, but flag the trace for later sampling.",
                                     "stop": "Halt the agent now."}},
             "needs_review": {"type": "noul",
                              "instructions": "This trace requires human review."}}
print(agent.predict(state, questions)["answers"])
```

A clean trace sits near the continue/observe boundary, and review is confidently declined:

```json
{"action": {"type": "choice", "choice": "continue",
            "probabilities": {"continue": 0.4461, "human_review": 0.1126,
                              "observe": 0.4196, "stop": 0.0217},
            "confidence": 0.24, "answer_confidence": 0.4461},
 "needs_review": {"type": "noul", "noul": 0.1119, "confidence": 0.8881}}
```

`answer_confidence` is the calibrated per-answer confidence (one number you can gate on across
all three types); `confidence` keeps the per-type legacy meaning.

## Training

- Base: `convaiinnovations/laya-multilingual` (mmBERT-base, 322M)
- Data: `LocalLLaMA/typed-decisions` train split (~30k items), full-encoder RLCD
  (proper-scoring-rule reward + noisy-logit policy gradient + soft CE), 4 epochs
- Hardware: Kaggle 2x T4 (DDP), ~1.5 h wall clock
- Post-training: per-type temperature calibration, `temperature_by_options` removed from the config

If this is your first small-model fine-tune: the point of this repo is that the whole thing —
30k rows, free 2×T4, unmodified public notebook — reproduces end to end, and the result beats
the published English-encoder checkpoint on the official split.

## Sibling head

[`Modusnsus/laya-nli-memory-conflict`](https://huggingface.co/Modusnsus/laya-nli-memory-conflict) —
a memory-conflict noul head (v4) trained to detect when new information contradicts stored memory;
the two models share the Laya base and API but answer different questions.

## Reproduce

Run [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
with `convaiinnovations/laya-multilingual` as the base checkpoint, then evaluate on the
`all/test` split of `LocalLLaMA/typed-decisions`.

`model.safetensors` SHA256: `8bb8cdd4875313baae2cdea52a4f682a2fbc86fd174c9973165b6b04a1aa2718`

## License

Apache 2.0, matching the base checkpoint. Original model by
[Convai Innovations](https://huggingface.co/convaiinnovations).
