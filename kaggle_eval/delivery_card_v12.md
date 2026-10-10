---
license: apache-2.0
library_name: transformers
base_model: convaiinnovations/laya-multilingual
pipeline_tag: text-classification
tags:
- laya
- system-one
- typed-decisions
- nli
- memory-conflict
- multilingual
- fine-tuned
datasets:
- nyu-mll/multi_nli
- slow-stack/nli-conflict-pairs
- slow-stack/nli-conflict-train-lineage
- slow-stack/laya-nli-conflict-eval
metrics:
- accuracy
- ece
model-index:
- name: laya-nli-conflict-v12
  results:
  - task:
      type: text-classification
      name: Memory-conflict decision (noul) on frozen held-out pairs
    dataset:
      type: other
      name: MultiNLI-derived memory pairs (frozen 1000-pair val)
    metrics:
    - type: accuracy
      value: 0.903
    - type: ece
      value: 0.0209
---

# laya-nli-conflict-v12

A Laya decision head that answers one question about a memory pair:

> **新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关**

`noul` question, two custom labels `{"false": "兼容", "true": "冲突"}` — "should the stored
memory be superseded by what the user just said, or not?". Built for memory plugins that
must decide whether to update a user's stored facts, preferences and constraints.

**This is the delivered head as of 2026-10-07, superseding
[`slow-stack/laya-nli-memory-conflict`](https://huggingface.co/slow-stack/laya-nli-memory-conflict) (v4).**
It is run **r2** of a preregistered n=3 equal-status training protocol
([HANDOFF_NLI_V12.md](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md))
whose 14 gate axes **all passed** — the first round to clear the full protocol. The weights
in this repo are r2's; r1/r3 are published unmodified as sibling archives
([v12-r1](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r1) /
[v12-r3](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r3)).

## Why r2 ships

Preregistration treats the three runs as equal for *gating* (no run selection), but exactly
one set of weights must serve traffic. r2 is chosen by an objective, pre-stated criterion:
**it is the only run with all three acceptance batteries at full marks** (old-20 20/20,
negation-5 5/5, new-10 10/10); every other reading sits at or near the three-run median.

## Three-run readings (the card must carry all of them — §1.3)

| axis (gate) | r1 | **r2 (ships)** | r3 | gate | verdict |
|---|---|---|---|---|---|
| main val acc (median ≥ 0.896) | 0.904 | **0.903** | 0.907 | median 0.904 | PASS |
| main val ECE (report) | 0.0207 | **0.0209** | 0.0190 | — | — |
| old-20 real cases (median = 20, no case missed in ≥2 runs) | 19 | **20** | 20 | 20 | PASS |
| negation-5 (median = 5, no repeat miss) | 5 | **5** | 4 | 5 | PASS |
| new-10 (median = 10, no repeat miss) | 10 | **10** | 10 | 10 | PASS |
| soft-conflict val_soft errors ≤3 | 2 | **1** | 1 | ≤3 | PASS |
| label-surface swap | PASS | **PASS** | PASS | all PASS | PASS |
| polarity real·diag ≤1 | PASS | **PASS** | PASS | all ≤1 | PASS |
| polarity val_soft ≤1 | PASS | **PASS** | PASS | all ≤1 | PASS |
| band ≥8pp | PASS | **PASS** | PASS | all | PASS |
| conformal adopted rule ≥50% @ ≤35% | 36/50@34.2% (s1) | **34/50@33.4% (s1)** | 33/47@34.0% (s1) | all | PASS |
| bias diag ≥13/14 in ≥2/3 runs | 13/14 | **13/14** | 13/14 | 3/3 runs | PASS |
| family rate, compat (median mean-p <0.5, ≤1 case p≥0.9/run) | 0.0806 | **0.0919** | 0.0814 | median 0.0814, 0 high-conf | PASS |
| negation family rate (median mean-p ≥0.5, ≤1 case p<0.5/run) | 0.7872 | **0.8022** | 0.7930 | median 0.7930, 1 low-conf/run | PASS |
| τ(noul) post-hoc fit (report) | 1.0887 | **1.0914** | 1.1080 | — | — |
| automation@5% (**auxiliary, non-gating**) | 0.814 | **0.781** | 0.781 | — | — |

Same-config variance band: main val 0.903–0.907; acceptance thin-edge flips are single-run
and never repeat across runs (r1 old-20 one miss; r3 negation one miss) — exactly the
bimodal thin-edge behaviour the preregistered median aggregation models.

## Protocol (multi-run, preregistered)

- **n=3 equal-status runs**, same kernel, same corpus (Kaggle dataset v17 = 15,950 rows =
  v11's 15,914 carried verbatim + 36 negation-known-conflict lever rows across 6 domains),
  same pure-CE recipe; no run selection/substitution/addition for gating.
- 14 gate axes per [HANDOFF_NLI_V12.md §1.2](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md);
  threshold numbers are frozen across v9–v12. Thin-edge axes (new-10 / negation-5) use
  **median = full marks AND no case missed in ≥2 runs** — preregistered before the runs
  (single-run flips at q≈0.2–0.25 made "all-three" a per-round lottery; the repeated-miss
  clause keeps detection power for真 weakness).
- **Family-rate instruments** (both preregistered): *compat family* (9 cases, gold=false) —
  median over runs of family-mean p_conflict < 0.5 AND every run has ≤1 case at p ≥ 0.9;
  *negation family* (8 cases, gold=true) — median over runs of family-mean p_conflict ≥ 0.5
  AND every run has ≤1 case at p < 0.5. Both families are eval-side only, never trained on.
- Hygiene: **the 35 acceptance cases have zero verbatim / field-level intersection with any
  training corpus since v6** (`leak_audit` 0/35; v12 gen asserts 0 overlap against 35 cases
  + 14 diag pairs + both families = 0/51). The chronic acceptance texts themselves are
  excluded from training.
- Verdict tool: `protocol_verdict.py --protocol v12` (claims pack
  `claims_v12.json` + `verify_claims_v12.py` verified exit 0 before this publish).

## Weights integrity

| repo | run | model.safetensors SHA256 |
|---|---|---|
| **this repo** | **r2** | `6648d892449c13f81107b0b5891aca260293d176567ccbe432b8a6ef4af3f72b` |
| [v12-r1](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r1) | r1 | `4269ec6bf8a64eee8aed9693395ac2f3d37119531f7d582869b05f048a27dc9b` |
| [v12-r3](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r3) | r3 | `b2bf87bee5a0d379a1f8a633458389c389ceee7e61fe532ff52203d37d79f9f8` |

All three weights are 643,835,524 bytes (same architecture).

## Usage

```python
import laya, json

agent = laya.load('slow-stack/laya-nli-conflict-v12')   # or a local checkpoint dir
q = {"type": "noul",
     "instructions": "新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关",
     "labels": {"false": "兼容", "true": "冲突"}}
a = agent.predict({'known': '用户对花生过敏', 'new': '用户说他可以吃花生酱了'}, {'conflict': q})
p_conflict = float(a['answers']['conflict']['noul'])   # >= 0.5 -> conflict (update the memory)
```

The per-bucket post-hoc temperature (τ noul ≈ 1.091) ships in `rl_agent_config.json` and is
picked up automatically by the laya runtime. Keep the label words exactly as above — option
polarity is part of the wire contract (see the handoffs' polarity notes).

## Provenance

- Training corpora (v3–v12, one per round, SHA256 manifest):
  [slow-stack/nli-conflict-train-lineage](https://huggingface.co/datasets/slow-stack/nli-conflict-train-lineage) ·
  current-round rolling mirror: [slow-stack/nli-conflict-pairs](https://huggingface.co/datasets/slow-stack/nli-conflict-pairs)
- Frozen evaluation set (66 cases, 7 instruments, scoring recipe):
  [slow-stack/laya-nli-conflict-eval](https://huggingface.co/datasets/slow-stack/laya-nli-conflict-eval)
- Full index: [the "Laya NLI memory-conflict heads — v12 & three-run protocol" collection](https://huggingface.co/collections/slow-stack/laya-nli-memory-conflict-heads-v12-and-three-run-protocol-6ac698ed9a92a16917804f5b)
- Base model: [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual);
  tooling & handoffs: [kaggle_eval/](https://github.com/modusensus/laya/tree/main/kaggle_eval)
- All content synthetic; no personal or real-user data.
