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
- research-archive
- not-delivered
datasets:
- nyu-mll/multi_nli
metrics:
- accuracy
- ece
---

# laya-nli-conflict-v5-ce — research archive (round 5, arm B), NOT delivered

> ⚠️ **Research archive — NOT a delivered model.** This checkpoint failed its round's acceptance gates and was never shipped. The current production head is [`slow-stack/laya-nli-memory-conflict`](https://huggingface.co/slow-stack/laya-nli-memory-conflict) (v4). Uploaded 2026-10-02 for provenance/backup while round 11 (multi-run verdict protocol) waits for Kaggle GPU quota.

Round 5 (2026-09-28) arm B: **pure CE on graded soft targets**, same corpus as arm A (see [`laya-nli-conflict-v5`](https://huggingface.co/slow-stack/laya-nli-conflict-v5)).

## Headline results (frozen 1000-pair main val unless noted)
- main val **0.902** ✓ (over v4's 0.901), new-10 **10/10** ✓
- failed: val_soft 7 errors (✗), polarity +2 (✗)
- **Key finding**: graded soft targets widened the confidence band 0.28 → **16.2pp in both arms** (the compression-band main attack succeeded; arm B's main-val bins recovered monotonicity). Combined with arm A's collapse, this established pure CE as the round-6+ main arm.

## Artifacts
| file | value |
|---|---|
| model.safetensors | SHA256 `997bd810…db8717` (full hash in [`archive_sha256_manifest.txt`](https://github.com/modusensus/laya/blob/main/kaggle_eval/archive_sha256_manifest.txt)) |
| rl_agent_config.json | τ(noul) = 1.0647; encoder `jhu-clsp/mmBERT-base`; bf16 |
| metrics.json | val_accuracy 0.902, val_ece 0.0261, n_val 1000, no_rl true |
| val_probs.json | frozen-val probability dump (calibration analyses) |

## Provenance
- Training: Kaggle GPU kernel `daphnelaurent/laya-nli-conflict-ce` v1, dataset `daphnelaurent/nli-conflict-pairs` v10
- Round record & full gate table: [`kaggle_eval/HANDOFF_NLI_V5.md`](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V5.md)
- Base: fine-tuned from [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) lineage; bf16 weights; `checkpoint_latest/` intentionally not uploaded
