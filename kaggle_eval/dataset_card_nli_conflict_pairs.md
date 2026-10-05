---
license: cc0-1.0
task_categories:
- text-classification
language:
- zh
- en
size_categories:
- 10K<n<100K
source_datasets:
- nyu-mll/multi_nli
tags:
- laya
- system-one
- typed-decisions
- nli
- memory-conflict
- synthetic
pretty_name: NLI Conflict Pairs for Laya
---

# NLI Conflict Pairs for Laya

Training data for the laya 0.3B **memory-conflict decision head**: MultiNLI pairs recast as
user-memory pairs `{"known": premise, "new": hypothesis}`, labeled for the `noul` question
"新信息(new)与已有记忆(known)是否冲突？" — 冲突(`true`)=矛盾需更新旧记忆, 兼容(`false`)=一致或无关.

Mirror of the Kaggle dataset [`daphnelaurent/nli-conflict-pairs`](https://www.kaggle.com/datasets/daphnelaurent/nli-conflict-pairs)
(**CC0-1.0**, mirror taken from Kaggle version v15, 2026-09-30). **Fully synthetic** —
MultiNLI-derived pairs + template-generated soft-conflict blocks; contains no personal or real-user data.

## Files

| file | rows | content |
|---|---|---|
| `nli_conflict_train.jsonl` | 15,950 | v17 train = v12 arm (Kaggle dataset v17): v11's 15,914 rows carried VERBATIM + 36 negation-known-conflict lever rows (6 domains × 6, zh 24 / en 12, gold=true; the two chronic acceptance texts stay out of training) per [`HANDOFF_NLI_V12.md`](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md) §2.1 |
| `nli_conflict_val.jsonl` | 1,000 | frozen main val (byte-frozen across rounds) |
| `nli_conflict_val_soft.jsonl` | 300 | frozen soft val (graded soft targets) |
| `kaggle_README.md` | — | the Kaggle dataset description, incl. the full v1→v15 version table |

## Format

Each line of the `.jsonl` files:

```json
{"state": "{\"known\": \"用户的发型是短发\", \"new\": \"用户觉得短发比较好打理\"}",
 "questions": {"conflict": {"type": "noul", "instructions": "…", "labels": {"false": "不冲突", "true": "需更新旧记忆"}}},
 "gold": {"conflict": {"label": "false", "probabilities": {"false": 0.9204, "true": 0.0796}}},
 "meta": {"cat": "hair", "kind": "compat", "lang": "zh", "labelset": 3, "name_a": "不冲突", "name_b": "需更新旧记忆"}}
```

MNLI label mapping (v2 onward): entailment → `false` (compatible), neutral → `false`
(unrelated supplement — the class v1 dropped), contradiction → `true` (conflict).
Contradiction:non-contradiction ≈ 1:2. Soft gold is graded (`0.95/0.05` style), not hard labels.

## Provenance

- Models trained on this data: [`Modusnsus/laya-nli-memory-conflict`](https://huggingface.co/Modusnsus/laya-nli-memory-conflict) (delivered head) and the rounds 5–10 research-archive repos (see the archive index on the v4 model card).
- Data generators, acceptance/diagnostic sets and the full gate protocol: [`modusensus/laya` → `kaggle_eval/`](https://github.com/modusensus/laya/tree/main/kaggle_eval)
- The 35 acceptance cases and 18 diagnostic contrast pairs are **not** in this dataset (they live in the eval code in `kaggle_eval/` and share zero verbatim overlap with the training corpus by enforced assertion).
- Round 11 (Kaggle version v16) added: B2-isomorphic upsampling (lever 1), unrelated-supplement / temporary-preference control rows (lever 1b) and a 9-case isomorphic family diagnostic — per [`HANDOFF_NLI_V11.md`](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V11.md).
- Round 12 (Kaggle version v17) adds: a single negation-known-conflict lever (36 rows) plus an 8-case negation-family diagnostic (eval-side, never in train), and preregistered median aggregation for the thin-edge new10/negation5 axes — per [`HANDOFF_NLI_V12.md`](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md).

## License

CC0-1.0, same as the Kaggle source. Content is template-rewritten MultiNLI derivatives plus synthetic blocks.
