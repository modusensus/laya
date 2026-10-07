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
metrics:
- accuracy
- ece
model-index:
- name: laya-nli-memory-conflict
  results:
  - task:
      type: text-classification
      name: Memory-conflict decision (noul) on frozen held-out pairs
    dataset:
      type: other
      name: MultiNLI-derived memory pairs (frozen 1000-pair val)
    metrics:
    - type: accuracy
      value: 0.901
    - type: ece
      value: 0.0192
---

# laya-nli-memory-conflict

A Laya decision head that answers one question about a memory pair:

> **新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关**

`noul` question, two custom labels `{"false": "兼容", "true": "冲突"}` — i.e. "should the
stored memory be superseded by what the user just said, or not?". Built for memory plugins
that must decide whether to update a user's stored facts, preferences and constraints.

Fine-tuned from [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
with the official public recipe (the Kaggle 2xT4 notebook: RLCD loss + post-hoc per-type
temperature fit), by [modusensus](https://github.com/modusensus). Training tooling, the
data generators and every evaluation script are in
[the `kaggle_eval/` directory of this fork](https://github.com/modusensus/laya/tree/main/kaggle_eval),
together with two handoff documents that record the full gate tables.

## Results

Four candidates over the same frozen eval sets. **v4 is the delivered head**
(`model.safetensors` in this repo, SHA256 `e9e1c4a5…d4b13758`).

| metric | v2 | v3 | **v4 (this repo)** | gate |
|---|---|---|---|---|
| main val (frozen 1000, MultiNLI-derived) | 0.902 | 0.902 | **0.901** | ≥0.90 |
| main val ECE | 0.021 | 0.0146 | **0.0192** | ≤0.02 |
| bias diagnostic (14 control + input-swap pairs) | 12/14 | 13/14 | **13/14 PASS** | PASS, no pinned output |
| real case, frozen 20 hand-written pairs | 16/20 | 19/20 | **20/20** | ≥19/20 |
| real case, 10 template-free new pairs | — | — | **10/10** | ≥8/10 |
| soft-conflict set (300 pairs, 4 never-trained classes) | 206/300 | 295/300 | **297/300** | ≥295/300 |
| ├ held-out classes | 62/100 | 95/100 | **97/100** | ≥95/100 |
| label-surface swap (labels exchanged at inference) | — | — | **0 cases flipped** | ≤1 |
| negation side-set | 2/5 | 4/5 | **4/5** | report only |

## Usage

```python
import laya

agent = laya.load('slow-stack/laya-nli-memory-conflict', device='cpu')

QUESTION = {
    'type': 'noul',
    'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
    'labels': {'false': '兼容', 'true': '冲突'},
}

state = {'known': '用户对花生过敏', 'new': '用户想试试花生酱饼干'}
p_conflict = float(agent.predict(state, {'conflict': QUESTION})['answers']['conflict']['noul'])
print(p_conflict)   # ~0.92 -> conflict: the intent violates a stored hard constraint
```

## Evaluation protocol (why these numbers are comparable)

- **Frozen sets, append-only.** The 1000-pair val split, the 20 hand-written Chinese real
  cases and the 300-pair soft-conflict set were never edited between versions; new cases are
  added in separate sets (the 10 template-free pairs in v4).
- **Bias diagnostic before any accuracy claim.** 7 positive/negative control pairs × 2 input
  directions. A pinned output distribution invalidates every accuracy number measured after it
  (the v1 head answered "conflict" on everything at conf 0.91 — upstream
  [#156](https://github.com/NandhaKishorM/laya/issues/156)).
- **Label-surface rotation + label-swap gate (v4).** Training rows rotate 4 label/instruction
  phrasings (and the boolean key names), and acceptance re-runs the eval with the two label
  strings exchanged: **0 of 20 real cases and 0 of 14 diagnostic probes flipped**, so the head
  reads the pair's meaning rather than the label word.
- Training data (12k rows): 6,000 MultiNLI entailment/neutral/contradiction pairs mapped to
  false/false/true, plus 5,400 synthetic bilingual memory pairs (soft conflicts, compatible
  details, "considering/temporary" intents, unrelated mentions) and a 600-row hard-constraint
  class with 1:1 controls. Conflict : non-conflict ≈ 1:2.

## Known limits (read before integrating)

- **Negation (#377, upstream, unfixed):** 4/5 on the negation side-set — "do NOT cancel" style
  input is still weak. Route irreversible actions to a human check.
- **Confidence ceiling ≈ 0.92.** `rl_agent_config.json` carries the fitted inference
  temperature (`temperature[2] = 1.2176` for `noul`, a softening >1). Consequence: a downstream
  auto-write threshold above ~0.92 never fires. Sharpening is a config-level change, not a
  retrain — but see the next point before you pick one.
- **No single temperature satisfies all eval sets.** Measured trade-off (`temperature_sweep.py`):
  main val is best at τ≳1.2, the real-case and soft-conflict sets at τ≲0.8. ECE at perfect
  accuracy degenerates to `1 − confidence`, which is not a quality signal. Pick τ from the
  distribution your application actually sees, on a calibration slice disjoint from any set you
  report.
- **Hard-constraint class is synthetic** (300 positives + 300 controls) and covers
  allergy/safety-style intent violations only; it exists because v3 missed "wants to try the
  food he is allergic to" and it should be extended with real cases before trusting it broadly.
- Assets on Hugging Face are pinned by name, not by commit; the SHA256 above is the identity of
  the delivered weights.

## Round-4 diagnostics (option-name polarity & abstention layer)

- **Option-name axis (new).** The label-surface swap above exchanges the two label *values*;
  renaming the label *system* itself (neutral `A/B`, or per-case random strings — the axis
  [arXiv:2609.26758](https://arxiv.org/abs/2609.26758) shows typed decision heads latch onto)
  does move answers: on the 300-pair soft-conflict set accuracy drops 0.990 → 0.983 (net +2
  errors, above the ±1 gate), random renames shift `p_true` by ~0.05 on average, and the frozen
  real-20 loses 1 case to random renames (19/20, within gate). The head partly reads option-name
  polarity; name-system augmentation is on the v5 plan.
- **Confidence is a compression band.** On main val, `max_conf` lands between 0.918 and 0.928
  for essentially every pair (q10 0.9185, max 0.9279) — the tidy 0.901 acc / 0.918 mean-conf
  pairing is largely a constant-output artifact, so confidence has almost no ranking power.
  The `max_conf ≥ 0.92` subset is 87.1% of main val with a 6.66% error rate.
- **Conformal abstention layer (optional; red line unchanged).** A 90%-nominal split-conformal
  rule fitted on the soft-conflict set and transferred to main val captures **58 of 99 errors**
  at a **29.4% abstention rate** (write-error rate 6.66% → 5.81%). The plugin auto-write
  threshold stays ≤ 0.92 and the layer is additive: write = at red line AND not abstained.
  These are transfer numbers across a distribution shift (synthetic fit → MNLI-style eval),
  not finite-sample guarantees. Reproduce: `kaggle_eval/conformal_abstain.py`,
  precomputed gates in `kaggle_eval/HANDOFF_NLI_V4.1.md`.
- **No external same-benchmark comparison exists** for this task — the frozen memory-conflict
  sets are ours alone. The same-ruler comparison discipline is applied on the sibling
  typed-decisions card, where a shared public benchmark does exist.

## License

Apache-2.0, following upstream. Base model © [Convai Innovations](https://huggingface.co/convaiinnovations).
