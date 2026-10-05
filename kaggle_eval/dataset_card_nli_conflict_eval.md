# Laya NLI memory-conflict eval — frozen multi-run protocol evaluation set

The frozen **evaluation face** of the Laya 0.3B noul memory-conflict head's
preregistered multi-run protocol ([HANDOFF_NLI_V12.md](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md)
§1.2; round numbering follows that document). Every training corpus since v6
asserts **zero field-level overlap** with these cases (`leak_audit` 0/35 +
gen-time asserts), and the acceptance-case texts themselves are never trained
on — this set measures the protocol, it is not training data.

**66 cases across 7 instruments** (`laya_nli_conflict_eval.jsonl`, one JSON object per line):

| instrument | cases | protocol gate |
|---|---|---|
| `acceptance_old20` | 20 | #2 old-20: median = 20/20 AND no case missed in ≥2 of 3 runs |
| `acceptance_new10` | 10 | #3 new-10: median = 10/10 AND no case missed in ≥2 runs |
| `acceptance_negation5` | 5 | #4 negation-5: median = 5/5 AND no case missed in ≥2 runs |
| `bias_diag_control` | 10 | #11 bias diag: minimal pairs share `known`, differ only in conflict; ≥2/3 runs ≥13/14 |
| `bias_diag_swap` | 4 | #11 bias diag: literal known/new swaps keep their label |
| `family_compat` | 9 | #12 family rate: gold=false isomorphic family; median mean p_conflict <0.5, every run ≤1 case at p≥0.9 |
| `family_negation` | 8 | #13 negation family rate: gold=true family (negated known + violating new); median mean p_conflict ≥0.5, every run ≤1 case at p<0.5 |

Row schema: `instrument`, `gate`, `note` (case name), `lang` (zh/en),
`known`, `new` (the two memory sentences), `gold` ("true" = conflict),
`flags` (`is_b2`: the historic B2 probe inside `family_compat`;
`chronic_domain`: negation-family case in a chronic acceptance domain;
`chronic_acceptance`: the two cases that repeat-miss across rounds and drove
rounds 11–12 levers — 「否定-养宠物反转」, 「新-订阅服务(否定句)」).

## Scoring

Load a checkpoint and ask the **noul** question with `question_noul.json`
(rendering must match inference — label words matter, see the option-name
polarity note in the handoffs):

```python
import laya, json
agent = laya.load(<ckpt>, device='cpu')
q = json.load(open('question_noul.json'))
a = agent.predict({'known': row['known'], 'new': row['new']}, {'conflict': q})
p_conflict = float(a['answers']['conflict']['noul'])   # >= 0.5 -> "true"
```

Per-run artifacts and cross-run medians are then judged exactly as the `gate`
column states; reference implementations: `realtest_v2.py`/`realtest_v4.py`,
`noul_bias_diag.py`, `family_diag.py`, `negfam_diag.py`,
`protocol_verdict.py` in the [fork](https://github.com/modusensus/laya/tree/main/kaggle_eval).

## Provenance & license

- Case sources (exported programmatically, no hand transcription): `realtest_v2.py`,
  `realtest_v4.py`, `noul_bias_diag.py`, `family_diag.py`, `negfam_diag.py`.
- Companion datasets: training corpora mirror
  [Modusnsus/nli-conflict-pairs](https://huggingface.co/datasets/Modusnsus/nli-conflict-pairs);
  per-round corpus lineage [Modusnsus/nli-conflict-train-lineage](https://huggingface.co/datasets/Modusnsus/nli-conflict-train-lineage).
- All content is synthetic; no personal or real-user data. License: CC0-1.0.
