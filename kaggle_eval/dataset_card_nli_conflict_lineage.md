# Laya NLI conflict training-corpus lineage (v3-v12)

One training corpus per round of the Laya 0.3B noul memory-conflict head
([project handoffs](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md)),
with row counts and SHA256 computed at publish time. This completes the
reproducibility chain for the published checkpoints: Kaggle dataset versions
are rolling (only the latest is pullable), so the per-round corpora live here.

Corpora are deterministic outputs of the `gen_soft_conflicts_v*.py` scripts
(fixed seeds, hard leak asserts: zero field-level overlap with the 35
acceptance cases from v6 onward, val/val_soft byte-frozen). The frozen
evaluation sets live in
[slow-stack/laya-nli-conflict-eval](https://huggingface.co/datasets/slow-stack/laya-nli-conflict-eval);
the current round's corpus is also mirrored (rolling) at
[slow-stack/nli-conflict-pairs](https://huggingface.co/datasets/slow-stack/nli-conflict-pairs).

| file | round | rows | sha256 | trains | lineage |
|---|---|---|---|---|---|
| `nli_conflict_train_v3.jsonl` | v3 (round 3) | 9,000 | `d5e944eb6ff62f60664ffe97dc07dca2eb4ac2ac2d4aba6a1b2a5e9d42dee194` | — (v3 ckpt not published; MNLI-extraction source for later gens) |  |
| `nli_conflict_train_v4.jsonl` | v4 (round 4) | 12,000 | `cd68adb26c0801ab5d56d3f8c157db9a590db214aa0fe3e1e756c0aa3793d886` | [slow-stack/laya-nli-memory-conflict](https://huggingface.co/slow-stack/laya-nli-memory-conflict) (delivered) | MNLI re-rotation + v4 additions |
| `nli_conflict_train_v5.jsonl` | v5 (round 5) | 13,400 | `71d3b507f1eb9bbfe3828baa80d9b2a0fcd13d4c95766e8f3762bb9f25007577` | [laya-nli-conflict-v5](https://huggingface.co/slow-stack/laya-nli-conflict-v5), [v5-ce](https://huggingface.co/slow-stack/laya-nli-conflict-v5-ce) | v5 lever blocks; both arms trained on this corpus |
| `nli_conflict_train_v6.jsonl` | v6 (round 6) | 14,500 | `a6f5d21414bd398ba9012920884553a490e20d7c015928fe0580dae76287fb9f` | [laya-nli-conflict-v6](https://huggingface.co/slow-stack/laya-nli-conflict-v6) | data-hygiene reset: 35 acceptance sentences cleared (leak 0/35 from here) |
| `nli_conflict_train_v7.jsonl` | v7 (round 7) | 15,300 | `c0f30b905898a3284572e27d8c2222208362ff52f51875778cff5e18fabc2536` | [laya-nli-conflict-v7](https://huggingface.co/slow-stack/laya-nli-conflict-v7) | + bare-intent HC, alt-praise, metric blocks |
| `nli_conflict_train_v8.jsonl` | v8 (round 8) | 15,420 | `8538eb8436cdda4b8aee7989da257119f209ef74c03c086c41656752b85efb89` | [laya-nli-conflict-v8](https://huggingface.co/slow-stack/laya-nli-conflict-v8) | +120 same-subject attribute-consistent rows (B2 dose-response) |
| `nli_conflict_train_v9.jsonl` | v9 (round 9) | 15,820 | `795e242950fb445cd73e66a9347158b2833c2ba86f50a865c3e865d52782d757` | [laya-nli-conflict-v9](https://huggingface.co/slow-stack/laya-nli-conflict-v9), [v10-s2](https://huggingface.co/slow-stack/laya-nli-conflict-v10-s2) (S-arm reuses v9 verbatim) | +400 attr-add/attr-contra/waver/change rows |
| `nli_conflict_train_v10.jsonl` | v10 (round 10) | 15,780 | `77c7154ac1c9928e2746e1320b8ce787cb5ccd2beec8e1f9f40c1be3daf04d5a` | [v10-l1](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l1), [v10-l2](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l2) | = v9 minus exactly the 40 attr_contradiction rows (carried rows byte-identical) |
| `nli_conflict_train_v11.jsonl` | v11 (round 11, r1-r3) | 15,914 | `b107d63c491553bc7be0329328b1571dfae8e7aaba9c5d152254bca63b7a866d` | — (verdict FAIL, no delivery; Kaggle dataset v16) | = v9 carried verbatim (block B stays) + 94 lever rows |
| `nli_conflict_train_v12.jsonl` | v12 (round 12, r1-r3) | 15,950 | `c23870d8b78961a169f416b1db6e537036171bc30bb01a146f7837da6dffeb0b` | — (runs in flight; Kaggle dataset v17) | = v11 carried verbatim + 36 negation-known-conflict lever rows |

## Notes

- `nli_conflict_train.jsonl` without a version suffix is NOT included: the
  per-round files above supersede it; v1/v2-era corpora and checkpoints are
  pre-hygiene (sentence-reuse era) and deliberately not published.
- Carried-row discipline: from v10 on, each round's corpus carries the
  previous round's rows VERBATIM (byte-identical lines) with only preregistered
  lever rows added/dropped, asserted at generation time.
- All content is synthetic (MultiNLI-derived + template-generated); no
  personal or real-user data. License: CC0-1.0.
