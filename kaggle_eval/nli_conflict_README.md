# Fine-tuning Laya for memory-conflict detection (MultiNLI -> noul)

Trains a "does the new information conflict with the known memory?" head on top of
`convaiinnovations/laya-multilingual`, using MultiNLI entailment/contradiction pairs
reformatted as memory pairs. Same RLCD recipe as the official typed-decisions notebook.

**Data**: 6,000 train / 1,000 validation pairs (entailment -> "no conflict", contradiction ->
"conflict", neutral dropped). State is `{"known": <premise>, "new": <hypothesis>}` JSON.

**Cells:**

1. env check + deps
2. build training items (upload `data_local/nli_conflict_train.jsonl` + `nli_conflict_val.jsonl`
   as Kaggle datasets first, or download from wherever you host them)
3. write the DDP training script (adapted from the official notebook; reads JSONL, 3 epochs)
4. launch 2xT4 training (~40 min)
5. evaluate on the validation pair set: conflict-detection accuracy + ECE
6. spot-check with realistic memory-plugin cases (zh + en)
7. save results.json to output

Run with Accelerator = GPU T4 x2, Internet on.
