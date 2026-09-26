"""Generate v2 memory-conflict JSONL from MultiNLI.

v1 dropped the neutral class entirely, so the fine-tuned head never saw "unrelated"
and answered "conflict" on everything it was asked about. v2 keeps neutral:

  entailment    -> gold false (compatible, hypothesis follows from the memory)
  neutral       -> gold false (unrelated, hypothesis neither confirmed nor contradicted)
  contradiction -> gold true  (conflict, memory must be updated)

Sampling ratio contradiction : non-contradiction ~= 1:2 (real memory streams are
mostly non-conflicting). Output structure is byte-compatible with v1 except the
question text and the new `labels` key (upstream #163), per HANDOFF_NLI_V2.md.
"""
import json
import os
import random
from collections import Counter

from datasets import load_dataset

SEED = 20260927
TRAIN_PER_CLASS = 2000          # 2000 x {contradiction, entailment, neutral} = 6000
VAL_PER_CLASS = (334, 333, 333)  # 1000 total, same 1:2 conflict ratio
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_local')
BACKUP_DIR = os.path.join(OUT_DIR, 'v1_backup')

# v2 question: semantic label words instead of true/false (upstream #156 / #163).
QUESTION = {
    'conflict': {
        'type': 'noul',
        'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
        'labels': {'false': '兼容', 'true': '冲突'},
    }
}
GOLD_FALSE = {'label': 'false', 'probabilities': {'false': 0.95, 'true': 0.05}}
GOLD_TRUE = {'label': 'true', 'probabilities': {'false': 0.05, 'true': 0.95}}

MNLI_LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


def rows_for_split(ds, want, rng):
    """Sample `want` = {class_name: n} pairs, deduped, premise==hypothesis dropped."""
    buckets = {c: [] for c in want}
    seen = set()
    for row in ds:
        premise, hyp = row['premise'].strip(), row['hypothesis'].strip()
        if not premise or not hyp or premise == hyp:
            continue
        key = (premise, hyp)
        if key in seen:
            continue
        seen.add(key)
        c = MNLI_LABEL[row['label']]
        if c in want and len(buckets[c]) < want[c]:
            buckets[c].append({'known': premise, 'new': hyp})
        if all(len(buckets[c]) >= want[c] for c in want):
            break
    missing = {c: want[c] - len(buckets[c]) for c in want if len(buckets[c]) < want[c]}
    if missing:
        raise RuntimeError(f'split exhausted before filling quotas: missing {missing}')
    return buckets


def write_jsonl(path, buckets, rng):
    rows = []
    for cls, pairs in buckets.items():
        gold = GOLD_TRUE if cls == 'contradiction' else GOLD_FALSE
        for p in pairs:
            rows.append({
                'state': json.dumps(p, ensure_ascii=False),
                'questions': QUESTION,
                'gold': {'conflict': dict(gold)},
            })
    rng.shuffle(rows)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(rows)} rows | label mix {dict(Counter(r["gold"]["conflict"]["label"] for r in rows))}')


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for name in ('nli_conflict_train.jsonl', 'nli_conflict_val.jsonl'):
        src = os.path.join(OUT_DIR, name)
        dst = os.path.join(BACKUP_DIR, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.replace(src, dst)
            print(f'v1 -> {dst}')

    rng = random.Random(SEED)
    train = load_dataset('nyu-mll/glue', 'mnli', split='train')
    val = load_dataset('nyu-mll/glue', 'mnli', split='validation_matched')
    print(f'MNLI train={len(train)} val_matched={len(val)}')

    train_buckets = rows_for_split(train, {'contradiction': TRAIN_PER_CLASS,
                                           'entailment': TRAIN_PER_CLASS,
                                           'neutral': TRAIN_PER_CLASS}, rng)
    val_buckets = rows_for_split(val, {'contradiction': VAL_PER_CLASS[0],
                                       'entailment': VAL_PER_CLASS[1],
                                       'neutral': VAL_PER_CLASS[2]}, rng)
    write_jsonl(os.path.join(OUT_DIR, 'nli_conflict_train.jsonl'), train_buckets, rng)
    write_jsonl(os.path.join(OUT_DIR, 'nli_conflict_val.jsonl'), val_buckets, rng)


if __name__ == '__main__':
    main()
