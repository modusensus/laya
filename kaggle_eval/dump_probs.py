# -*- coding: utf-8 -*-
"""Dump per-row predictions on the frozen val_soft 300 (V5 §6 deliverable).

Output shape identical to data_local/val_soft_v4.json: a list of
{pred, p_true, label, meta} in input order. Works for any conflict-head
checkpoint (local dir). The val_soft rows carry their own questions, so this
script is labelset-agnostic.

Usage: python dump_probs.py <ckpt> <out_json>
"""
import json
import sys

import laya

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v5'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\val_soft_probs_v5.json'
VAL_SOFT = r'D:\laya\data_local\nli_conflict_val_soft.jsonl'


def main():
    rows = [json.loads(l) for l in open(VAL_SOFT, encoding='utf-8') if l.strip()]
    assert len(rows) == 300, len(rows)
    agent = laya.load(CKPT, device='cpu')
    out = []
    for i, r in enumerate(rows):
        state = json.loads(r['state']) if isinstance(r['state'], str) else r['state']
        p = float(agent.predict(state, r['questions'])['answers']['conflict']['noul'])
        meta = r.get('meta', {})
        out.append({'pred': 'true' if p >= 0.5 else 'false', 'p_true': round(p, 4),
                    'label': r['gold']['conflict']['label'],
                    'meta': {'cat': meta.get('cat', 'val_soft'), 'kind': meta.get('kind', 'val_soft'),
                             'lang': meta.get('lang', 'zh'), 'labelset': meta.get('labelset', 1)}})
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/300', flush=True)
    errs = sum(1 for o in out if o['pred'] != o['label'])
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'err {errs}/300 -> saved: {OUT}')
    sys.exit(0 if errs <= 3 else 1)


if __name__ == '__main__':
    main()
