# -*- coding: utf-8 -*-
"""Holdout dual-read table (V7 §2.2 / §5.2): per-family error counts on the
frozen val_soft 300 for every round's dump, in ONE table.

Dual read: (a) frozen 4-holdout-cats criterion (email/desk_floor/degree/barber),
(b) 3-cat read excluding barber (barber overlaps the training domain since v5
-- V7 §2.2 note).  Row order of every dump maps directly onto
nli_conflict_val_soft.meta.json (dumps iterate the val_soft file in order;
v4 has no meta of its own, v6+ carry row_id which agrees with the order).

Usage: python holdout_dualread.py <dump1.json> [<dump2.json> ...]
       (tag inferred from filename: val_soft_v4.json -> v4)
"""
import json
import sys
from collections import Counter
from pathlib import Path

SIDE = Path(__file__).resolve().parent.parent / 'data_local' / 'nli_conflict_val_soft.meta.json'
HOLDOUT = ['email', 'desk_floor', 'degree', 'barber']


def main():
    side = json.load(open(SIDE, encoding='utf-8'))
    assert len(side) == 300
    print(f'val_soft 300 | holdout rows = {sum(1 for m in side if m["cat"] in HOLDOUT)} '
          f'(4-cat) / {sum(1 for m in side if m["cat"] in HOLDOUT and m["cat"] != "barber")} (3-cat excl barber)')
    header = ['round', 'total_err', 'in_dist_err', 'holdout_err_4cat', 'holdout_err_3cat',
              'email', 'desk_floor', 'degree', 'barber']
    print('\t'.join(header))
    for arg in sys.argv[1:]:
        rows = json.load(open(arg, encoding='utf-8'))
        assert len(rows) == 300, (arg, len(rows))
        per_cat = Counter()
        total = in_dist = h4 = h3 = 0
        for i, r in enumerate(rows):
            cat = side[i]['cat']
            ok = r['pred'] == r['label'] if 'pred' in r else None
            if ok is None:  # v4 schema: pred absent -> derive from p_true+label? val_soft_v4.json has pred
                ok = ('true' if r['p_true'] >= 0.5 else 'false') == r['label']
            if not ok:
                total += 1
                per_cat[cat] += 1
                if cat in HOLDOUT:
                    h4 += 1
                    if cat != 'barber':
                        h3 += 1
                else:
                    in_dist += 1
        tag = Path(arg).stem.replace('val_soft_', '').replace('probs_', '')
        print('\t'.join([tag, str(total), str(in_dist), str(h4), str(h3)] +
                        [str(per_cat.get(c, 0)) for c in HOLDOUT]))


if __name__ == '__main__':
    main()
