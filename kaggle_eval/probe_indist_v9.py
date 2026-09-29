# -*- coding: utf-8 -*-
"""V9 §5 report item (no gate): in-dist hit rate of the waver / change-done
shapes, on FRESH literals that never entered training (train_v9 / val /
val_soft all excluded at state level) -- a held-out generalization reading
for the two shapes block C/D added, complementing val_soft's holdout view.

32 rows: per in-dist cat 1 waver + 1 change per language (gym_brand ZH-only).
Semantic ZH question (same rendering family as the realtest battery).
"""
import json
import os
import random
import sys

import laya

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import OUT_DIR, patch_categories
from gen_soft_conflicts_v6 import swap_vals
from gen_soft_conflicts_v9 import WAVER_PHRASES, WAVER_ACTION, CHANGE_T, WAVER_PLAN, CHANGE_PLAN
from realtest_v2 import QUESTION

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v9'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\indist_probe_v9.json'

SEED = 20260930
CATS = list(WAVER_PLAN)  # city, car, job, coffee, lang, team, cloud, gym_brand


def build_rows():
    rng = random.Random(SEED)
    cats = swap_vals(patch_categories())
    taken = set()
    for p in ('nli_conflict_train_v9.jsonl', 'nli_conflict_val.jsonl', 'nli_conflict_val_soft.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                taken.add(json.loads(l)['state'])
    rows = []
    for fam in CATS:
        for lang in ((ZH,) if fam == 'gym_brand' else (ZH, EN)):
            c = next(cc for cc in cats if cc['key'] == fam and cc['lang'] == lang)
            for shape in ('waver_consider', 'change_done'):
                made, guard = 0, 0
                while made < 1 and guard < 5000:
                    guard += 1
                    a, b = rng.sample(c['vals'], 2)
                    known = rng.choice(c['known']).format(A=a)
                    if shape == 'waver_consider':
                        act = rng.choice(WAVER_ACTION[(fam, lang)])
                        new = rng.choice(WAVER_PHRASES[lang]).format(X=act.format(B=b))
                    else:
                        new = rng.choice(CHANGE_T[(fam, lang)]).format(A=a, B=b)
                    state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                    if state in taken:
                        continue
                    taken.add(state)
                    rows.append({'state': state, 'questions': {'conflict': QUESTION},
                                 'gold': {'conflict': {'label': 'false' if shape == 'waver_consider' else 'true'}},
                                 'meta': {'family': fam, 'lang': lang, 'shape': shape}})
                    made += 1
                assert made == 1, (fam, lang, shape)
    rng.shuffle(rows)
    return rows


def main():
    rows = build_rows()
    agent = laya.load(CKPT, device='cpu')
    results = []
    for r in rows:
        st = json.loads(r['state'])
        a = agent.predict(st, r['questions'])['answers']['conflict']
        p_true = float(a['noul'])
        got = 'true' if p_true >= 0.5 else 'false'
        results.append({'shape': r['meta']['shape'], 'family': r['meta']['family'],
                        'lang': r['meta']['lang'], 'expect': r['gold']['conflict']['label'],
                        'got': got, 'p_conflict': p_true, 'ok': got == r['gold']['conflict']['label'],
                        'known': st['known'], 'new': st['new']})
    from collections import Counter
    per_shape = Counter((r['shape'], r['ok']) for r in results)
    n_waver = sum(1 for r in results if r['shape'] == 'waver_consider')
    n_change = len(results) - n_waver
    waver_ok = sum(1 for r in results if r['shape'] == 'waver_consider' and r['ok'])
    change_ok = sum(1 for r in results if r['shape'] == 'change_done' and r['ok'])
    for r in results:
        print('OK  ' if r['ok'] else 'MISS', r['shape'], r['family'], r['lang'],
              r['expect'], '->', r['got'], round(r['p_conflict'], 4))
    print(f'waver(=false) in-dist: {waver_ok}/{n_waver} | change-done(=true) in-dist: {change_ok}/{n_change}'
          f' | overall {waver_ok + change_ok}/{len(results)}')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': CKPT, 'seed': SEED, 'rows': results,
                   'waver_ok': waver_ok, 'waver_n': n_waver,
                   'change_ok': change_ok, 'change_n': n_change}, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
