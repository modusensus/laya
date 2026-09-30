# -*- coding: utf-8 -*-
"""v10 data arm (HANDOFF_NLI_V10.md §3): SINGLE-LEVER L-arm corpus = v9 minus
block B.  S-arm reuses the v9 corpus unchanged (dataset v14, no new file).

The one change (§3.2): the 40 attr_contradiction rows (kind='true',
cat='pet_attr', same surface family as B2 / attr_consistent) are REMOVED --
the pre-registered suspect for re-inflating "same-subject pet attribute =>
conflict" (v8 without block B had the lowest B2, 0.7491; v9 with block B
rebounded to 0.889 despite 2.7x dose).

Replay discipline (§3.2): block B is STILL GENERATED and only its rows are
dropped, so every rng-stream position after it stays identical to v9 and the
A/C/D rows' `state` values are byte-identical to v9 (§3.3-6 hard assert).
Block B gets a PRIVATE labelset pool + label_rng: the main pool is sized
15780 (exact §3.3-3 ratios) and must not be over-popped; block B consumes no
target_rng draws (kind='true' has zero jitter), so C/D golds stay byte-
identical to v9 too.  Labelset SURFACE is reshuffled wholesale vs v9 (pool
size change; v8->v9 precedent).

v10 train = 15420 carried + 360 v9 new (attr_consistent 200 + waver 120 +
change_done 40) = 15780 rows.  val (1000) / val_soft (300) NOT regenerated.

Deterministic: seed 20260930 (== SEED from v4..v9).
"""
import json
import os
import random
from collections import Counter

from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import OUT_DIR, patch_categories, patch_hc
from gen_soft_conflicts_v6 import (
    SEED_V6, question_for, make_ls_pool, next_ls, TARGETS, gold_for,
    swap_vals, rewrite_hc, make_neg_rows, make_neg_unrel_rows, make_neg5_rows,
    SHAPE_CATS, make_cf_rows, make_cf_unrel_rows, make_soft_rows_v6, assert_no_case_leak)
from gen_soft_conflicts_v7 import (
    PET_HC, DOCTOR_HC, NEW_HC_SUBFAMS, ALT_PRAISE_T, ALT_PRAISE_FAMS,
    make_alt_praise_rows, METRIC_CATS, make_metric_rows, make_hc_rows_pool)
from gen_soft_conflicts_v8 import (
    SEED_V8, PET_HC_SPECIES, ATTR_FORBIDDEN, make_attr_rows, assert_no_case_leak_v8)
from gen_soft_conflicts_v9 import (
    SEED_V9, make_attr_add_rows, make_attr_contra_rows, make_waver_rows,
    make_change_rows, assert_newblock_field_no_leak, HOLDOUT_CATS)

SEED_V10 = 20260930  # == SEED from v4..v9: carried blocks replay v9's stream


def main():
    rng = random.Random(SEED_V10)
    label_rng = random.Random(SEED_V10 + 1)
    target_rng = random.Random(SEED_V10 + 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # acceptance sets must never leak into train (carried)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: same extraction as v4..v9 (exact state match), name-rotated too
    ls_pool = make_ls_pool(label_rng, 15780)
    v3_rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, 'nli_conflict_train_v3.jsonl'),
                                           encoding='utf-8') if l.strip()]
    mnli_states = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, 'nli_conflict_train.jsonl'),
                                                       encoding='utf-8') if l.strip()}
    mnli_rows = [r for r in v3_rows if r['state'] in mnli_states]
    assert len(mnli_rows) == 6000, f'MNLI extraction got {len(mnli_rows)}'
    mnli_kind = {'false': 'entailment/neutral', 'true': 'contradiction'}
    for r in mnli_rows:
        lab = r['gold']['conflict']['label']
        r['_meta'] = {'cat': 'mnli', 'kind': mnli_kind[lab], 'lang': 'en',
                      'labelset': next_ls(ls_pool)}
        r['questions'] = question_for(r['_meta']['labelset'], ZH, label_rng)

    # carried v6/v7/v8/v9 blocks in v9's exact stream order (byte-identical)
    cats = swap_vals(patch_categories())
    rewrite_hc()
    train_pool = [c for c in cats if not c['holdout']]
    soft_rows = make_soft_rows_v6(rng, train_pool,
                                  {'true': 1800, 'compat': 1440, 'consid': 1080, 'unrelated': 1080},
                                  {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    alt_praise_rows = make_alt_praise_rows(rng, train_pool, 240, {ZH: 0.72, EN: 0.28},
                                           seen, label_rng, target_rng, ls_pool)
    metric_rows = make_metric_rows(rng, METRIC_CATS, {'true': 40, 'consid': 40, 'compat': 40, 'unrelated': 40},
                                   {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    base_pool = [h for h in patch_hc() if h['key'] not in {f['key'] for f in NEW_HC_SUBFAMS}]
    hc_base_rows = make_hc_rows_pool(rng, {'hardconstraint': 300, 'hc_safe': 180, 'hc_unrelated': 120},
                                     {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, base_pool, ls_pool)
    pet_rows = make_hc_rows_pool(rng, {'hardconstraint': 140, 'hc_safe': 40, 'hc_unrelated': 20},
                                 {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, PET_HC, ls_pool)
    doctor_rows = make_hc_rows_pool(rng, {'hardconstraint': 140, 'hc_safe': 40, 'hc_unrelated': 20},
                                    {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, DOCTOR_HC, ls_pool)
    neg_rows = make_neg_rows(rng, {'neg_true': 400, 'neg_false': 300, 'neg_false_boundary': 100},
                             {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    neg5_rows = make_neg5_rows(rng, seen, label_rng, target_rng, ls_pool)
    neg_unrel_rows = make_neg_unrel_rows(rng, 400, {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    shape_cats = [dict(c, lang=EN if c['key'].endswith('_en') else ZH) for c in SHAPE_CATS]
    cf_rows = make_cf_rows(rng, train_pool, shape_cats, {'cf_true': 300, 'cf_false': 300},
                           {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    cf_unrel_rows = make_cf_unrel_rows(rng, shape_cats, 300, {ZH: 0.72, EN: 0.28}, seen,
                                       label_rng, target_rng, ls_pool)

    # v8/v9 attr replay + v9 blocks A/C/D (byte-identical states to v9)
    attr_rows = make_attr_rows(rng, seen, label_rng, target_rng, ls_pool)      # v8's +120
    block_a = make_attr_add_rows(rng, seen, label_rng, target_rng, ls_pool)    # +200
    # §3.2: block B STILL GENERATED (keeps the main rng stream + seen exactly
    # at v9's position so C/D replay byte-identically) but its rows are DROPPED.
    # Private ls pool + label_rng keep the 15780 main pool exact (§3.3-3);
    # target_rng untouched (kind='true' consumes zero draws either way).
    b_ls_pool = make_ls_pool(random.Random(SEED_V10 + 100), 40)
    block_b = make_attr_contra_rows(rng, seen, random.Random(SEED_V10 + 101),
                                    target_rng, b_ls_pool)                     # +40, dropped
    assert len(block_b) == 40 and all(r['_meta']['shape'] == 'attr_contradiction' for r in block_b)
    block_c = make_waver_rows(rng, train_pool, seen, label_rng, target_rng, ls_pool)   # +120
    block_d = make_change_rows(rng, train_pool, seen, label_rng, target_rng, ls_pool)  # +40
    new_rows = block_a + block_c + block_d  # block B intentionally excluded

    v10_train = (mnli_rows + soft_rows + alt_praise_rows + metric_rows
                 + hc_base_rows + pet_rows + doctor_rows
                 + neg_rows + neg5_rows + neg_unrel_rows + cf_rows + cf_unrel_rows
                 + attr_rows + new_rows)
    rng.shuffle(v10_train)

    # ---- §3.3 hard invariants -------------------------------------------------
    assert len(v10_train) == 15780, len(v10_train)
    n_true = sum(1 for r in v10_train if r['gold']['conflict']['label'] == 'true')
    n_false = len(v10_train) - n_true
    states = [r['state'] for r in v10_train]
    assert len(set(states)) == 15780, 'duplicate states'
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        other = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, p), encoding='utf-8') if l.strip()}
        assert set(states).isdisjoint(other), f'leak into {p}'

    kind_counts = Counter(r['_meta']['kind'] for r in v10_train)
    expect_kinds = {'contradiction': 2000, 'entailment/neutral': 4000, 'true': 1880, 'compat': 1800,
                    'consid': 1480, 'unrelated': 1120, 'hardconstraint': 580, 'hc_safe': 260,
                    'hc_unrelated': 160, 'neg_true': 568, 'neg_false': 404, 'neg_false_boundary': 228,
                    'neg_unrelated': 400, 'cf_true': 300, 'cf_false': 300, 'cf_unrelated': 300}
    assert dict(kind_counts) == expect_kinds, (kind_counts, expect_kinds)

    # §3.3-4: attr_contradiction must NOT appear in the output corpus
    shape_counts = Counter(r['_meta'].get('shape') for r in v10_train if r['_meta'].get('shape'))
    assert shape_counts.get('attr_contradiction', 0) == 0, shape_counts
    expect_shapes = {'alt_praise': 240, 'metric_shift': 40, 'attr_consistent': 320,
                     'waver_consider': 120, 'change_done': 40,
                     'pet': 80, 'never': 80, 'relapse': 80, 'dneg': 80, 'nocar': 80}
    assert dict(shape_counts) == expect_shapes, (shape_counts, expect_shapes)

    label_mix = Counter(r['_meta']['labelset'] for r in v10_train)
    assert label_mix[5] == 3156 and label_mix[6] == 3156, label_mix
    assert sum(label_mix[s] for s in (1, 2, 3, 4)) == 15780 - 6312, label_mix
    for ls in range(1, 7):
        kinds_with = {r['_meta']['kind'] for r in v10_train if r['_meta']['labelset'] == ls}
        assert len(kinds_with) >= 3, (ls, len(kinds_with))

    for r in v10_train:
        g = r['gold']['conflict']
        probs = g['probabilities']
        assert abs(sum(probs.values()) - 1) < 1e-6, probs
        assert max(probs, key=probs.get) == g['label'], (probs, g['label'])
        pg = probs[g['label']]
        kind = r['_meta']['kind']
        base, j = TARGETS[kind]
        assert abs(pg - 0.5) >= 0.2, (kind, pg)
        if j == 0:
            assert pg == base, (kind, pg)
        else:
            assert base - j - 1e-9 <= pg <= base + j + 1e-9, (kind, pg, base, j)
        assert r['_meta']['labelset'] in (1, 2, 3, 4, 5, 6)
        for k in ('cat', 'kind', 'lang', 'labelset'):
            assert k in r['_meta'], r['_meta']

    # §3.3-5: forbidden literals / defiance markers on the kept new blocks (=0)
    for r in new_rows:
        st = json.loads(r['state'])
        for w in ATTR_FORBIDDEN:
            assert w not in st['known'] and w not in st['new'], (w, st)

    # §3.3-5: kept new blocks -- case/diag field 0, val/val_soft delta-new 0
    assert_newblock_field_no_leak(new_rows, 'v10 kept new blocks')

    # §3.3-5: holdout cats zero rows in the new blocks (carried controls stay)
    holdout_hits = [r['_meta'] for r in new_rows
                    if r['_meta']['cat'] in HOLDOUT_CATS or r['_meta'].get('family') in HOLDOUT_CATS]
    assert not holdout_hits, f'holdout-cat rows in new blocks: {holdout_hits[:2]}'

    # §3.3-6 对照断言: diff vs v9 = exactly block B's 40 rows; the other 15780
    # rows are state-identical with equal gold to v9
    v9_rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, 'nli_conflict_train_v9.jsonl'),
                                           encoding='utf-8') if l.strip()]
    v9_by_state = {r['state']: json.dumps(r['gold'], sort_keys=True) for r in v9_rows}
    assert len(v9_rows) == 15820
    b_states = {r['state'] for r in block_b}
    v10_state_set = set(states)
    v9_only = {s for s in v9_by_state if s not in v10_state_set}
    assert v9_only == b_states and len(v9_only) == 40, (len(v9_only), len(b_states & v10_state_set))
    assert not (v10_state_set & b_states), 'dropped block-B rows leaked into the corpus'
    gold_mismatch = [r['state'] for r in v10_train
                     if r['state'] in v9_by_state
                     and json.dumps(r['gold'], sort_keys=True) != v9_by_state[r['state']]]
    assert not gold_mismatch, f'gold drifted on {len(gold_mismatch)} carried rows'
    print(f'§3.3-6 diff vs v9: exactly 40 dropped attr_contradiction rows; '
          f'15780 carried rows state-identical, gold equal (0 mismatches)')

    # carried upgrade: whole-train field-level vs 35 cases + 18 diag pairs
    assert_no_case_leak_v8(v10_train)

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v10.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v10_train:
            meta = r.pop('_meta')
            meta['name_a'] = r['questions']['conflict']['labels']['false']
            meta['name_b'] = r['questions']['conflict']['labels']['true']
            r['meta'] = meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v10_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false / n_true:.2f})')
    print('meta.kind:', dict(sorted(Counter(r["meta"]["kind"] for r in v10_train).items())))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v10_train)))
    print('meta.labelset:', dict(sorted(Counter(r["meta"]["labelset"] for r in v10_train).items())))
    sem = sum(label_mix[s] for s in (1, 2, 3, 4))
    print(f'labelset ratio: semantic {sem / 15780:.1%}, LS5 {label_mix[5] / 15780:.1%}, LS6 {label_mix[6] / 15780:.1%}')
    print('new-block shapes (B dropped):', {k: v for k, v in sorted(shape_counts.items())
                                            if k in ('attr_consistent', 'waver_consider', 'change_done')})
    print('-- kept new-block sample rows:')
    shown = {'pet_name': 0, 'waver': 0, 'change_done': 0}
    for r in new_rows:
        st = json.loads(r['state'])
        fam = r['meta']['family']
        if fam in shown and shown[fam] < 1:
            print(f'   [{r["meta"]["shape"]}|{r["meta"]["lang"]}] known={st["known"][:42]} | new={st["new"][:42]}')
            shown[fam] += 1


if __name__ == '__main__':
    main()
