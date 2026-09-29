# -*- coding: utf-8 -*-
"""v8 data arm (HANDOFF_NLI_V8.md §3): same-subject attribute-consistency rows
to recover bias-diag control B2 -- the single v7 FAIL.  Everything else is
frozen at v7 ("少动原则": carried blocks replay v7's rng stream byte-identically,
the attr block is generated last, only the final shuffle sees more rows).

The one new block (§3.1, +120, kind='compat' @ TARGETS['compat'] 0.93+-0.02):

  known and new describe the SAME subject (pet / user) and can both be true,
  no constraint violated => false (compatible).  B2 is the only diag pair of
  the form "pet known + pet-attribute new"; after the v7 pet family (140 true
  rows) the model maps that shape to true (p 0.8985).  Three shapes, all used:

    1. pet_name   -- name x appearance, B2's exact shape with fresh literals
                     (name pool != 小白, color pool != 白色);
    2. pet_benign -- constrained subject (PET_HC known, verbatim) + benign
                     attribute ("allergic to beef" + "is a corgi" => false);
    3. doctor_bg  -- medical context (DOCTOR_HC known, verbatim) + background
                     fact ("no spicy" + "had a gastroscopy last year" => false).

  Quotas (exact): pet_attr 80 (zh 48 / en 32) = pet_name 40 + pet_benign 40;
  doctor_attr 40 (zh 24 / en 16).  meta.shape='attr_consistent' on all.
  Guards: no defiance markers; no field equality with the 35 acceptance
  cases, the 18 diag pairs, or val/val_soft; forbidden literals
  小白 / 白色 / 开心果 (asserted).

Leak assert upgraded (§3.3-6): the v6 field-level 35-case check + the 18
bias-diag pairs (from noul_bias_diag, safe to import: __main__ guard, lazy
laya import).  v7 train already scanned 0-hit against all 36 diag fields
(V8 §2.1), so the upgraded assert should pass untouched.

v8 train = v7's 15300 + 120 attr rows = 15420 rows.  val (1000) / val_soft
(300) are NOT regenerated (byte-identical, directly comparable).

Deterministic: seed 20260930 (== SEED from v4..v7; carried blocks consume the
stream in v7's exact order).
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

SEED_V8 = 20260930  # == SEED from v4..v7: carried blocks replay v7's stream


# --- §3.1 shape 1: name x appearance (B2's shape, fresh literals) -------------
# name pool excludes 小白, color pool excludes 白色 (§3.1 禁词, asserted below)
PET_NAME_T = {
    ZH: dict(known=['用户家的猫叫{N}', '用户养的猫名字是{N}', '用户家小猫的名字是{N}'],
             names=['咪咪', '汤圆', '年糕'],
             new=['用户的猫是只{C}猫', '用户的猫是一只{C}猫'],
             colors=['黑', '橘', '灰', '三花']),
    EN: dict(known=["The user's cat is named {N}", "The user's cat goes by {N}",
                    "The cat the user keeps is called {N}"],
             names=['Mochi', 'Milo', 'Muffin'],
             new=["The user's cat is a {C} cat", "The user's cat is {C}"],
             colors=['black', 'orange', 'gray', 'calico']),
}

# --- §3.1 shape 2: constrained subject (PET_HC known verbatim) + benign attr --
# species per PET_HC key so known/new always refer to the same animal
PET_HC_SPECIES = {
    ('pet_beef', ZH): 'dog', ('pet_beef', EN): 'dog',
    ('pet_chicken', ZH): 'cat', ('pet_chicken', EN): 'cat',
    ('pet_dairy', ZH): 'dog', ('pet_dairy', EN): 'dog',
    ('pet_avocado', ZH): 'parrot', ('pet_avocado', EN): 'parrot',
    ('pet_grape', ZH): 'dog', ('pet_grape', EN): 'dog',
}
PET_BENIGN_ATTR = {
    ZH: {
        'dog': ['用户家的狗是柯基', '用户家的狗是去年领养的', '用户家的狗今年三岁',
                '用户家的狗是金毛', '用户家的狗疫苗打齐了', '用户家的狗特别爱玩接球',
                '用户家的狗是朋友送的', '用户家的狗每天早晚各遛一次'],
        'cat': ['用户家的猫是英短', '用户家的猫是去年领养的', '用户家的猫今年两岁',
                '用户家的猫是狸花', '用户家的猫已经绝育了', '用户家的猫白天最爱晒太阳'],
        'parrot': ['用户家的鹦鹉是玄凤', '用户家的鹦鹉养了两年了', '用户家的鹦鹉是去年买的',
                   '用户家的鹦鹉会说你好'],
    },
    EN: {
        'dog': ["The user's dog is a corgi", "The user's dog was adopted last year",
                "The user's dog is three years old", "The user's dog is a golden retriever",
                "The user's dog is fully vaccinated", "The user's dog loves playing fetch"],
        'cat': ["The user's cat is a British shorthair", "The user's cat was adopted last year",
                "The user's cat is two years old", "The user's cat is a tabby",
                "The user's cat is already neutered", "The user's cat loves sunbathing"],
        'parrot': ["The user's parrot is a cockatiel", "The user's parrot has been with them for two years",
                   "The user's parrot was bought last year", "The user's parrot can say hello"],
    },
}

# --- §3.1 shape 3: medical context (DOCTOR_HC known verbatim) + background fact
DOCTOR_BG = {
    ZH: {
        'doctor_rest': ['用户办公室给配了升降桌', '用户上个月换了偏硬的床垫',
                        '用户家的浴室刚装了扶手', '用户住的小区里有康复器材区'],
        'doctor_postop': ['用户把复查约在了下个月', '用户请了两周的假在家休养',
                          '用户把体检报告收进了文件袋', '用户的病假条已经交给了公司'],
        'doctor_gastritis': ['用户去年做过一次胃镜', '用户口味一直偏清淡',
                             '用户早饭常喝小米粥', '用户家里的药盒放在餐边柜'],
        'doctor_bp': ['用户家里备了血压仪', '用户口味一直偏清淡',
                      '用户每天晚饭后散步半小时', '用户体检时其他指标都正常'],
    },
    EN: {
        'doctor_rest': ["The user's office provided a standing desk", 'The user switched to a firmer mattress last month',
                        "The user's bathroom just got grab bars", 'The user lives next to a rehab gym'],
        'doctor_postop': ['The user booked the follow-up for next month', 'The user took two weeks off to recover',
                          'The user filed the checkup report away', 'The user already handed the sick note to HR'],
        'doctor_gastritis': ['The user had a gastroscopy last year', 'The user has always preferred mild food',
                             'The user often has millet porridge for breakfast', 'The user keeps the medicine box by the pantry'],
        'doctor_bp': ['The user keeps a blood-pressure monitor at home', 'The user has always preferred mild food',
                      'The user walks for half an hour after dinner', 'Everything else was normal at the last checkup'],
    },
}

# exact per-cell quotas (V8 §3.1): pet_attr 80 = zh 48 / en 32 (pet_name +
# pet_benign evenly), doctor_attr 40 = zh 24 / en 16; overall lang_mix 60/40
ATTR_PLAN = [
    # (cat, family/shape, quota per lang)
    ('pet_attr', 'pet_name', {ZH: 24, EN: 16}),
    ('pet_attr', 'pet_benign', {ZH: 24, EN: 16}),
    ('doctor_attr', 'doctor_bg', {ZH: 24, EN: 16}),
]

ATTR_FORBIDDEN = ['小白', '白色', '开心果', '不顾', '偏要', '硬要', '执意', '照样']


def make_attr_rows(rng, seen, label_rng, target_rng, ls_pool):
    """+120 same-subject attribute-consistency rows => false (compat tier)."""
    rows = []
    for cat, fam, quota in ATTR_PLAN:
        for lang, n in quota.items():
            # enumerate the (known, new) literal space for this cell
            cands = []
            if fam == 'pet_name':
                t = PET_NAME_T[lang]
                for kt in t['known']:
                    for name in t['names']:
                        k = kt.format(N=name)
                        for nt in t['new']:
                            for color in t['colors']:
                                cands.append((k, nt.format(C=color)))
            elif fam == 'pet_benign':
                attrs = PET_BENIGN_ATTR[lang]
                for h in PET_HC:
                    if h['lang'] != lang:
                        continue
                    sp = PET_HC_SPECIES[(h['key'], lang)]
                    for k in h['known']:
                        for new in attrs[sp]:
                            cands.append((k, new))
            else:  # doctor_bg
                bgs = DOCTOR_BG[lang]
                for h in DOCTOR_HC:
                    if h['lang'] != lang:
                        continue
                    for k in h['known']:
                        for new in bgs[h['key']]:
                            cands.append((k, new))
            rng.shuffle(cands)
            made, guard = 0, 0
            while made < n and guard < n * 2000:
                guard += 1
                k, new = cands[guard % len(cands)]
                state = json.dumps({'known': k, 'new': new}, ensure_ascii=False)
                if state in seen:
                    continue
                seen.add(state)
                ls = next_ls(ls_pool)
                rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                             'gold': {'conflict': gold_for('compat', target_rng)},
                             '_meta': {'cat': cat, 'kind': 'compat', 'lang': lang,
                                       'labelset': ls, 'shape': 'attr_consistent',
                                       'family': fam}})
                made += 1
            assert made == n, (cat, fam, lang, made, n)
            # §3.1 guards: forbidden literals + defiance markers never appear
            for k, new in ((json.loads(r['state'])['known'], json.loads(r['state'])['new'])
                           for r in rows if r['_meta']['family'] == fam and r['_meta']['lang'] == lang):
                for w in ATTR_FORBIDDEN:
                    assert w not in k and w not in new, (fam, lang, w, k, new)
    return rows


def assert_attr_field_no_leak(attr_rows):
    """§3.1 护栏: attr rows share no known/new field with the 35 acceptance
    cases, the 18 diag pairs, or val/val_soft (field-level equality)."""
    from realtest_v2 import CASES, NEGATION_CASES
    from realtest_v4 import NEW_CASES
    from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS
    ks, ns = set(), set()
    for (state, _e), _note in (list(CASES) + list(NEGATION_CASES) + list(NEW_CASES)
                               + list(CONTROL_PAIRS) + list(SWAP_PAIRS)):
        ks.add(state['known'])
        ns.add(state['new'])
    for p in ('nli_conflict_val.jsonl', 'nli_conflict_val_soft.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                st = json.loads(l)['state']
                st = json.loads(st) if isinstance(st, str) else st
                ks.add(st.get('known'))
                ns.add(st.get('new'))
    hits = []
    for r in attr_rows:
        st = json.loads(r['state'])
        if st['known'] in ks or st['new'] in ns:
            hits.append((st['known'], st['new']))
    assert not hits, f'attr row field-equality leak: {hits[:3]}'


def assert_no_case_leak_v8(rows):
    """v6 §3.6 (35 cases, field-level) upgraded per V8 §3.3-6 with the 18
    bias-diag pairs (14 controls + 4 swaps)."""
    assert_no_case_leak(rows)
    from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS
    ks, ns = set(), set()
    for (state, _e), _note in list(CONTROL_PAIRS) + list(SWAP_PAIRS):
        ks.add(state['known'])
        ns.add(state['new'])
    hits = []
    for r in rows:
        st = r['state']
        st = json.loads(st) if isinstance(st, str) else st
        if st.get('known') in ks or st.get('new') in ns:
            hits.append((st.get('known'), st.get('new')))
    assert not hits, f'diag-pair field leak {len(hits)} rows, e.g. {hits[:2]}'


def main():
    rng = random.Random(SEED_V8)
    label_rng = random.Random(SEED_V8 + 1)
    target_rng = random.Random(SEED_V8 + 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # acceptance sets must never leak into train (V7 §3.1/§9, carried)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: same extraction as v4..v7 (exact state match), name-rotated too
    ls_pool = make_ls_pool(label_rng, 15420)
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

    # carried v6/v7 blocks in v7's exact stream order (byte-identical content)
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

    # §3.1 attr block LAST: carried blocks replayed v7's exact rng positions,
    # so their content is byte-identical to v7; the attr rows consume the tail
    attr_rows = make_attr_rows(rng, seen, label_rng, target_rng, ls_pool)

    v8_train = (mnli_rows + soft_rows + alt_praise_rows + metric_rows
                + hc_base_rows + pet_rows + doctor_rows
                + neg_rows + neg5_rows + neg_unrel_rows + cf_rows + cf_unrel_rows
                + attr_rows)
    rng.shuffle(v8_train)

    # ---- §3.3 hard invariants -------------------------------------------------
    assert len(v8_train) == 15420, len(v8_train)
    n_true = sum(1 for r in v8_train if r['gold']['conflict']['label'] == 'true')
    n_false = len(v8_train) - n_true
    states = [r['state'] for r in v8_train]
    assert len(set(states)) == 15420, 'duplicate states'
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        other = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, p), encoding='utf-8') if l.strip()}
        assert set(states).isdisjoint(other), f'leak into {p}'

    kind_counts = Counter(r['_meta']['kind'] for r in v8_train)
    expect_kinds = {'contradiction': 2000, 'entailment/neutral': 4000, 'true': 1840, 'compat': 1600,
                    'consid': 1360, 'unrelated': 1120, 'hardconstraint': 580, 'hc_safe': 260,
                    'hc_unrelated': 160, 'neg_true': 568, 'neg_false': 404, 'neg_false_boundary': 228,
                    'neg_unrelated': 400, 'cf_true': 300, 'cf_false': 300, 'cf_unrelated': 300}
    assert dict(kind_counts) == expect_kinds, (kind_counts, expect_kinds)

    # §3.3-2: new-block per-family counts, exact
    attr = [r for r in v8_train if r['_meta'].get('shape') == 'attr_consistent']
    attr_cat = Counter(r['_meta']['cat'] for r in attr)
    attr_cell = Counter((r['_meta']['cat'], r['_meta']['family'], r['_meta']['lang']) for r in attr)
    expect_cells = {('pet_attr', 'pet_name', ZH): 24, ('pet_attr', 'pet_name', EN): 16,
                    ('pet_attr', 'pet_benign', ZH): 24, ('pet_attr', 'pet_benign', EN): 16,
                    ('doctor_attr', 'doctor_bg', ZH): 24, ('doctor_attr', 'doctor_bg', EN): 16}
    assert dict(attr_cell) == expect_cells, (attr_cell, expect_cells)
    assert attr_cat['pet_attr'] == 80 and attr_cat['doctor_attr'] == 40, attr_cat

    label_mix = Counter(r['_meta']['labelset'] for r in v8_train)
    assert label_mix[5] == 3084 and label_mix[6] == 3084, label_mix
    assert sum(label_mix[s] for s in (1, 2, 3, 4)) == 15420 - 6168, label_mix
    for ls in range(1, 7):
        kinds_with = {r['_meta']['kind'] for r in v8_train if r['_meta']['labelset'] == ls}
        assert len(kinds_with) >= 3, (ls, len(kinds_with))

    for r in v8_train:
        q = r['questions']['conflict']
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

    # §3.3-7: forbidden literals / defiance markers on the new block (=0)
    for r in attr:
        st = json.loads(r['state'])
        for w in ATTR_FORBIDDEN:
            assert w not in st['known'] and w not in st['new'], (w, st)
        assert r['gold']['conflict']['label'] == 'false', r['gold']

    # §3.1 guard: attr rows share no field with 35 cases / diag pairs / val sets
    assert_attr_field_no_leak(attr)

    # ---- §3.3-6 leak assert (hard invariant, upgraded with the 18 diag pairs) -
    assert_no_case_leak_v8(v8_train)

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v8.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v8_train:
            meta = r.pop('_meta')
            meta['name_a'] = r['questions']['conflict']['labels']['false']
            meta['name_b'] = r['questions']['conflict']['labels']['true']
            r['meta'] = meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v8_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false / n_true:.2f})')
    print('meta.kind:', dict(sorted(Counter(r["meta"]["kind"] for r in v8_train).items())))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v8_train)))
    print('meta.labelset:', dict(sorted(Counter(r["meta"]["labelset"] for r in v8_train).items())))
    sem = sum(label_mix[s] for s in (1, 2, 3, 4))
    print(f'labelset ratio: semantic {sem / 15420:.1%}, LS5 {label_mix[5] / 15420:.1%}, LS6 {label_mix[6] / 15420:.1%}')
    print('attr block per cell (cat/family/lang):', {f'{c}/{f}/{l}': n for (c, f, l), n in sorted(attr_cell.items())})
    fam_counts = Counter((r['meta'].get('family'), r['meta']['kind']) for r in v8_train
                         if r['meta'].get('family') in {f['key'] for f in NEW_HC_SUBFAMS})
    pet_n = sum(n for (fam, _), n in fam_counts.items() if fam.startswith('pet_'))
    doc_n = sum(n for (fam, _), n in fam_counts.items() if fam.startswith('doctor_'))
    print(f'HC new families (unchanged): pet {pet_n} / doctor {doc_n}')
    print('-- attr sample rows (B2-shape, fresh literals):')
    shown = {'pet_name': 0, 'pet_benign': 0, 'doctor_bg': 0}
    for r in attr:
        st = json.loads(r['state'])
        fam = r['meta']['family']
        if shown[fam] < 2:
            p = r['gold']['conflict']['probabilities']['false']
            print(f'   [{fam}|{r["meta"]["lang"]}] known={st["known"]} | new={st["new"]} | p(false)={p}')
            shown[fam] += 1


if __name__ == '__main__':
    main()
