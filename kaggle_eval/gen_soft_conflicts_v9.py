# -*- coding: utf-8 -*-
"""v9 data arm (HANDOFF_NLI_V9.md §3): dose the attr block, thicken the
waver/change shapes in-dist, add the attr reverse guard -- everything else
frozen at v8 ("照着 v8 改": carried blocks replay the rng stream byte-identically,
the four new blocks consume the stream tail).

Blocks (§3.1-§3.4, +400 total -> 15820 rows):

  A. attr add (+200, kind='compat'): extended pools (names +3, colors +2,
     pet benign +6, doctor bg +4 per language -- never copied rows),
     quotas pet_name 120 (zh 72/en 48) + pet_benign 40 (zh 24/en 16)
     + doctor_bg 40 (zh 24/en 16).  pet_name knowns include the B2-isomorphic
     「用户的猫叫{N}」 shape; 叫/named-family share asserted >= 1/3.
  B. attr reverse guard (+40, kind='true', shape='attr_contradiction'):
     same-subject attribute CONTRADICTION => true (appearance x20, name x20;
     zh 12/en 8 each).  Protects the true side at 2.5x attr dose.
  C. waver (+120, kind='consid', shape='waver_consider'): known = current
     state, new = wavering about a change (not yet happened) => false.
     In-dist cats only (city/car/job/coffee/lang/gym_brand/team/cloud;
     gym_brand is ZH-only per the v3 set), holdout cats zero rows (asserted).
  D. change-done (+40, kind='true', shape='change_done'): known = old state,
     new = change already happened => true.  Same in-dist cats, pairs with
     block C (wavering => false / done => true semantic contrast).

Leak asserts (§3.6-6/7): carried v8 upgrade (35 cases + 18 diag pairs,
field-level, whole train) + the four new blocks field-level vs
val/val_soft/cases/diag; forbidden literals 小白/白色/开心果 and defiance
markers asserted 0 on all new rows; holdout cats zero rows asserted.

v9 train = v8's 15420 (byte-identical, verified against the v8 file) + 400.
val (1000) / val_soft (300) NOT regenerated (byte-identical).

Deterministic: seed 20260930 (== SEED from v4..v8; v8 blocks replay the
stream in v8's exact order).
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

SEED_V9 = 20260930  # == SEED from v4..v8: carried blocks replay v8's stream


# --- §3.1 block A pools: v8 pools EXTENDED (names +3, colors +2; no rows
#     copied -- combos are resampled from the larger space) ----------------------
PET_NAME_T_V9 = {
    ZH: dict(known=['用户的猫叫{N}', '用户家的猫叫{N}', '用户养的猫名字是{N}', '用户家小猫的名字是{N}'],
             names=['咪咪', '汤圆', '年糕', '布丁', '煤球', '花卷'],
             new=['用户的猫是只{C}猫', '用户的猫是一只{C}猫'],
             colors=['黑', '橘', '灰', '三花', '奶牛', '玳瑁']),
    EN: dict(known=["The user's cat is named {N}", "The user's cat goes by {N}",
                    "The cat the user keeps is called {N}"],
             names=['Mochi', 'Milo', 'Muffin', 'Oreo', 'Pepper', 'Pumpkin'],
             new=["The user's cat is a {C} cat", "The user's cat is {C}"],
             colors=['black', 'orange', 'gray', 'calico', 'tuxedo', 'tortoiseshell']),
}
PET_BENIGN_ATTR_V9 = {
    ZH: {
        'dog': ['用户家的狗是柯基', '用户家的狗是去年领养的', '用户家的狗今年三岁',
                '用户家的狗是金毛', '用户家的狗疫苗打齐了', '用户家的狗特别爱玩接球',
                '用户家的狗是朋友送的', '用户家的狗每天早晚各遛一次',
                '用户家的狗会握手', '用户家的狗是小型犬', '用户家的狗办了狗证'],
        'cat': ['用户家的猫是英短', '用户家的猫是去年领养的', '用户家的猫今年两岁',
                '用户家的猫是狸花', '用户家的猫已经绝育了', '用户家的猫白天最爱晒太阳',
                '用户家的猫会用猫砂盆', '用户家的猫爱趴窗台'],
        'parrot': ['用户家的鹦鹉是玄凤', '用户家的鹦鹉养了两年了', '用户家的鹦鹉是去年买的',
                   '用户家的鹦鹉会说你好', '用户家的鹦鹉羽毛很顺'],
    },
    EN: {
        'dog': ["The user's dog is a corgi", "The user's dog was adopted last year",
                "The user's dog is three years old", "The user's dog is a golden retriever",
                "The user's dog is fully vaccinated", "The user's dog loves playing fetch",
                "The user's dog can shake hands", "The user's dog is a small breed",
                "The user's dog is licensed"],
        'cat': ["The user's cat is a British shorthair", "The user's cat was adopted last year",
                "The user's cat is two years old", "The user's cat is a tabby",
                "The user's cat is already neutered", "The user's cat loves sunbathing",
                "The user's cat is litter-trained", "The user's cat naps on the windowsill"],
        'parrot': ["The user's parrot is a cockatiel", "The user's parrot has been with them for two years",
                   "The user's parrot was bought last year", "The user's parrot can say hello",
                   "The user's parrot has glossy feathers"],
    },
}
DOCTOR_BG_V9 = {
    ZH: {
        'doctor_rest': ['用户办公室给配了升降桌', '用户上个月换了偏硬的床垫',
                        '用户家的浴室刚装了扶手', '用户住的小区里有康复器材区',
                        '用户把办公室的椅子换成了腰托款', '用户把客厅的沙发换成了偏硬的',
                        '用户预约了下个月的康复科号'],
        'doctor_postop': ['用户把复查约在了下个月', '用户请了两周的假在家休养',
                          '用户把体检报告收进了文件袋', '用户的病假条已经交给了公司',
                          '用户把跑鞋收进了鞋柜顶层', '用户把私教课转给了同事',
                          '用户的复查单据都收在抽屉里'],
        'doctor_gastritis': ['用户去年做过一次胃镜', '用户口味一直偏清淡',
                             '用户早饭常喝小米粥', '用户家里的药盒放在餐边柜',
                             '用户把外卖软件里的重辣店都取关了', '用户把冰美式换成了温红茶',
                             '用户家里囤的是龙须面'],
        'doctor_bp': ['用户家里备了血压仪', '用户口味一直偏清淡',
                      '用户每天晚饭后散步半小时', '用户体检时其他指标都正常',
                      '用户把家里的盐罐换成了小号的', '用户把腌菜罐子清空收了起来',
                      '用户上次的血脂检查是正常的'],
    },
    EN: {
        'doctor_rest': ["The user's office provided a standing desk", 'The user switched to a firmer mattress last month',
                        "The user's bathroom just got grab bars", 'The user lives next to a rehab gym',
                        'The user swapped the office chair for a lumbar-support one',
                        'The user swapped the living-room sofa for a firmer one',
                        'The user booked a rehab appointment for next month'],
        'doctor_postop': ['The user booked the follow-up for next month', 'The user took two weeks off to recover',
                          'The user filed the checkup report away', 'The user already handed the sick note to HR',
                          'The user put the running shoes on the top shelf',
                          'The user transferred the personal-training sessions to a coworker',
                          'The user keeps the follow-up paperwork in a drawer'],
        'doctor_gastritis': ['The user had a gastroscopy last year', 'The user has always preferred mild food',
                             'The user often has millet porridge for breakfast', 'The user keeps the medicine box by the pantry',
                             'The user unfollowed all the spicy places on the delivery app',
                             'The user switched from iced americano to warm tea',
                             'The user stocked plain noodles at home'],
        'doctor_bp': ['The user keeps a blood-pressure monitor at home', 'The user has always preferred mild food',
                      'The user walks for half an hour after dinner', 'Everything else was normal at the last checkup',
                      'The user switched to a smaller salt shaker at home',
                      'The user cleared out the pickle jars',
                      'The lipid panel at the last checkup was normal'],
    },
}

ATTR_PLAN_V9 = [
    # (cat, family, quota per lang)  -- §3.1 exact quotas, assert-exact
    ('pet_attr', 'pet_name', {ZH: 72, EN: 48}),
    ('pet_attr', 'pet_benign', {ZH: 24, EN: 16}),
    ('doctor_attr', 'doctor_bg', {ZH: 24, EN: 16}),
]


def _emit(rng, cands, n, cat, fam, kind, lang, seen, label_rng, target_rng, ls_pool,
          shape, extra=None):
    """Shared emitter: walk a shuffled candidate space, dedupe via seen,
    attach ls/gold/meta; exact quota asserted."""
    assert len(set(cands)) >= n, (fam, lang, len(set(cands)), n)
    rng.shuffle(cands)
    rows, made, guard = [], 0, 0
    while made < n and guard < n * 2000:
        guard += 1
        k, new = cands[guard % len(cands)]
        state = json.dumps({'known': k, 'new': new}, ensure_ascii=False)
        if state in seen:
            continue
        seen.add(state)
        ls = next_ls(ls_pool)
        meta = {'cat': cat, 'kind': kind, 'lang': lang, 'labelset': ls,
                'shape': shape, 'family': fam}
        if extra:
            meta.update(extra)
        rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                     'gold': {'conflict': gold_for(kind, target_rng)}, '_meta': meta})
        made += 1
    assert made == n, (fam, lang, made, n)
    return rows


def make_attr_add_rows(rng, seen, label_rng, target_rng, ls_pool):
    """§3.1 block A (+200): attr dose on the EXTENDED pools; same skeleton
    as v8's make_attr_rows (which is replayed untouched for v8's 120 rows)."""
    rows = []
    for cat, fam, quota in ATTR_PLAN_V9:
        for lang, n in quota.items():
            cands = []
            if fam == 'pet_name':
                t = PET_NAME_T_V9[lang]
                for kt in t['known']:
                    for name in t['names']:
                        k = kt.format(N=name)
                        for nt in t['new']:
                            for color in t['colors']:
                                cands.append((k, nt.format(C=color)))
            elif fam == 'pet_benign':
                attrs = PET_BENIGN_ATTR_V9[lang]
                for h in PET_HC:
                    if h['lang'] != lang:
                        continue
                    sp = PET_HC_SPECIES[(h['key'], lang)]
                    for k in h['known']:
                        for new in attrs[sp]:
                            cands.append((k, new))
            else:
                bgs = DOCTOR_BG_V9[lang]
                for h in DOCTOR_HC:
                    if h['lang'] != lang:
                        continue
                    for k in h['known']:
                        for new in bgs[h['key']]:
                            cands.append((k, new))
            rows += _emit(rng, cands, n, cat, fam, 'compat', lang, seen,
                          label_rng, target_rng, ls_pool, 'attr_consistent')
    # §3.1 重点: B2-isomorphic (「…的猫叫N」/ named / called / goes by) known
    # share over pet_name rows must be >= 1/3
    pn = [r for r in rows if r['_meta']['family'] == 'pet_name']
    def _iso(known, lang):
        if lang == ZH:
            return '的猫叫' in known
        return ('is named' in known) or ('goes by' in known) or ('is called' in known)
    iso = [r for r in pn if _iso(json.loads(r['state'])['known'], r['_meta']['lang'])]
    assert len(iso) >= len(pn) / 3, (len(iso), len(pn))
    print(f'block A: pet_name B2-isomorphic knowns {len(iso)}/{len(pn)} '
          f'(>= 1/3 asserted); exact 「用户的猫叫{{N}}」 rows '
          f'{sum(1 for r in pn if json.loads(r["state"])["known"].startswith("用户的猫叫"))}')
    return rows


def make_attr_contra_rows(rng, seen, label_rng, target_rng, ls_pool):
    """§3.2 block B (+40, kind='true'): same-subject attribute CONTRADICTION.
    appearance x20 + name x20 (zh 12 / en 8 each); pools exclude 小白/小黑."""
    rows = []
    for sub, quota in (('appearance', {ZH: 12, EN: 8}), ('name', {ZH: 12, EN: 8})):
        for lang, n in quota.items():
            t = PET_NAME_T_V9[lang]
            cands = []
            if sub == 'appearance':
                tpl = t['new'][0]  # 用户的猫是只{C}猫 / The user's cat is a {C} cat
                for c1 in t['colors']:
                    for c2 in t['colors']:
                        if c1 != c2:
                            cands.append((tpl.format(C=c1), tpl.format(C=c2)))
            else:
                kt = '用户的猫叫{N}' if lang == ZH else "The user's cat is named {N}"
                for n1 in t['names']:
                    for n2 in t['names']:
                        if n1 != n2:
                            cands.append((kt.format(N=n1), kt.format(N=n2)))
            rows += _emit(rng, cands, n, 'pet_attr', 'attr_contradiction', 'true', lang,
                          seen, label_rng, target_rng, ls_pool, 'attr_contradiction',
                          extra={'contra': sub})
    return rows


# --- §3.3 block C: waver (in-dist cats only; holdout cats zero rows) ----------
WAVER_PHRASES = {
    ZH: ['用户在想要不要{X}', '用户有点想试试{X}', '用户最近在琢磨要不要{X}',
         '用户纠结了一阵要不要{X}', '用户偶尔会想{X}', '用户今年一直有点想{X}',
         '用户跟朋友聊过想{X}', '用户犹豫着要不要{X}'],
    EN: ['The user is thinking about whether to {X}', 'The user is considering whether to {X}',
         'The user keeps wondering whether to {X}', 'The user has been tempted to {X}',
         'The user sometimes thinks about whether to {X}', 'The user brought up whether to {X} with friends'],
}
WAVER_ACTION = {
    ('city', ZH): ['搬到{B}发展'], ('city', EN): ['move to {B}', 'relocate to {B}'],
    ('car', ZH): ['把车换成{B}'], ('car', EN): ['trade the car in for {B}'],
    ('job', ZH): ['转岗去做{B}'], ('job', EN): ['move into a {B} role'],
    ('coffee', ZH): ['改喝{B}'], ('coffee', EN): ['switch to {B}', 'give {B} a try'],
    ('lang', ZH): ['把主力语言换成{B}'], ('lang', EN): ['switch the main language to {B}'],
    ('team', ZH): ['转投{B}门下'], ('team', EN): ['start following {B}'],
    ('gym_brand', ZH): ['换成去{B}'], ('gym_brand', EN): [],
    ('cloud', ZH): ['把项目迁到{B}'], ('cloud', EN): ['migrate the project to {B}'],
}
# gym_brand is ZH-only in the v3 set (v7 alt_praise convention) -> EN cats = 7
WAVER_PLAN = {'city': {ZH: 9, EN: 7}, 'car': {ZH: 9, EN: 7}, 'job': {ZH: 9, EN: 7},
              'coffee': {ZH: 9, EN: 7}, 'lang': {ZH: 9, EN: 7}, 'team': {ZH: 9, EN: 7},
              'cloud': {ZH: 9, EN: 6}, 'gym_brand': {ZH: 9}}  # zh 72 / en 48 = +120


def make_waver_rows(rng, cats, seen, label_rng, target_rng, ls_pool):
    rows = []
    for fam, quota in WAVER_PLAN.items():
        for lang, n in quota.items():
            c = next(cc for cc in cats if cc['key'] == fam and cc['lang'] == lang)
            cands = []
            for kt in c['known']:
                for a in c['vals']:
                    known = kt.format(A=a)
                    for b in c['vals']:
                        if b == a:
                            continue
                        for act in WAVER_ACTION[(fam, lang)]:
                            x = act.format(B=b)
                            for ph in WAVER_PHRASES[lang]:
                                cands.append((known, ph.format(X=x)))
            rows += _emit(rng, cands, n, fam, 'waver', 'consid', lang, seen,
                          label_rng, target_rng, ls_pool, 'waver_consider')
    return rows


# --- §3.4 block D: change already happened (in-dist cats, pairs with C) -------
CHANGE_T = {
    ('city', ZH): ['用户已经搬到{B}了', '用户搬去了{B},不住{A}了'],
    ('city', EN): ['The user has already moved to {B}', 'The user moved to {B} and no longer lives in {A}'],
    ('car', ZH): ['用户把{A}换成了{B}', '用户不再开{A},换{B}了'],
    ('car', EN): ['The user traded {A} in for {B}', 'The user no longer drives {A}; they drive {B} now'],
    ('job', ZH): ['用户已经转岗做{B}了,不做{A}了', '用户跳槽去做{B}了'],
    ('job', EN): ['The user has already moved into a {B} role', 'The user switched jobs and is now a {B}'],
    ('coffee', ZH): ['用户现在改喝{B}了', '用户把常喝的{A}换成了{B}'],
    ('coffee', EN): ['The user has switched to {B} now', 'The user replaced their usual {A} with {B}'],
    ('lang', ZH): ['用户把主力语言从{A}换成了{B}', '用户现在主力写{B}了'],
    ('lang', EN): ['The user switched the main language from {A} to {B}', 'The user now mainly codes in {B}'],
    ('team', ZH): ['用户已经转投{B}门下了', '用户不再看{A}的比赛,改看{B}了'],
    ('team', EN): ['The user has already started following {B}', 'The user stopped watching {A} and follows {B} now'],
    ('gym_brand', ZH): ['用户把健身卡从{A}换到了{B}', '用户现在常去{B}了'],
    ('cloud', ZH): ['用户已经把项目迁到{B}了', '用户把项目从{A}迁移到了{B}'],
    ('cloud', EN): ['The user has already migrated the project to {B}', 'The user moved the project from {A} to {B}'],
}
# zh 24 (7 bilingual cats x3 + gym 3) / en 16 (city,job x3; others x2)
CHANGE_PLAN = {'city': {ZH: 3, EN: 3}, 'job': {ZH: 3, EN: 3}, 'car': {ZH: 3, EN: 2},
               'coffee': {ZH: 3, EN: 2}, 'lang': {ZH: 3, EN: 2}, 'team': {ZH: 3, EN: 2},
               'cloud': {ZH: 3, EN: 2}, 'gym_brand': {ZH: 3}}


def make_change_rows(rng, cats, seen, label_rng, target_rng, ls_pool):
    rows = []
    for fam, quota in CHANGE_PLAN.items():
        for lang, n in quota.items():
            c = next(cc for cc in cats if cc['key'] == fam and cc['lang'] == lang)
            cands = []
            for kt in c['known']:
                for a in c['vals']:
                    known = kt.format(A=a)
                    for b in c['vals']:
                        if b == a:
                            continue
                        for tpl in CHANGE_T[(fam, lang)]:
                            cands.append((known, tpl.format(A=a, B=b)))
            rows += _emit(rng, cands, n, fam, 'change_done', 'true', lang, seen,
                          label_rng, target_rng, ls_pool, 'change_done')
    return rows


def assert_newblock_field_no_leak(new_rows, tag):
    """§3.6-6 (new for v9: all four blocks), operationalized per V9 §2.2 口径注:
    (a) field-level equality vs the 35 acceptance cases + 18 diag pairs = 0
        (hard assert -- the real leak concern);
    (b) field-level equality vs val/val_soft: any hit must already exist as a
        field in the carried v8 corpus (家族级句式共享, 310 hits, accepted);
        the new blocks must ADD no new overlap (hard assert).  Residual hits
        are counted and reported, not asserted away."""
    from realtest_v2 import CASES, NEGATION_CASES
    from realtest_v4 import NEW_CASES
    from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS
    case_ks, case_ns = set(), set()
    for (state, _e), _note in (list(CASES) + list(NEGATION_CASES) + list(NEW_CASES)
                               + list(CONTROL_PAIRS) + list(SWAP_PAIRS)):
        case_ks.add(state['known'])
        case_ns.add(state['new'])
    val_ks = set()
    for p in ('nli_conflict_val.jsonl', 'nli_conflict_val_soft.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                st = json.loads(l)['state']
                st = json.loads(st) if isinstance(st, str) else st
                val_ks.add(st.get('known'))
                val_ks.add(st.get('new'))
    hard = []
    for r in new_rows:
        st = json.loads(r['state'])
        if st['known'] in case_ks or st['new'] in case_ns or st['known'] in case_ns or st['new'] in case_ks:
            hard.append((st['known'], st['new']))
    assert not hard, f'{tag} case/diag field-equality leak: {hard[:3]}'
    # (b) vs val/val_soft, minus fields the carried corpus already shares
    carried_fields = set()
    for l in open(os.path.join(OUT_DIR, 'nli_conflict_train_v8.jsonl'), encoding='utf-8'):
        if l.strip():
            st = json.loads(l)['state']
            st = json.loads(st) if isinstance(st, str) else st
            carried_fields.add(st.get('known'))
            carried_fields.add(st.get('new'))
    val_hits = 0
    for r in new_rows:
        st = json.loads(r['state'])
        for fld in (st['known'], st['new']):
            if fld in val_ks:
                val_hits += 1
                assert fld in carried_fields, f'{tag} NEW val/val_soft field overlap: {fld}'
    print(f'{tag}: case/diag field leak 0; val/val_soft field touches '
          f'{val_hits} (all pre-existing family-level sharing, delta-new = 0 asserted)')


HOLDOUT_CATS = {'email', 'desk_floor', 'degree', 'barber'}


def main():
    rng = random.Random(SEED_V9)
    label_rng = random.Random(SEED_V9 + 1)
    target_rng = random.Random(SEED_V9 + 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # acceptance sets must never leak into train (carried)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: same extraction as v4..v8 (exact state match), name-rotated too
    ls_pool = make_ls_pool(label_rng, 15820)
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

    # carried v6/v7/v8 blocks in v8's exact stream order (byte-identical content)
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

    # v8 attr block replayed byte-identically (v8 pools, v8 quotas, same
    # rng position), then the four v9 blocks consume the stream tail
    attr_rows = make_attr_rows(rng, seen, label_rng, target_rng, ls_pool)  # v8's +120
    block_a = make_attr_add_rows(rng, seen, label_rng, target_rng, ls_pool)      # +200
    block_b = make_attr_contra_rows(rng, seen, label_rng, target_rng, ls_pool)   # +40
    block_c = make_waver_rows(rng, train_pool, seen, label_rng, target_rng, ls_pool)  # +120
    block_d = make_change_rows(rng, train_pool, seen, label_rng, target_rng, ls_pool)  # +40
    new_rows = block_a + block_b + block_c + block_d

    v9_train = (mnli_rows + soft_rows + alt_praise_rows + metric_rows
                + hc_base_rows + pet_rows + doctor_rows
                + neg_rows + neg5_rows + neg_unrel_rows + cf_rows + cf_unrel_rows
                + attr_rows + new_rows)
    rng.shuffle(v9_train)

    # ---- §3.6 hard invariants -------------------------------------------------
    assert len(v9_train) == 15820, len(v9_train)
    n_true = sum(1 for r in v9_train if r['gold']['conflict']['label'] == 'true')
    n_false = len(v9_train) - n_true
    states = [r['state'] for r in v9_train]
    assert len(set(states)) == 15820, 'duplicate states'
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        other = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, p), encoding='utf-8') if l.strip()}
        assert set(states).isdisjoint(other), f'leak into {p}'

    kind_counts = Counter(r['_meta']['kind'] for r in v9_train)
    expect_kinds = {'contradiction': 2000, 'entailment/neutral': 4000, 'true': 1920, 'compat': 1800,
                    'consid': 1480, 'unrelated': 1120, 'hardconstraint': 580, 'hc_safe': 260,
                    'hc_unrelated': 160, 'neg_true': 568, 'neg_false': 404, 'neg_false_boundary': 228,
                    'neg_unrelated': 400, 'cf_true': 300, 'cf_false': 300, 'cf_unrelated': 300}
    assert dict(kind_counts) == expect_kinds, (kind_counts, expect_kinds)

    # §3.6-2/4: per-block counts, exact
    shape_counts = Counter(r['_meta'].get('shape') for r in v9_train if r['_meta'].get('shape'))
    expect_shapes = {'alt_praise': 240, 'metric_shift': 40, 'attr_consistent': 320,
                     'attr_contradiction': 40, 'waver_consider': 120, 'change_done': 40,
                     'pet': 80, 'never': 80, 'relapse': 80, 'dneg': 80, 'nocar': 80}
    assert dict(shape_counts) == expect_shapes, (shape_counts, expect_shapes)
    assert shape_counts['attr_consistent'] == 320  # v8 120 (byte-identical replay) + v9 200
    cells = Counter((r['_meta']['family'], r['_meta']['lang']) for r in v9_train
                    if r['_meta'].get('shape') in ('attr_consistent', 'attr_contradiction',
                                                   'waver_consider', 'change_done'))
    expect_cells = {}
    # attr_consistent cells = v8's replayed 120 + v9's 200
    expect_cells[('pet_name', ZH)] = 24 + 72
    expect_cells[('pet_name', EN)] = 16 + 48
    expect_cells[('pet_benign', ZH)] = 24 + 24
    expect_cells[('pet_benign', EN)] = 16 + 16
    expect_cells[('doctor_bg', ZH)] = 24 + 24
    expect_cells[('doctor_bg', EN)] = 16 + 16
    expect_cells[('attr_contradiction', ZH)] = 24
    expect_cells[('attr_contradiction', EN)] = 16
    for fam, quota in WAVER_PLAN.items():
        for lang, n in quota.items():
            expect_cells[('waver', lang)] = expect_cells.get(('waver', lang), 0) + n
    for fam, quota in CHANGE_PLAN.items():
        for lang, n in quota.items():
            expect_cells[('change_done', lang)] = expect_cells.get(('change_done', lang), 0) + n
    assert dict(cells) == expect_cells, (cells, expect_cells)
    # block B sub-shape split (appearance/name x20)
    contra = Counter((r['_meta']['contra'], r['_meta']['lang']) for r in v9_train
                     if r['_meta'].get('shape') == 'attr_contradiction')
    assert dict(contra) == {('appearance', ZH): 12, ('appearance', EN): 8,
                            ('name', ZH): 12, ('name', EN): 8}, contra

    label_mix = Counter(r['_meta']['labelset'] for r in v9_train)
    assert label_mix[5] == 3164 and label_mix[6] == 3164, label_mix
    assert sum(label_mix[s] for s in (1, 2, 3, 4)) == 15820 - 6328, label_mix
    for ls in range(1, 7):
        kinds_with = {r['_meta']['kind'] for r in v9_train if r['_meta']['labelset'] == ls}
        assert len(kinds_with) >= 3, (ls, len(kinds_with))

    for r in v9_train:
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

    # §3.6-7: forbidden literals / defiance markers on all four new blocks (=0)
    for r in new_rows:
        st = json.loads(r['state'])
        for w in ATTR_FORBIDDEN:
            assert w not in st['known'] and w not in st['new'], (w, st)

    # §3.6-6: new blocks share no known/new field with 35 cases / diag / val sets
    assert_newblock_field_no_leak(new_rows, 'v9 new blocks')

    # §3.6-8: holdout cats get ZERO new rows this round (scope = the four new
    # blocks; the carried corpus's 44 barber-topic neg_unrelated controls are
    # v6 delivered rows, untouched per §9, and stay in the report)
    holdout_hits = [r['_meta'] for r in new_rows
                    if r['_meta']['cat'] in HOLDOUT_CATS or r['_meta'].get('family') in HOLDOUT_CATS]
    assert not holdout_hits, f'holdout-cat rows in new blocks: {holdout_hits[:2]}'
    carried_holdout = sum(1 for r in v9_train
                          if r['_meta']['cat'] in HOLDOUT_CATS or r['_meta'].get('family') in HOLDOUT_CATS)
    print(f'holdout-cat check: new blocks 0 rows (asserted); carried corpus '
          f'{carried_holdout} barber-topic neg_unrelated control rows (v6 delivered, untouched)')

    # carried upgrade: whole-train field-level vs 35 cases + 18 diag pairs
    assert_no_case_leak_v8(v9_train)

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v9.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v9_train:
            meta = r.pop('_meta')
            meta['name_a'] = r['questions']['conflict']['labels']['false']
            meta['name_b'] = r['questions']['conflict']['labels']['true']
            r['meta'] = meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v9_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false / n_true:.2f})')
    print('meta.kind:', dict(sorted(Counter(r["meta"]["kind"] for r in v9_train).items())))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v9_train)))
    print('meta.labelset:', dict(sorted(Counter(r["meta"]["labelset"] for r in v9_train).items())))
    sem = sum(label_mix[s] for s in (1, 2, 3, 4))
    print(f'labelset ratio: semantic {sem / 15820:.1%}, LS5 {label_mix[5] / 15820:.1%}, LS6 {label_mix[6] / 15820:.1%}')
    print('new blocks per shape:', dict(sorted(shape_counts.items())))
    print('waver per cat:', {f'{k}/{l}': n for (k, l), n in sorted(cells.items()) if k == 'waver'})
    print('change per cat-lang:', {f'{f}/{l}': n for (f, l), n in sorted(cells.items()) if f in CHANGE_PLAN})
    print('-- new-block sample rows:')
    shown = {'pet_name': 0, 'attr_contradiction': 0, 'waver': 0, 'change_done': 0}
    for r in new_rows:
        st = json.loads(r['state'])
        fam = r['meta']['family']
        key = fam if fam in shown else ('attr_contradiction' if fam == 'attr_contradiction' else None)
        if key and shown[key] < 2:
            p = r['gold']['conflict']['probabilities']
            print(f'   [{r["meta"]["shape"]}|{r["meta"]["lang"]}] known={st["known"][:42]} | '
                  f'new={st["new"][:42]} | p(gold)={p[r["gold"]["conflict"]["label"]]}')
            shown[key] += 1


if __name__ == '__main__':
    main()
