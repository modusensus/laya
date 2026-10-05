# -*- coding: utf-8 -*-
"""v12 data arm (HANDOFF_NLI_V12.md §2.1): v11 corpus CARRIED WHOLE (15914
rows, byte-identical lines) + the single preregistered lever at the stream
tail:

  Lever (§2.1, exactly 36 rows per §4-2): negation-known CONFLICT shape,
  6 domains x 6 rows (zh 4 + en 2): known carries an explicit negation
  (「没有/从不/从未/不」系), new provides concrete info violating it, gold=true.
  Domains (all non-pet per §2.1): subscription / diet / commute / property /
  schedule / device.  The two chronic acceptance texts (「用户没有养宠物…」
  「用户没有订阅任何视频会员…」) and every 35-case / 18-diag-pair / family-9 /
  negfam-8 field stay out by assertion.  Dose-evidence is honestly mixed
  (§2.1 note) -- the protocol, not the lever, is the judgement.

  Template disjointness (review note, stricter than the preregistered floor):
  lever templates deliberately avoid the negfam 8-case surface templates, so a
  negfam pass cannot be template recall of the lever itself -- the instrument
  then measures shape generalization, not template matching.

v12 train = 15914 carried + 36 new = 15950 rows.  val (1000) / val_soft (300)
NOT regenerated.  New rows consume a FRESH seeded stream (v11 precedent).

Authenticity chain (self-contained, no network): the 15914 carried lines are
reused VERBATIM; asserted that exactly 15820 of them are byte-identical lines
of the v9 file (block-B lineage) and the remaining 94 are the v11 lever rows.

Deterministic: SEED_V12 = 20261006.
"""
import json
import os
import random
import re
from collections import Counter

from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import OUT_DIR
from gen_soft_conflicts_v6 import question_for, make_ls_pool, next_ls, TARGETS, gold_for
from gen_soft_conflicts_v8 import ATTR_FORBIDDEN, assert_no_case_leak_v8
from gen_soft_conflicts_v9 import HOLDOUT_CATS, assert_newblock_field_no_leak
from family_diag import FAMILY_CASES
from negfam_diag import NEGFAM_CASES

SEED_V12 = 20261006

# §2.1 lever: (known, [news]) per language; zh 2 knowns x 2 news = 4 rows,
# en 1 known x 2 news = 2 rows per domain => 6 x 6 = 36.
NEG_LEVER = {
    'subscription': {
        ZH: [('用户从未开通过任何付费会员', ['用户昨天续费了视频平台的年卡', '用户刚开通了音乐会员的连续包月']),
             ('用户名下没有任何订阅服务', ['用户昨天续费了视频平台的年卡', '用户刚开通了音乐会员的连续包月'])],
        EN: [("The user has never paid for any membership",
              ['The user renewed a video platform annual plan yesterday', 'The user just signed up for a monthly music membership'])],
    },
    'diet': {
        ZH: [('用户不吃乳制品', ['用户中午喝了一大杯拿铁', '用户昨天下午茶点了芝士蛋糕']),
             ('用户三年前起就不吃肉', ['用户昨天在烤肉店吃了牛排', '用户中午点了份红烧肉'])],
        EN: [("The user avoids dairy",
              ['The user had a large latte at noon', 'The user ordered cheesecake for dessert yesterday'])],
    },
    'commute': {
        ZH: [('用户从不自驾通勤', ['用户今天开车去了公司', '用户昨天开车去见客户了']),
             ('用户没有买摩托车', ['用户上周提了一辆摩托', '用户刚给新摩托上了牌照'])],
        EN: [("The user never bikes to the office",
              ['The user cycled to the office this morning', 'The user just bought a commuter bike'])],
    },
    'property': {
        ZH: [('用户名下没有任何房产', ['用户上周签了一套小公寓的合同', '用户刚交了一套LOFT的首付']),
             ('用户从来不向别人借钱', ['用户昨天借了三千块给同事', '用户上周借给亲戚两万块'])],
        EN: [("The user doesn't own a car",
              ['The user registered a used car last week', 'The user just paid a deposit on a vehicle'])],
    },
    'schedule': {
        ZH: [('用户周末从不处理工作消息', ['用户周六上午开了三小时的项目会', '用户周日晚上还在改方案']),
             ('用户晚上十点后不安排任何日程', ['用户昨晚十一点见了客户', '用户把体检约在了周三晚上十点半'])],
        EN: [("The user never schedules work on weekends",
              ['The user joined a work call this Saturday', 'The user spent Sunday afternoon revising the deck'])],
    },
    'device': {
        ZH: [('用户没有安装任何手游', ['用户昨晚下载了两款新手游', '用户刚装上了市榜第一的手游']),
             ('用户家里没有打印机', ['用户今天装好了一台新打印机', '用户上周网购了一台激光打印机'])],
        EN: [("The user's phone has no games installed",
              ['The user downloaded two games last night', 'The user just installed a puzzle game'])],
    },
}

ZH_NEG_MARK = re.compile(r'没有|从不|从未|不|未')
EN_NEG_MARK = re.compile(r"never|not|n't|no |without|avoid", re.IGNORECASE)


def _emit_v12(rng, cands, n, cat, fam, kind, lang, seen, label_rng, target_rng, ls_pool, shape):
    """v11's _emit shape: walk shuffled cands, dedupe via seen."""
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
        rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                     'gold': {'conflict': gold_for(kind, target_rng)},
                     '_meta': {'cat': 'negation', 'kind': kind, 'lang': lang, 'labelset': ls,
                               'shape': shape, 'family': fam}})
        made += 1
    assert made == n, (fam, lang, made, n)
    return rows


def make_neg_lever_rows(rng, seen, label_rng, target_rng, ls_pool):
    rows = []
    for fam, spec in NEG_LEVER.items():
        for lang, plan_n in ((ZH, 4), (EN, 2)):
            cands = []
            for k, news in spec[lang]:
                for new in news:
                    cands.append((k, new))
            rows += _emit_v12(rng, cands, plan_n, 'negation', fam, 'true', lang, seen,
                              label_rng, target_rng, ls_pool, 'neg_known_conflict')
    assert len(rows) == 36, len(rows)
    for r in rows:
        st = json.loads(r['state'])
        m = ZH_NEG_MARK if r['_meta']['lang'] == ZH else EN_NEG_MARK
        assert m.search(st['known']) and st['known'] != st['new'], st
        assert r['gold']['conflict']['label'] == 'true'
    return rows


def main():
    rng = random.Random(SEED_V12)
    label_rng = random.Random(SEED_V12 + 1)
    target_rng = random.Random(SEED_V12 + 2)

    # ---- carried v11 corpus, VERBATIM lines + lineage check --------------------
    v11_path = os.path.join(OUT_DIR, 'nli_conflict_train_v11.jsonl')
    carried_lines = [l.rstrip('\n') for l in open(v11_path, encoding='utf-8') if l.strip()]
    assert len(carried_lines) == 15914, len(carried_lines)
    v9_lines = {l.rstrip('\n') for l in open(os.path.join(OUT_DIR, 'nli_conflict_train_v9.jsonl'),
                                             encoding='utf-8') if l.strip()}
    inherited = [l for l in carried_lines if l in v9_lines]
    added = [l for l in carried_lines if l not in v9_lines]
    added_shapes = Counter(json.loads(l)['meta'].get('shape') for l in added)
    assert len(inherited) == 15820, len(inherited)
    assert len(added) == 94 and added_shapes == Counter({'attr_consistent': 34,
                                                         'unrel_supplement': 30,
                                                         'temp_preference': 30}), added_shapes
    print(f'v11 file lineage: 15820 lines byte-identical to v9 file + 94 v11 lever rows '
          f'({dict(added_shapes)})')

    # seen: everything new rows must avoid (train carried + frozen eval sets)
    v11_states = {json.loads(l)['state'] for l in carried_lines}
    seen = set(v11_states)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # ---- lever consumes a fresh stream tail -------------------------------------
    ls_pool = make_ls_pool(label_rng, 100)
    lever = make_neg_lever_rows(rng, seen, label_rng, target_rng, ls_pool)

    # ---- hard invariants on the new rows ----------------------------------------
    # 0/51 hygiene floor: 35 cases + 18 diag pairs (field-level) + family 9 + negfam 8
    assert_newblock_field_no_leak(lever, 'v12 neg lever')
    for tag, CASES in (('family9', FAMILY_CASES), ('negfam8', NEGFAM_CASES)):
        inst_fields = set()
        for c in CASES:
            st = c[0][0]
            inst_fields.add(st['known'])
            inst_fields.add(st['new'])
        for r in lever:
            s = json.loads(r['state'])
            assert s['known'] not in inst_fields and s['new'] not in inst_fields, (tag, s)
    # negfam/lever template disjointness: no shared known or new surface either way
    lever_fields = {json.loads(r['state'])['known'] for r in lever} | {json.loads(r['state'])['new'] for r in lever}
    negfam_fields = {c[0][0]['known'] for c in NEGFAM_CASES} | {c[0][0]['new'] for c in NEGFAM_CASES}
    assert not (lever_fields & negfam_fields), 'lever/negfam surface overlap'
    holdout_hits = [r['_meta'] for r in lever
                    if r['_meta']['cat'] in HOLDOUT_CATS or r['_meta'].get('family') in HOLDOUT_CATS]
    assert not holdout_hits, f'holdout-cat rows in v12 lever: {holdout_hits[:2]}'
    for r in lever:
        st = json.loads(r['state'])
        for w in ATTR_FORBIDDEN:
            assert w not in st['known'] and w not in st['new'], (w, st)
        g = r['gold']['conflict']
        base, j = TARGETS[r['_meta']['kind']]
        pg = g['probabilities'][g['label']]
        assert abs(pg - 0.5) >= 0.2 and abs(pg - base) <= j + 1e-9, (r['_meta']['kind'], pg)

    new_kinds = Counter(r['_meta']['kind'] for r in lever)
    new_fams = Counter(r['_meta']['family'] for r in lever)
    new_langs = Counter(r['_meta']['lang'] for r in lever)

    # ---- assemble: carried lines VERBATIM + new rows serialized v10-style ------
    new_lines = []
    for r in lever:
        meta = r.pop('_meta')
        meta['name_a'] = r['questions']['conflict']['labels']['false']
        meta['name_b'] = r['questions']['conflict']['labels']['true']
        r['meta'] = meta
        new_lines.append(json.dumps(r, ensure_ascii=False))
    entries = carried_lines + new_lines
    rng.shuffle(entries)
    assert len(entries) == 15950, len(entries)
    assert len({json.loads(e)['state'] for e in entries}) == 15950, 'duplicate states'

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v12.jsonl')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for e in entries:
            f.write(e + '\n')

    # carried lines survived the round-trip byte-identically
    out_lines = {l.rstrip('\n') for l in open(path, encoding='utf-8') if l.strip()}
    assert len(out_lines & set(carried_lines)) == 15914, 'carried lines drifted'
    n_true = sum(1 for e in entries if json.loads(e)['gold']['conflict']['label'] == 'true')
    print(f'{path}: {len(entries)} rows (15914 carried verbatim + 36 neg-lever)')
    print(f'gold: true {n_true} / false {len(entries) - n_true}')
    print('lever kinds:', dict(new_kinds), '| langs:', {str(k): v for k, v in new_langs.items()})
    print('lever families:', dict(sorted(new_fams.items())))
    print('-- sample lever rows:')
    for fam in ('subscription', 'diet', 'commute'):
        r = next(x for x in lever if x['meta']['family'] == fam)
        st = json.loads(r['state'])
        print(f'   [{r["meta"]["family"]}|{r["meta"]["lang"]}] known={st["known"]} | new={st["new"]}')


if __name__ == '__main__':
    main()
