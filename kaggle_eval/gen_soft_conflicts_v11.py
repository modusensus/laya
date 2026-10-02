# -*- coding: utf-8 -*-
"""v11 data arm (HANDOFF_NLI_V11.md §2): v9 corpus CARRIED WHOLE (block B
stays -- v10 proved it is the load-bearing wall) + two evidence-driven
levers at the stream tail:

  Lever 1 (§2.1): pet_name exact-shape top-up, zh +34 rows: the B2-isomorphic
  「用户的猫叫{N}」 + 「用户的猫是只{C}猫」 pair raised to >=40.  Counting note:
  the handoff quoted "v9 仅 14 行" -- that is the KNOWN-template count; the
  strict PAIR shape measures 6 in the v9 file.  Dosed to the preregistered
  target on the strict-pair definition (6 carried + 34 new = 40).
  Fresh literals come from the v9 pools EXTENDED (names +3, colors +2 --
  v9's own extension precedent, never copied rows).  kind='compat',
  shape='attr_consistent', family='pet_name'.

  Lever 1b (§2.1, backtest-driven): the two old-20 cases that repeat-missed
  in 2/3 same-config runs (「无关补充」「临时的不违背偏好」 -- protocol_verdict.py
  138c56c; v9 真实弱点, not run variance) each get 30 shape-matched,
  fresh-literal control rows:
    - unrel_supplement (zh 18 / en 12, kind='compat'): stable fact/preference
      known, new = a one-off activity/detail that does NOT contradict it,
      including the missed same-category-different-topic shape (学X / 今天学Y)
      => false.
    - temp_preference (zh 18 / en 12, kind='consid'): stored stable preference
      known, new = CONSIDERING a one-off surface-conflicting activity (not a
      preference switch, not a hard-constraint violation) => false.
      Complements v9's waver_consider (preference-switch wavering) with the
      event-attendance sub-shape the acceptance case uses.

v11 train = 15820 carried + 94 new = 15914 rows.  Carried rows are loaded
VERBATIM from the v9 file (equality by construction; authenticity
cross-checked against the Kaggle v15 corpus = v9 minus exactly block B's 40
rows).  New rows consume a FRESH seeded stream (SEED_V11; the v10-style
mid-stream drop that forced full replay does not apply this round).
val (1000) / val_soft (300) NOT regenerated.

Family instrument (§2.2, eval-side, never in train): the 8 new isomorphic
cases live in family_diag.py; this gen asserts zero field overlap between
the corpus and those 8 cases on top of the 35+18 leak discipline (0/43).

Deterministic: SEED_V11 = 20261003.
"""
import json
import os
import random
from collections import Counter

from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import OUT_DIR
from gen_soft_conflicts_v6 import question_for, make_ls_pool, next_ls, TARGETS, gold_for
from gen_soft_conflicts_v8 import ATTR_FORBIDDEN, assert_no_case_leak_v8
from gen_soft_conflicts_v9 import PET_NAME_T_V9, HOLDOUT_CATS, assert_newblock_field_no_leak
from family_diag import FAMILY_CASES

SEED_V11 = 20261003

# Lever 1: v9 pools EXTENDED (names +3, colors +2; fresh combos only)
PET_NAMES_V11 = PET_NAME_T_V9[ZH]['names'] + ['团子', '芝麻', '雪碧']
PET_COLORS_V11 = PET_NAME_T_V9[ZH]['colors'] + ['虎斑', '银渐层']

# Lever 1b-a: unrelated supplement (fresh fields only -- the acceptance
# cases' own fields are forbidden; shaped on 学X / 今天学Y + cross-topic extras)
UNREL_SUPPLEMENT = {
    ZH: [
        ('用户在学钢琴', ['用户今天学了乐理', '用户今天练了半小时音阶', '用户昨天背了一首曲子的谱子']),
        ('用户在准备马拉松', ['用户今天研究了碳水补给', '用户中午吃了意面', '用户昨天买了一双新袜子']),
        ('用户在做开源项目', ['用户今天给文档补了个目录', '用户中午点了份炒饭', '用户昨晚整理了收藏夹']),
        ('用户在学摄影', ['用户今天看了构图教程', '用户中午吃了碗牛肉面', '用户昨天擦了擦镜头']),
        ('用户在准备考驾照', ['用户今天背了标志题', '用户中午吃了份饺子', '用户昨天把自行车打了个气']),
        ('用户在学法语', ['用户今天背了二十个单词', '用户中午吃了份沙拉', '用户昨天给绿萝浇了水']),
    ],
    EN: [
        ('The user is learning the piano', ['The user studied some music theory today', 'The user had pasta for lunch',
                                            'The user practiced scales for half an hour yesterday']),
        ('The user is training for a marathon', ['The user looked up carb loading today', 'The user ate a sandwich at noon',
                                                 'The user bought new running socks yesterday']),
        ('The user is working on an open-source project', ['The user added a table of contents to the docs today',
                                                           'The user had fried rice for lunch',
                                                           'The user cleaned up their bookmarks last night']),
        ('The user is learning photography', ['The user watched a composition tutorial today', 'The user had noodles for lunch',
                                              'The user wiped their lens clean yesterday']),
        ('The user is preparing for the driving test', ['The user memorized some road-sign questions today',
                                                        'The user had dumplings for lunch',
                                                        'The user pumped up their bicycle tire yesterday']),
    ],
}

# Lever 1b-b: temporary preference (one-off consideration, surface-conflicting)
TEMP_PREFERENCE = {
    ZH: [
        ('用户平时喜欢安静', ['用户在考虑这周末去逛逛夜市', '用户在纠结要不要去看一场露天演唱会',
                          '用户在想这周末要不要去趟游乐场', '用户在想要不要去参加一次音乐节',
                          '用户在纠结要不要去听一场 livehouse']),
        ('用户习惯早睡', ['用户在想要不要熬夜看一次发布会', '用户在纠结周末要不要通宵看一场球赛',
                      '用户在想跨年夜要不要熬到零点']),
        ('用户口味偏清淡', ['用户在想要不要跟同事去吃一顿重辣火锅', '用户在纠结周末要不要尝一次麻辣香锅',
                        '用户在想聚餐要不要点一份水煮鱼']),
        ('用户平时很节省', ['用户在考虑要不要花一天工资看场演出', '用户在纠结要不要给游戏氪一次金',
                        '用户在想这周要不要去自助餐厅吃一顿']),
        ('用户喜欢宅在家里', ['用户在想要不要这周末去短途徒步', '用户在纠结要不要跟团出去玩两天',
                          '用户在想五一要不要去邻市逛一天']),
        ('用户平时很少吃甜食', ['用户在想要不要尝一块朋友的生日蛋糕', '用户在纠结要不要点一份下午茶',
                            '用户在想周末要不要去尝一下新开的甜品店']),
    ],
    EN: [
        ('The user usually likes quiet places', ['The user is considering checking out a loud street fair this weekend',
                                                 'The user is thinking about going to an open-air concert',
                                                 'The user is considering a night at a karaoke box with friends']),
        ('The user always goes to bed early', ['The user is considering staying up for one product launch',
                                               'The user is thinking about pulling an all-nighter for a match',
                                               'The user is considering a midnight movie premiere once']),
        ('The user prefers mild food', ['The user is considering joining coworkers for extra-spicy hotpot once',
                                        'The user is thinking about trying spicy stir-fry this weekend',
                                        'The user is considering a one-off buffalo-wings night']),
        ('The user usually saves money', ["The user is considering spending a day's wage on one show",
                                          'The user is thinking about splurging on one game skin',
                                          'The user is considering one expensive gadget this year']),
        ('The user prefers staying home', ['The user is considering a short weekend hike',
                                           'The user is thinking about joining a two-day group tour',
                                           'The user is considering a one-day trip to the neighboring city']),
        ('The user rarely eats sweets', ['The user is considering a slice of birthday cake at a party',
                                         'The user is thinking about ordering afternoon tea once',
                                         'The user is considering dessert at the new bakery this weekend']),
    ],
}

LEVER1_N = 34          # 6 carried strict-pair (handoff's 14 = known-template count) + 34 = 40 (§2.1 >=40)
UNREL_PLAN = {ZH: 18, EN: 12}
TEMP_PLAN = {ZH: 18, EN: 12}


def _emit_v11(rng, cands, n, cat, fam, kind, lang, seen, label_rng, target_rng, ls_pool, shape):
    """v9's _emit, imported shape: walk shuffled cands, dedupe via seen."""
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
                     '_meta': {'cat': cat, 'kind': kind, 'lang': lang, 'labelset': ls,
                               'shape': shape, 'family': fam}})
        made += 1
    assert made == n, (fam, lang, made, n)
    return rows


def make_lever1_rows(rng, seen, label_rng, target_rng, ls_pool):
    kt, nt = '用户的猫叫{N}', '用户的猫是只{C}猫'
    cands = []
    for name in PET_NAMES_V11:
        for color in PET_COLORS_V11:
            cands.append((kt.format(N=name), nt.format(C=color)))
    rows = _emit_v11(rng, cands, LEVER1_N, 'pet_attr', 'pet_name', 'compat', ZH, seen,
                     label_rng, target_rng, ls_pool, 'attr_consistent')
    for r in rows:
        st = json.loads(r['state'])
        assert st['known'].startswith('用户的猫叫') and st['new'].startswith('用户的猫是只'), st
    return rows


def _pair_rows(spec, rng, seen, label_rng, target_rng, ls_pool, plan, cat, fam, kind, shape):
    rows = []
    for lang, n in plan.items():
        cands = []
        for k, news in spec[lang]:
            for new in news:
                cands.append((k, new))
        rows += _emit_v11(rng, cands, n, cat, fam, kind, lang, seen, label_rng,
                          target_rng, ls_pool, shape)
    return rows


def main():
    rng = random.Random(SEED_V11)
    label_rng = random.Random(SEED_V11 + 1)
    target_rng = random.Random(SEED_V11 + 2)

    # ---- carried v9 corpus, VERBATIM ------------------------------------------
    v9_path = os.path.join(OUT_DIR, 'nli_conflict_train_v9.jsonl')
    carried_lines = [l.rstrip('\n') for l in open(v9_path, encoding='utf-8') if l.strip()]
    v9_rows = [json.loads(l) for l in carried_lines]
    assert len(v9_rows) == 15820, len(v9_rows)
    v9_states = {r['state'] for r in v9_rows}

    # authenticity cross-check: Kaggle v15 == v9 minus exactly block B (40 rows)
    k15 = r'D:\kaggle_pull\nli-conflict-pairs-v15\nli_conflict_train.jsonl'
    s15 = {json.loads(l)['state'] for l in open(k15, encoding='utf-8') if l.strip()}
    only9 = v9_states - s15
    assert not (s15 - v9_states), 'v15 has states absent from the v9 file'
    only9_shapes = Counter(r['meta'].get('shape') for r in v9_rows if r['state'] in only9)
    assert len(only9) == 40 and set(only9_shapes) == {'attr_contradiction'}, (len(only9), only9_shapes)
    print(f'v9 file cross-check vs Kaggle v15: diff = exactly {len(only9)} attr_contradiction rows')

    # seen: everything new rows must avoid (train carried + frozen eval sets)
    seen = set(v9_states)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # ---- levers consume a fresh stream tail ------------------------------------
    ls_pool = make_ls_pool(label_rng, 100)   # sized up; only new rows consume
    lever1 = make_lever1_rows(rng, seen, label_rng, target_rng, ls_pool)
    unrel = _pair_rows(UNREL_SUPPLEMENT, rng, seen, label_rng, target_rng, ls_pool,
                       UNREL_PLAN, 'unrel', 'unrel_supp', 'compat', 'unrel_supplement')
    temp = _pair_rows(TEMP_PREFERENCE, rng, seen, label_rng, target_rng, ls_pool,
                      TEMP_PLAN, 'temp', 'temp_pref', 'consid', 'temp_preference')
    new_rows = lever1 + unrel + temp
    assert len(new_rows) == LEVER1_N + sum(UNREL_PLAN.values()) + sum(TEMP_PLAN.values()) == 94

    # ---- hard invariants on the new rows ---------------------------------------
    # 0/43 hygiene: 35 cases + 18 diag pairs (field-level) + the 8 family cases
    assert_newblock_field_no_leak(new_rows, 'v11 new blocks')
    fam_fields = set()
    for (st, _e), _n, _b in FAMILY_CASES:
        fam_fields.add(st['known'])
        fam_fields.add(st['new'])
    for r in new_rows:
        s = json.loads(r['state'])
        assert s['known'] not in fam_fields and s['new'] not in fam_fields, s
    holdout_hits = [r['_meta'] for r in new_rows
                    if r['_meta']['cat'] in HOLDOUT_CATS or r['_meta'].get('family') in HOLDOUT_CATS]
    assert not holdout_hits, f'holdout-cat rows in v11 new blocks: {holdout_hits[:2]}'
    for r in new_rows:
        st = json.loads(r['state'])
        for w in ATTR_FORBIDDEN:
            assert w not in st['known'] and w not in st['new'], (w, st)
        g = r['gold']['conflict']
        base, j = TARGETS[r['_meta']['kind']]
        pg = g['probabilities'][g['label']]
        assert abs(pg - 0.5) >= 0.2 and abs(pg - base) <= j + 1e-9, (r['_meta']['kind'], pg)

    # §2.1 target: exact-shape pet_name rows >= 40 across carried + new
    def _exact(r):
        st = json.loads(r['state'])
        return st['known'].startswith('用户的猫叫') and '用户的猫是只' in st['new']
    carried_exact = sum(1 for r in v9_rows
                        if r['meta'].get('family') == 'pet_name' and _exact(r))
    assert carried_exact + len(lever1) >= 40, (carried_exact, len(lever1))
    print(f'lever 1: exact-shape rows carried {carried_exact} + new {len(lever1)} '
          f'= {carried_exact + len(lever1)} (>= 40 asserted)')

    new_kinds = Counter(r['_meta']['kind'] for r in new_rows)
    new_shapes_named = {k: sum(1 for r in new_rows if r['_meta']['shape'] == k)
                        for k in ('unrel_supplement', 'temp_preference')}
    new_ls = dict(sorted(Counter(r['_meta']['labelset'] for r in new_rows).items()))

    # ---- assemble: carried lines VERBATIM + new rows serialized v10-style ------
    new_lines = []
    for r in new_rows:
        meta = r.pop('_meta')
        meta['name_a'] = r['questions']['conflict']['labels']['false']
        meta['name_b'] = r['questions']['conflict']['labels']['true']
        r['meta'] = meta
        new_lines.append(json.dumps(r, ensure_ascii=False))
    entries = carried_lines + new_lines
    rng.shuffle(entries)
    assert len(entries) == 15914, len(entries)
    assert len({json.loads(e)['state'] for e in entries}) == 15914, 'duplicate states'

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v11.jsonl')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for e in entries:
            f.write(e + '\n')

    shapes = Counter(json.loads(e)['meta'].get('shape') for e in entries if json.loads(e)['meta'].get('shape'))
    n_true = sum(1 for e in entries if json.loads(e)['gold']['conflict']['label'] == 'true')
    print(f'{path}: {len(entries)} rows (15820 carried verbatim + 94 new)')
    print(f'gold: true {n_true} / false {len(entries) - n_true}')
    print('new-block kinds:', dict(new_kinds))
    print('new shapes:', new_shapes_named, '| attr_consistent total:', shapes.get('attr_consistent'))
    print('labelset of new rows:', new_ls)
    print('-- sample new rows:')
    for r in lever1[:1] + unrel[:1] + temp[:1]:
        st = json.loads(r['state'])
        print(f'   [{r["meta"]["shape"]}|{r["meta"]["lang"]}] '
              f'known={st["known"][:40]} | new={st["new"][:40]}')


if __name__ == '__main__':
    main()
