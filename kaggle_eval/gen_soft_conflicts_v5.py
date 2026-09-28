# -*- coding: utf-8 -*-
"""v5 data arm (HANDOFF_NLI_V5.md §3): name-system augmentation + negation/
counterfactual blocks + graded soft targets.

Changes vs gen_soft_conflicts_v4.py ("照着 v4 改"):

  1. NAME-SYSTEM AUGMENTATION (§3.1, mandatory): 6 labelsets now rotate over ALL
     training rows (incl. the 6000 MNLI rows and both new blocks) -- v4 rotated
     synthetic rows only and left MNLI on LS1, which left a semantic-shortcut
     path.  LS5 = neutral A/B (instructions stay semantic); LS6 = random 8-char
     no-vowel strings, byte-identical rule to option_polarity_check.py's probe
     (train/probe must share one construction).  Ratio (row-level): semantic
     4 sets >=50%, LS5 >=20%, LS6 >=20%.
     NOTE: this intentionally breaks v4's "MNLI rows byte-identical" invariant
     (training block is mutable; acceptance sets val/val_soft stay untouched).

  2. NEGATION-OPERATOR BLOCK (§3.2 A, 800 = 400 neg_true + 300 neg_false +
     100 neg_false_boundary): negated-known statements paired with conflicting
     acts (true) / avoiding acts (false) / double negations (false, >=1/4 of
     negatives -- the shape behind the 5th NEGATION_CASE).

  3. KNOWN-SIDE ONE-STEP PERTURBATION BLOCK (§3.2 B, 600 = 300 cf_true +
     300 cf_false): qualifiers/synonyms on known; surface-conflict negatives
     (safe/same-value new) to densify the boundary class.  Realtest SHAPES only
     -- verbatim realtest text never enters training (acceptance sets must stay
     unseen; see V5 §8).

  4. GRADED SOFT TARGETS (§3.3): p(gold) per kind (consid 0.80±0.05 is the main
     band-widener), row-level jitter, gold.label == argmax asserted, and
     abs(p(gold)-0.5) >= 0.2 so no row sits near the decision line.

  5. MNLI rows keep p(gold)=0.95 exactly (main val is built from the same
     distribution; per-round comparability).

v5 train = 6000 MNLI + 5400 synthetic + 600 hard-constraint + 800 negation
         + 600 perturbation = 13400 rows.  val (1000) / val_soft (300) are NOT
         regenerated (byte-identical, directly comparable).

Known cost, explicitly accepted (V5 §3.3): val/val_soft gold stays 0.95 while
v5 training targets are graded, so label-confidence consistency / ECE on those
sets is NOT a round-5 criterion (V5 §5.2 rewrite); they carry accuracy/error
count and polarity-flip judgements only.

Deterministic: seed 20260930 (same family as v4; blocks consume the stream in
fixed order).
"""
import json
import os
import random
from collections import Counter

from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import (SEED, OUT_DIR, LABEL_SETS, patch_categories,
                                   patch_hc, render_hc, render_soft, HC)

SEED_V5 = 20260930  # == SEED from v4; named locally for clarity

# --- 6 labelsets: v4's 4 semantic sets + LS5 neutral + LS6 random (§3.1) ------
LS5_INSTRUCTIONS = {
    ZH: '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
    EN: 'Does the new information (new) conflict with the known (old) memory? '
        'supersede = the new information contradicts the old memory and must replace it, '
        'keep = consistent or unrelated',
}
RAND_ALPHABET = 'BCDFGHJKLMNPQRSTVWXZ2456789'  # no vowels -> no readable words (probe-identical)


def random_label(rng):
    return ''.join(rng.choices(RAND_ALPHABET, k=8))


def question_for_v5(ls_id, lang, rng):
    """Build the question dict for labelset ls_id on a row of language `lang`.

    LS1-4: v4's paired instruction+label sets.  LS5: neutral A/B, instruction
    stays the language's semantic definition.  LS6: per-row random strings,
    same instruction policy.
    """
    if ls_id in (1, 2, 3, 4):
        for sid, ins, labels in LABEL_SETS:
            if sid == ls_id:
                return {'conflict': {'type': 'noul', 'instructions': ins, 'labels': dict(labels)}}
        raise ValueError(ls_id)
    ins = LS5_INSTRUCTIONS[lang]
    if ls_id == 5:
        labels = {'false': 'A', 'true': 'B'}
    elif ls_id == 6:
        labels = {'false': random_label(rng), 'true': random_label(rng)}
    else:
        raise ValueError(ls_id)
    return {'conflict': {'type': 'noul', 'instructions': ins, 'labels': labels}}


# --- graded soft targets (§3.3): kind -> (base p(gold), jitter) ---------------
TARGETS = {
    'contradiction': (0.95, 0.0), 'entailment/neutral': (0.95, 0.0),
    'true': (0.93, 0.02), 'compat': (0.93, 0.02), 'unrelated': (0.96, 0.01),
    'consid': (0.80, 0.05),
    'hardconstraint': (0.90, 0.03), 'hc_safe': (0.90, 0.03), 'hc_unrelated': (0.96, 0.01),
    'neg_true': (0.90, 0.03), 'neg_false': (0.88, 0.05), 'neg_false_boundary': (0.80, 0.05),
    'cf_true': (0.90, 0.03), 'cf_false': (0.88, 0.05),
}


GOLD_LABEL = {
    'contradiction': 'true', 'entailment/neutral': 'false',
    'true': 'true', 'compat': 'false', 'unrelated': 'false', 'consid': 'false',
    'hardconstraint': 'true', 'hc_safe': 'false', 'hc_unrelated': 'false',
    'neg_true': 'true', 'neg_false': 'false', 'neg_false_boundary': 'false',
    'cf_true': 'true', 'cf_false': 'false',
}


def gold_for(kind, target_rng):
    """p(gold) = probability mass ON THE GOLD LABEL (v4 golds are 0.95/0.05
    with the same convention); the label itself is fixed per kind."""
    base, j = TARGETS[kind]
    p = base if j == 0 else base + target_rng.uniform(-j, j)
    p = round(min(max(p, 0.7), 0.98), 4)
    assert abs(p - 0.5) >= 0.2, (kind, p)
    lab = GOLD_LABEL[kind]
    other = 'true' if lab == 'false' else 'false'
    return {'label': lab, 'probabilities': {lab: round(p, 4), other: round(1 - p, 4)}}


def cat_v5(key, vals, known, conflict, compat, consid, unrelated):
    """Shape category in the v3 `cat` interface (lang filled by caller)."""
    return dict(key=key, lang=None, vals=vals, known=known, conflict=conflict,
                compat=compat, consid=consid, unrelated=unrelated, holdout=False)


# --- block A: negation-operator families (§3.2 A) -----------------------------
# known forms carry a negated/abstaining statement; act = conflicting behavior
# (true); avoid = consistent avoidance (false); bnd = double negation (false).
NEG_FAMILIES = [
    dict(key='spicy', lang=ZH, A=['辣的', '麻辣香锅', '辣子鸡'],
         known=['用户不吃辣', '用户从不吃辣', '用户一点辣都不碰'],
         act=['用户昨天点了{A}', '用户午餐吃了{A}', '用户刚下单了一份{A}'],
         avoid=['用户昨天没吃辣', '用户点的是不辣的做法', '用户今天也没碰辣的'],
         known_b=['用户已经忌辣半年了', '用户早就不吃辣了'],
         bnd=['用户没有点{A}', '用户一直没再碰{A}']),
    dict(key='spicy', lang=EN, A=['the spicy version', 'extra chili', 'mala xiang guo'],
         known=['The user cannot eat spicy food', 'The user never eats chili',
                'The user strictly avoids spice'],
         act=['The user ordered {A} yesterday', 'The user had {A} for lunch', 'The user just ordered {A}'],
         avoid=['The user did not eat spicy food yesterday', 'The user ordered the mild version',
                'The user skipped chili again today'],
         known_b=['The user has been off spice for months', 'The user quit spicy food long ago'],
         bnd=['The user did not order {A}', 'The user has not touched {A} since']),
    dict(key='peanut', lang=ZH, A=['花生酱饼干', '加了花生的沙律', '花生碎冰淇淋'],
         known=['用户对花生过敏', '用户碰花生就起疹子', '用户花生重度过敏'],
         act=['用户昨天吃了{A}', '用户刚才点了{A}', '用户下午尝了{A}'],
         avoid=['用户昨天没碰花生', '用户点的是无花生版本', '用户今天也没吃坚果'],
         known_b=['用户早就查明确认花生过敏', '用户一直对花生过敏'],
         bnd=['用户没有点含花生的东西', '用户没让厨房加花生']),
    dict(key='peanut', lang=EN, A=['peanut butter cookies', 'a peanut-topped salad', 'peanut ice cream'],
         known=['The user has a severe peanut allergy', 'Peanuts give the user hives',
                'The user is highly allergic to peanuts'],
         act=['The user ate {A} yesterday', 'The user just ordered {A}', 'The user tried {A} this afternoon'],
         avoid=['The user avoided peanuts yesterday', 'The user ordered the peanut-free version',
                'The user has not touched nuts today either'],
         known_b=['The user has a confirmed peanut allergy on file', 'The user has always been allergic to peanuts'],
         bnd=['The user did not order anything with peanuts', 'The user asked the kitchen to skip peanuts']),
    dict(key='lactose', lang=ZH, A=['芝士蛋糕', '拿铁', '奶油浓汤'],
         known=['用户乳糖不耐受', '用户喝牛奶就拉肚子', '用户对乳糖不耐受'],
         act=['用户昨天吃了{A}', '用户下午点了{A}', '用户刚才喝了一份{A}'],
         avoid=['用户昨天没喝牛奶', '用户点的是燕麦奶基底', '用户今天也没碰奶制品'],
         known_b=['用户一直乳糖不耐受', '用户早就查出乳糖不耐受'],
         bnd=['用户没有点奶制的{A}', '用户没加牛奶']),
    dict(key='lactose', lang=EN, A=['cheesecake', 'a latte', 'cream soup'],
         known=['The user is lactose intolerant', 'Milk makes the user sick',
                'The user cannot digest lactose'],
         act=['The user had {A} yesterday', 'The user ordered {A} in the afternoon', 'The user just drank {A}'],
         avoid=['The user did not drink milk yesterday', 'The user ordered the oat-milk base',
                'The user skipped dairy today as well'],
         known_b=['The user has always been lactose intolerant', 'The user was diagnosed lactose intolerant'],
         bnd=['The user did not order the dairy {A}', 'The user skipped the milk']),
    dict(key='alcohol', lang=ZH, A=['一杯白酒', '啤酒', '鸡尾酒'],
         known=['用户对酒精过敏', '用户滴酒不能沾', '用户沾酒就脸红心跳快'],
         act=['用户昨晚喝了{A}', '用户刚才碰了{A}', '用户饭局上喝了{A}'],
         avoid=['用户昨晚没喝酒', '用户以茶代酒', '用户今天也没碰酒'],
         known_b=['用户早就戒酒了', '用户一直不喝酒'],
         bnd=['用户没有喝{A}', '用户一直没再沾酒']),
    dict(key='alcohol', lang=EN, A=['a glass of baijiu', 'beer', 'a cocktail'],
         known=['The user is allergic to alcohol', 'The user cannot touch alcohol',
                'Alcohol makes the user flush'],
         act=['The user drank {A} last night', 'The user just had {A}', 'The user drank {A} at the dinner'],
         avoid=['The user did not drink last night', 'The user substituted tea for alcohol',
                'The user has not touched alcohol today either'],
         known_b=['The user quit drinking long ago', 'The user never drinks'],
         bnd=['The user did not have {A}', 'The user has not drunk since quitting']),
    dict(key='vegetarian', lang=ZH, A=['红烧肉', '牛肉汉堡', '烤羊排'],
         known=['用户是素食者', '用户吃素五年了', '用户不碰任何肉类'],
         act=['用户昨天吃了{A}', '用户聚餐时点了{A}', '用户刚才吃了{A}'],
         avoid=['用户昨天没吃肉', '用户把{A}换成了豆腐版本', '用户今天也没碰荤菜'],
         known_b=['用户吃素已经很多年', '用户早就开始吃素'],
         bnd=['用户没有点{A}', '用户一直没再吃肉']),
    dict(key='vegetarian', lang=EN, A=['braised pork belly', 'a beef burger', 'lamb chops'],
         known=['The user is vegetarian', 'The user has eaten no meat for five years',
                'The user avoids all meat'],
         act=['The user ate {A} yesterday', 'The user ordered {A} at the dinner', 'The user just ate {A}'],
         avoid=['The user did not eat meat yesterday', 'The user swapped {A} for the tofu version',
                'The user skipped meat today as well'],
         known_b=['The user has been vegetarian for years', 'The user quit meat long ago'],
         bnd=['The user did not order {A}', 'The user has not eaten meat since']),
    dict(key='sugar', lang=ZH, A=['奶茶', '提拉米苏', '全糖气泡水'],
         known=['用户在控糖,医生叮嘱忌糖', '用户血糖偏高要忌甜', '用户正在严格控糖'],
         act=['用户昨天喝了{A}', '用户下午吃了{A}', '用户刚才来了一份{A}'],
         avoid=['用户昨天没碰甜的', '用户点的都是无糖版本', '用户今天也没吃甜食'],
         known_b=['用户控糖已经三个月', '用户早就开始控糖'],
         bnd=['用户没有点{A}', '用户一直没再吃甜的']),
    dict(key='sugar', lang=EN, A=['bubble tea', 'tiramisu', 'full-sugar soda'],
         known=['The user is on a strict low-sugar plan', 'The user must avoid sweets per doctor orders',
                'The user is cutting sugar'],
         act=['The user had {A} yesterday', 'The user ate {A} in the afternoon', 'The user just got {A}'],
         avoid=['The user did not have sweets yesterday', 'The user ordered sugar-free versions',
                'The user skipped desserts today as well'],
         known_b=['The user has been cutting sugar for months', 'The user started the low-sugar plan long ago'],
         bnd=['The user did not order {A}', 'The user has not had sweets since']),
    dict(key='gluten', lang=ZH, A=['韭菜盒子', '意面', '普通面包'],
         known=['用户对麸质过敏', '用户乳糜泻要无麸质饮食', '用户吃面食就过敏'],
         act=['用户昨天吃了{A}', '用户午餐点了{A}', '用户刚才吃了{A}'],
         avoid=['用户昨天没吃面食', '用户只吃无麸质面包', '用户今天也没碰小麦'],
         known_b=['用户确诊乳糜泻已经两年', '用户一直对麸质过敏'],
         bnd=['用户没有点{A}', '用户没让上含麸质的主食']),
    dict(key='gluten', lang=EN, A=['chive pockets', 'regular pasta', 'normal bread'],
         known=['The user has a gluten allergy', 'The user has celiac disease',
                'Wheat makes the user sick'],
         act=['The user ate {A} yesterday', 'The user ordered {A} for lunch', 'The user just had {A}'],
         avoid=['The user did not eat wheat yesterday', 'The user only eats gluten-free bread',
                'The user skipped gluten today as well'],
         known_b=['The user was diagnosed with celiac two years ago', 'The user has always avoided gluten'],
         bnd=['The user did not order {A}', 'The user skipped the gluten staples']),
    dict(key='smoke', lang=ZH, A=['烟', '一根烟', '电子烟'],
         known=['用户已经戒烟了', '用户戒烟半年了', '用户早就把烟戒了'],
         act=['用户昨天又抽上{A}了', '用户刚才抽了{A}', '用户聚会时抽了{A}'],
         avoid=['用户一直没有复吸', '用户昨天也没抽烟', '用户到现在都没再碰{A}'],
         known_b=['用户戒烟很久了', '用户把烟戒掉一年多了'],
         bnd=['用户没有复吸', '用户一直没再抽{A}']),
    dict(key='smoke', lang=EN, A=['cigarettes', 'a cigarette', 'the vape'],
         known=['The user quit smoking', 'The user has been smoke-free for half a year',
                'The user gave up smoking long ago'],
         act=['The user started smoking {A} again yesterday', 'The user just smoked {A}',
              'The user smoked {A} at the party'],
         avoid=['The user has not relapsed', 'The user did not smoke yesterday either',
                'The user has not touched {A} since quitting'],
         known_b=['The user quit smoking over a year ago', 'The user has been off cigarettes for ages'],
         bnd=['The user has not relapsed', 'The user has not smoked {A} since']),
    dict(key='barber', lang=ZH, A=['阿杰', 'Andy', 'Tony'],
         known=['用户不找{A}剪发了', '用户已经不在{A}那里剪发了', '用户再也没去{A}那剪过'],
         act=['用户昨天又去找{A}剪了头发', '用户刚预约了{A}', '用户上次还是{A}剪的'],
         avoid=['用户昨天没去找{A}', '用户现在固定找别的发型师', '用户一直没再约{A}'],
         known_b=['用户换掉{A}很久了', '用户早就不找{A}了'],
         bnd=['用户没有再约{A}', '用户一直没回去找{A}']),
    dict(key='barber', lang=EN, A=['Jeremy', 'Kevin', 'Tony'],
         known=['The user no longer gets haircuts from {A}', 'The user stopped seeing {A} months ago',
                'The user never went back to {A}'],
         act=['The user went back to {A} for a cut yesterday', 'The user just booked {A}',
              'The user last had {A} cut their hair'],
         avoid=['The user did not see {A} yesterday', 'The user now sees a different stylist regularly',
                'The user has not booked {A} since'],
         known_b=['The user switched away from {A} long ago', 'The user stopped seeing {A} ages ago'],
         bnd=['The user has not rebooked {A}', 'The user never went back to {A}']),
]


# 4th act per ZH family: the ZH neg_true space (~247 unique states) sits below
# the 288 zh quota once collisions with earlier blocks are excluded
ACT4 = {
    'spicy': '用户昨天没忍住,吃了{A}', 'peanut': '用户昨天没忍住,吃了{A}',
    'lactose': '用户昨天没忍住,碰了{A}', 'alcohol': '用户昨天没忍住,喝了{A}',
    'vegetarian': '用户昨天没忍住,吃了{A}', 'sugar': '用户昨天没忍住,吃了{A}',
    'gluten': '用户昨天没忍住,吃了{A}', 'smoke': '用户昨天没忍住,抽了{A}',
    'barber': '用户昨天没忍住,又找{A}剪了头发',
}
for _f in NEG_FAMILIES:
    if _f['lang'] == ZH and _f['key'] in ACT4:
        _f['act'] = _f['act'] + [ACT4[_f['key']]]


# widen the neg_false combinatorial space (300 unique states needed; the base
# avoid lists alone exhaust around 284 after dedup)
for _f in NEG_FAMILIES:
    if _f['lang'] == ZH:
        _f['avoid'] += ['用户这次也没点{A}', '用户到现在都没碰过{A}']
    else:
        _f['avoid'] += ['The user skipped {A} this time too', 'The user has never touched {A}']


def render_neg(rng, fam, kind):
    if kind == 'neg_true':
        known = rng.choice(fam['known']).format(A=rng.choice(fam['A']))
        new = rng.choice(fam['act']).format(A=rng.choice(fam['A']))
    elif kind == 'neg_false':
        known = rng.choice(fam['known']).format(A=rng.choice(fam['A']))
        new = rng.choice(fam['avoid']).format(A=rng.choice(fam['A']))
    elif kind == 'neg_false_boundary':
        known = rng.choice(fam['known_b']).format(A=rng.choice(fam['A']))
        new = rng.choice(fam['bnd']).format(A=rng.choice(fam['A']))
    else:
        raise ValueError(kind)
    return json.dumps({'known': known, 'new': new}, ensure_ascii=False)


def make_neg_rows(rng, quota, lang_mix, seen, label_rng, target_rng, ls_pool):
    # per-language exact quotas: rejection-redraw otherwise skews zh down when
    # zh states collide more with earlier blocks (the negation fix is aimed at
    # the zh NEGATION_CASES, so zh share must not silently erode)
    rows = []
    for kind, n in quota.items():
        n_en = round(n * lang_mix[EN])
        per_lang = {ZH: n - n_en, EN: n_en}
        for lang, n_lang in per_lang.items():
            pool = [f for f in NEG_FAMILIES if f['lang'] == lang]
            made, guard = 0, 0
            while made < n_lang and guard < n_lang * 800:
                guard += 1
                state = render_neg(rng, rng.choice(pool), kind)
                if state in seen:
                    continue
                seen.add(state)
                ls = next_ls(ls_pool)
                rows.append({'state': state, 'questions': question_for_v5(ls, lang, label_rng),
                             'gold': {'conflict': gold_for(kind, target_rng)},
                             '_meta': {'cat': 'negation', 'kind': kind, 'lang': lang, 'labelset': ls}})
                made += 1
            assert made == n_lang, (kind, lang, made, n_lang)
    return rows


# --- block B: known-side one-step perturbation (§3.2 B) -----------------------
# Realtest SHAPES (fresh wording; the 20 frozen realtest lines never enter
# training) rendered with the same interface as v3 categories.  Key suffix
# convention: *_en -> EN, else ZH.
SHAPE_CATS = [
    cat_v5('shape_meeting', ['周一上午十点', '周四下午三点', '周五早上九点', '周二下午两点'],
           known=['用户的周会固定在{A}', '用户例会时间是{A}'],
           conflict=['周会已经改到{B}了', '用户把例会挪到了{B}'],
           compat=['用户{A}开完会都会整理纪要', '会议纪要里写的是{A}的议程'],
           consid=['用户在考虑把周会挪到{B}', '用户在纠结要不要改到{B}'],
           unrelated=['会议室投影仪换了新的', '行政在统计下周订水数量']),
    cat_v5('shape_meeting_en', ['Monday 10am', 'Thursday 3pm', 'Friday 9am', 'Tuesday 2pm'],
           known=['The user has a standing meeting at {A}', 'The weekly sync is at {A}'],
           conflict=['The meeting has moved to {B}', 'The user moved the sync to {B}'],
           compat=['The user takes notes right after the {A} meeting', 'The agenda still lists {A} items'],
           consid=['The user is considering moving the sync to {B}', 'The user is debating a {B} slot'],
           unrelated=['The projector in the meeting room was replaced', 'Office is counting water orders']),
    cat_v5('shape_streaming', ['视频会员', '音乐会员', '网盘会员'],
           known=['用户没有订阅{A}', '用户没开{A}'],
           conflict=['用户昨晚开通了{A}', '用户刚续费了{A}'],
           compat=['用户在{A}的活动页领过优惠券', '用户收藏过{A}的片单'],
           consid=['用户在想要不要开{A}', '用户有点心动{A}的年卡'],
           unrelated=['用户给电视换了新壁挂架', '客厅路由器升级了']),
    cat_v5('shape_streaming_en', ['a video membership', 'a music subscription', 'cloud storage plan'],
           known=['The user has not subscribed to {A}', 'The user never signed up for {A}'],
           conflict=['The user signed up for {A} last night', 'The user just renewed {A}'],
           compat=['The user once grabbed a coupon from the {A} promo page', 'The user saved playlists from {A}'],
           consid=['The user is considering {A}', 'The user is tempted by the {A} annual plan'],
           unrelated=['The user mounted a new TV bracket', 'The living-room router got upgraded']),
    cat_v5('shape_commute', ['地铁', '公交', '骑车', '开车'],
           known=['用户通勤坐{A}', '用户上下班靠{A}'],
           conflict=['用户通勤改成{B}了', '用户现在上下班改{B}了'],
           compat=['用户{A}上会听播客', '用户的{A}卡刚充了值'],
           consid=['用户在考虑改成{B}通勤', '用户在纠结要不要换{B}'],
           unrelated=['地铁沿线新开了家面馆', '用户给车加了玻璃水']),
    cat_v5('shape_commute_en', ['the subway', 'the bus', 'a bike', 'driving'],
           known=['The user commutes by {A}', 'The user gets to work via {A}'],
           conflict=['The user switched their commute to {B}', 'The user now commutes by {B}'],
           compat=['The user listens to podcasts on {A}', 'The user just topped up the {A} card'],
           consid=['The user is considering switching to {B}', 'The user is debating {B} for the commute'],
           unrelated=['A new noodle shop opened along the line', 'The user topped up washer fluid']),
    cat_v5('shape_delivery', ['公司前台', '家门口的驿站', '物业代收点'],
           known=['用户的默认收货地址是{A}', '用户快递默认放{A}'],
           conflict=['用户把默认收货地址改成了公司楼下驿站', '用户的收货偏好换地方了,现在写的是新驿站的地址'],
           compat=['用户让{A}的同事帮忙代收过一个包裹', '用户的退货件也是走{A}'],
           consid=['用户在想要不要把收货地址改到公司附近', '用户在考虑换个代收点'],
           unrelated=['用户买了一卷打包胶带', '快递柜清了一次积件']),
    cat_v5('shape_delivery_en', ['the office front desk', 'the pickup point downstairs', 'the doorman'],
           known=['The user has packages delivered to {A}', 'The default delivery spot is {A}'],
           conflict=['The user changed the default delivery address to a new pickup point',
                     'The delivery preference now points to a different locker address'],
           compat=['A colleague at {A} once signed for a parcel for the user', 'Returns also go through {A}'],
           consid=['The user is considering moving deliveries near the office', 'The user is weighing a new pickup point'],
           unrelated=['The user bought a roll of packing tape', 'The parcel locker was emptied']),
    cat_v5('shape_injury', ['膝盖韧带拉伤', '脚踝扭伤', '腰肌劳损'],
           known=['用户{A},医生要求静养一个月', '用户{A}还在恢复期'],
           conflict=['用户周末打算去踢球,不顾{A}的医嘱', '用户不等{A}好利索就去跑步了'],
           compat=['用户{A}期间在 gym 做上肢训练', '用户按医嘱复查了{A}'],
           consid=['用户在想要不要提前结束{A}的静养', '用户有点想提前恢复训练'],
           unrelated=['用户给护膝换了新的', '用户在研究运动保险']),
    cat_v5('shape_injury_en', ['a sprained knee', 'a twisted ankle', 'a strained back'],
           known=['The user has {A} and the doctor ordered a month of rest', 'The user is recovering from {A}'],
           conflict=['The user plans to play soccer this weekend against medical advice about {A}',
                     'The user went running before the {A} healed'],
           compat=['The user does upper-body gym work during the {A} recovery', 'The user had the {A} reviewed as told'],
           consid=['The user is considering cutting the {A} rest short', 'The user is tempted to resume training early'],
           unrelated=['The user bought a new knee brace', 'The user is researching sports insurance']),
]


# qualifier / synonym rewrites of `known` -- the "one-step perturbation" (§3.2 B)
PERTURB = {
    ('city'): ['用户的户籍所在地是{A}', '用户身份证上住址是{A}'],
    ('lang'): ['用户的技术栈主力是{A}', '用户简历上写的编程语言是{A}'],
    ('gym_day'): ['用户的固定锻炼日是{A}', '用户的健身排期定在{A}'],
    ('coffee'): ['用户的咖啡订单默认{A}', '用户常点的那款是{A}'],
    ('cloud'): ['用户的云上资源都在{A}', '用户的生产环境在{A}'],
    ('phone'): ['用户在用的手机型号是{A}', '用户的主力机是{A}'],
    ('job'): ['用户的正式职位是{A}', '用户在公司担任{A}'],
    ('hair'): ['用户目前的发型是{A}', '用户理发后留的是{A}'],
    ('team'): ['用户主队是{A}', '用户主支持的俱乐部是{A}'],
    ('work_start'): ['用户的上班打卡时间是{A}', '用户公司要求{A}到岗'],
    ('rent'): ['用户的月租金为{A}', '用户每月房租是{A}'],
    ('car'): ['用户开的{A}', '用户名下车是{A}'],
    ('degree'): ['用户双眼近视{A}', '用户验光单上写的是近视{A}'],
    ('barber'): ['用户的固定发型师是{A}', '用户一直找{A}剪发'],
}


def make_cf_rows(rng, cats_attr, cats_shape, quota, lang_mix, seen, label_rng, target_rng, ls_pool):
    rows = []
    for kind, n in quota.items():
        n_en = round(n * lang_mix[EN])
        per_lang = {ZH: n - n_en, EN: n_en}
        for lang, n_lang in per_lang.items():
            pool_shape = [c for c in cats_shape if c['lang'] == lang]
            pool_attr = [c for c in cats_attr if c['lang'] == lang]
            made, guard = 0, 0
            while made < n_lang and guard < n_lang * 800:
                guard += 1
                # mix: 1/3 realtest-shape families, 2/3 regular attribute categories
                use_shape = pool_shape and rng.random() < 1 / 3
                pool = pool_shape if use_shape else pool_attr
                if not pool:
                    continue
                c = rng.choice(pool)
                a, b = rng.sample(c['vals'], 2)
                known = rng.choice(c['known']).format(A=a)
                perturbs = PERTURB.get(c['key'])
                if perturbs and rng.random() < 0.6:
                    known = rng.choice(perturbs).format(A=a)
                if kind == 'cf_true':
                    new = rng.choice(c['conflict']).format(A=a, B=b)
                else:
                    new = rng.choice(c['compat']).format(A=a, B=b)
                state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                if state in seen:
                    continue
                seen.add(state)
                ls = next_ls(ls_pool)
                rows.append({'state': state, 'questions': question_for_v5(ls, lang, label_rng),
                             'gold': {'conflict': gold_for(kind, target_rng)},
                             '_meta': {'cat': 'perturb', 'kind': kind, 'lang': lang, 'labelset': ls}})
                made += 1
            assert made == n_lang, (kind, lang, made, n_lang)
    return rows


LS_WEIGHTS = [0.15, 0.15, 0.15, 0.15, 0.20, 0.20]  # semantic 60% / LS5 20% / LS6 20%


def make_ls_pool(label_rng, total):
    """Exact 60/20/20 labelset quota as a shuffled pool (multinomial sampling
    drifts ~1.5-sigma below the hard 20% floors; V5 §3.1 makes them hard)."""
    n_sem = (total * 60) // 100 // 4
    n_neu = (total * 20) // 100
    pool = [1] * n_sem + [2] * n_sem + [3] * n_sem + [4] * n_sem + [5] * n_neu + [6] * n_neu
    assert len(pool) == total, (len(pool), total)
    label_rng.shuffle(pool)
    return pool


def next_ls(ls_pool):
    return ls_pool.pop()


def main():
    rng = random.Random(SEED_V5)
    label_rng = random.Random(SEED_V5 + 1)
    target_rng = random.Random(SEED_V5 + 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # acceptance sets must never leak into train (V5 §3.2/§8)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: same extraction as v4 (exact state match), now name-rotated too
    ls_pool = make_ls_pool(label_rng, 13400)
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
        r['questions'] = question_for_v5(r['_meta']['labelset'], ZH, label_rng)  # v4 kept zh instructions on MNLI

    # synthetic soft block (same quotas as v4) + hard-constraint block (same as v4)
    cats = patch_categories()
    train_pool = [c for c in cats if not c['holdout']]
    soft_rows = make_soft_rows_v5(rng, train_pool,
                                  {'true': 1800, 'compat': 1440, 'consid': 1080, 'unrelated': 1080},
                                  {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    hc_rows = make_hc_rows_v5(rng, 300, 300, {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng,
                              patch_hc(), ls_pool)

    # new blocks
    neg_rows = make_neg_rows(rng, {'neg_true': 400, 'neg_false': 300, 'neg_false_boundary': 100},
                             {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    shape_cats = []
    for c in SHAPE_CATS:
        c = dict(c, lang=EN if c['key'].endswith('_en') else ZH)
        shape_cats.append(c)
    cf_rows = make_cf_rows(rng, train_pool, shape_cats,
                           {'cf_true': 300, 'cf_false': 300},
                           {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)

    v5_train = mnli_rows + soft_rows + hc_rows + neg_rows + cf_rows
    rng.shuffle(v5_train)

    # ---- §3.4 hard invariants -------------------------------------------------
    assert len(v5_train) == 13400, len(v5_train)
    n_true = sum(1 for r in v5_train if r['gold']['conflict']['label'] == 'true')
    n_false = len(v5_train) - n_true
    states = [r['state'] for r in v5_train]
    assert len(set(states)) == 13400, 'duplicate states'
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        other = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, p), encoding='utf-8') if l.strip()}
        assert set(states).isdisjoint(other), f'leak into {p}'

    kind_counts = Counter(r['_meta']['kind'] for r in v5_train)
    expect_kinds = {'contradiction': 2000, 'entailment/neutral': 4000, 'true': 1800, 'compat': 1440,
                    'consid': 1080, 'unrelated': 1080, 'hardconstraint': 300, 'hc_safe': 180,
                    'hc_unrelated': 120, 'neg_true': 400, 'neg_false': 300, 'neg_false_boundary': 100,
                    'cf_true': 300, 'cf_false': 300}
    assert dict(kind_counts) == expect_kinds, (kind_counts, expect_kinds)

    label_mix = Counter(r['_meta']['labelset'] for r in v5_train)
    assert label_mix[5] == 2680 and label_mix[6] == 2680, label_mix
    assert sum(label_mix[s] for s in (1, 2, 3, 4)) == 13400 - 5360, label_mix
    for ls in range(1, 7):
        kinds_with = {r['_meta']['kind'] for r in v5_train if r['_meta']['labelset'] == ls}
        assert len(kinds_with) >= 3, (ls, len(kinds_with))

    ls_rng = random.Random(777)  # re-derive each row's label strings for the audit check
    for r in v5_train:
        q = r['questions']['conflict']
        labels = q['labels']
        g = r['gold']['conflict']
        assert g['label'] in ('true', 'false')
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

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v5.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v5_train:
            meta = r.pop('_meta')
            meta['name_a'] = r['questions']['conflict']['labels']['false']
            meta['name_b'] = r['questions']['conflict']['labels']['true']
            r['meta'] = meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v5_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false / n_true:.2f})')
    print('meta.kind:', dict(sorted(Counter(r["meta"]["kind"] for r in v5_train).items())))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v5_train)))
    print('meta.labelset:', dict(sorted(Counter(r["meta"]["labelset"] for r in v5_train).items())))
    sem = sum(Counter(r["meta"]["labelset"] for r in v5_train)[s] for s in (1, 2, 3, 4))
    print(f'labelset ratio: semantic {sem / 13400:.1%}, LS5 {label_mix[5] / 13400:.1%}, LS6 {label_mix[6] / 13400:.1%}')


# v5 wrappers around v4's row builders: 6-labelset rotation + graded targets ----
def make_soft_rows_v5(rng, cats, quota, lang_mix, seen, label_rng, target_rng, ls_pool):
    rows = []
    for kind, n in quota.items():
        made, guard = 0, 0
        while made < n and guard < n * 800:
            guard += 1
            lang = rng.choices(list(lang_mix), weights=list(lang_mix.values()))[0]
            cand = [c for c in cats if c['lang'] == lang]
            if not cand:
                continue
            state, _, catkey, k = render_soft(rng, rng.choice(cand), kind)
            if state in seen:
                continue
            seen.add(state)
            ls = next_ls(ls_pool)
            rows.append({'state': state, 'questions': question_for_v5(ls, lang, label_rng),
                         'gold': {'conflict': gold_for(k, target_rng)},
                         '_meta': {'cat': catkey, 'kind': k, 'lang': lang, 'labelset': ls}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


def make_hc_rows_v5(rng, n_true, n_control, lang_mix, seen, label_rng, target_rng, hc_pool, ls_pool):
    rows = []
    quota = {'hardconstraint': n_true, 'hc_safe': int(n_control * 0.6),
             'hc_unrelated': n_control - int(n_control * 0.6)}
    for kind, n in quota.items():
        made, guard = 0, 0
        while made < n and guard < n * 800:
            guard += 1
            lang = rng.choices(list(lang_mix), weights=list(lang_mix.values()))[0]
            pool = [h for h in hc_pool if h['lang'] == lang]
            if not pool:
                continue
            state, _gold = render_hc(rng, rng.choice(pool), kind)
            if state in seen:
                continue
            seen.add(state)
            ls = next_ls(ls_pool)
            rows.append({'state': state, 'questions': question_for_v5(ls, lang, label_rng),
                         'gold': {'conflict': gold_for(kind, target_rng)},
                         '_meta': {'cat': 'hardconstraint', 'kind': kind, 'lang': lang, 'labelset': ls}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


if __name__ == '__main__':
    main()
