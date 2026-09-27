# -*- coding: utf-8 -*-
"""v4 soft-conflict + hard-constraint augmentation (HANDOFF_NLI_V3.md, 改动1+2).

Based on gen_soft_conflicts_v3.py ("照着改"): same category structure and pair
shapes, with these v4 changes:

  1. four synthetic kinds scaled x1.8 (3000 -> 5400), block ratio stays 1:2
  2. NEW 5th class "hard-constraint violation": 300 true + 300 controls (>=1:1
     controls are mandatory -- without them the model just flips miss->false-alarm)
  3. label-surface rotation: 4 label/instruction sets assigned per row at random,
     synthetic rows only (the 6000 MNLI rows stay byte-identical to v2/v3)
  4. every train row carries meta {cat, kind, lang, labelset}
  5. val_soft (300) is NOT regenerated; its states are seeded into the dedup set
     so the v4 train block can never leak a val_soft row.

v4 train = 6000 MNLI (extracted from nli_conflict_train_v3.jsonl by state match)
         + 5400 scaled synthetic + 600 hard-constraint = 12000 rows, global
         conflict:non-conflict ~= 1:1.93.

Deterministic: seed 20260930.
"""
import json
import os
import random
from collections import Counter

from gen_soft_conflicts_v3 import CATEGORIES, ZH, EN, cat

SEED = 20260930
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_local')

GOLD_FALSE = {'label': 'false', 'probabilities': {'false': 0.95, 'true': 0.05}}
GOLD_TRUE = {'label': 'true', 'probabilities': {'false': 0.05, 'true': 0.95}}

# --- label-surface rotation (改动2): 4 sets, instructions matched to labels ---
LABEL_SETS = [
    (1, '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
     {'false': '兼容', 'true': '冲突'}),
    (2, '新信息(new)与已有记忆(known)是否矛盾？矛盾=与旧记忆相悖需更新，一致或无关则保留',
     {'false': '一致或无关', 'true': '矛盾'}),
    (3, '新信息(new)与已有记忆(known)是否冲突？需更新旧记忆=新信息与旧记忆相悖，不冲突=一致或无关',
     {'false': '不冲突', 'true': '需更新旧记忆'}),
    (4, 'Does the new information (new) conflict with the known (old) memory? '
        'supersede = the new information contradicts the old memory and must replace it, '
        'keep = consistent or unrelated',
     {'false': 'keep', 'true': 'supersede'}),
]


def question_for(labelset_id):
    for sid, ins, labels in LABEL_SETS:
        if sid == labelset_id:
            return {'conflict': {'type': 'noul', 'instructions': ins, 'labels': labels}}
    raise ValueError(labelset_id)


# --- 4th known / compat patterns per category (v3 had 3/3; compat quota x1.8
#     needs the extra combinatorial space) -------------------------------------
KNOWN4 = {
    (ZH, 'city'): '用户的居住地是{A}', (ZH, 'lang'): '用户的日常开发离不开{A}',
    (ZH, 'gym_day'): '用户的运动日固定在每周{A}', (ZH, 'coffee'): '用户点单默认是{A}',
    (ZH, 'cloud'): '用户的域名解析在{A}', (ZH, 'phone'): '用户口袋里装的是{A}',
    (ZH, 'job'): '用户的岗位是{A}', (ZH, 'home_city'): '用户的籍贯是{A}',
    (ZH, 'hair'): '用户的发型是{A}', (ZH, 'team'): '用户支持的球队是{A}',
    (ZH, 'gym_brand'): '用户的健身卡在{A}', (ZH, 'work_start'): '用户的上班时间是{A}',
    (ZH, 'rent'): '用户的月租金为{A}', (ZH, 'car'): '用户的座驾是{A}',
    (ZH, 'cat_food'): '用户家猫的主粮是{A}', (ZH, 'alarm'): '用户的晨起闹钟是{A}',
    (ZH, 'email'): '用户的联系邮箱是{A}', (ZH, 'desk_floor'): '用户的座位在{A}',
    (ZH, 'degree'): '用户的双眼近视{A}', (ZH, 'barber'): '用户的固定发型师是{A}',
    (EN, 'city'): 'The user resides in {A}', (EN, 'lang'): 'The user does most of their coding in {A}',
    (EN, 'gym_day'): "The user's workout day is {A}", (EN, 'coffee'): "The user's default order is {A}",
    (EN, 'cloud'): "The user's workloads live on {A}", (EN, 'job'): "The user's role is {A}",
    (EN, 'hair'): "The user's hairstyle is {A}", (EN, 'car'): "The user's ride is {A}",
    (EN, 'alarm'): "The user's morning alarm is {A}", (EN, 'team'): 'The team the user supports is {A}',
}
COMPAT4 = {
    (ZH, 'city'): '用户的常用联系人都登记在{A}', (ZH, 'lang'): '用户在{A}社区还很活跃',
    (ZH, 'gym_day'): '用户{A}练完还会去游泳', (ZH, 'coffee'): '用户囤的咖啡豆也是配{A}的',
    (ZH, 'cloud'): '用户的快照备份策略还在{A}', (ZH, 'phone'): '用户的手机支付绑在{A}上',
    (ZH, 'job'): '用户手头的需求还是{A}的', (ZH, 'home_city'): '用户的亲戚都在{A}',
    (ZH, 'hair'): '用户留{A}时候的照片还在', (ZH, 'team'): '用户还穿着{A}的球衣',
    (ZH, 'gym_brand'): '用户的储物柜还租在{A}', (ZH, 'work_start'): '考勤记录显示用户{A}到岗',
    (ZH, 'rent'): '用户的押金单上写的是{A}', (ZH, 'car'): '用户的保险单上写的是{A}',
    (ZH, 'cat_food'): '用户的自动喂食器里装的是{A}', (ZH, 'alarm'): '用户的晨间日程从{A}开始',
    (ZH, 'email'): '用户的银行账单还发到{A}', (ZH, 'desk_floor'): '用户在{A}的储物格还没清空',
    (ZH, 'degree'): '用户的隐形眼镜也是按{A}配的', (ZH, 'barber'): '用户的预约记录里全是{A}',
    (EN, 'city'): "The user's emergency contact form still lists {A}",
    (EN, 'lang'): 'The user is still active in the {A} community',
    (EN, 'gym_day'): 'The user goes swimming after {A} workouts',
    (EN, 'coffee'): "The user's bean stash pairs with {A}",
    (EN, 'cloud'): "The user's snapshot backups still run on {A}",
    (EN, 'job'): "The user's current tickets are all {A} work",
    (EN, 'hair'): 'The user still keeps photos from their {A} days',
    (EN, 'car'): "The user's insurance card lists {A}",
    (EN, 'alarm'): "The user's morning schedule starts at {A}",
    (EN, 'team'): 'The user still wears their {A} jersey',
}


def patch_categories():
    for c in CATEGORIES:
        key = (c['lang'], c['key'])
        if key in KNOWN4:
            c['known'].append(KNOWN4[key])
        if key in COMPAT4:
            c['compat'].append(COMPAT4[key])
    return CATEGORIES


# --- hard-constraint class (改动1 第5类) --------------------------------------
# constraints: hard limits where *intent to violate* is already a conflict.
HC = [
    dict(key='peanut', lang=ZH, known=['用户对花生过敏,重度那种', '用户一吃花生就起疹子', '用户花生重度过敏'],
         items=['花生酱饼干', '加了花生的沙律', '花生碎冰淇淋'],
         intent=['用户想试试{X}', '用户打算尝尝{X}', '用户说想点一份{X}'],
         act=['用户昨天吃了{X}', '用户刚才点了{X}'],
         safe=['用户买了不含花生的小饼干', '用户点餐时特意备注了花生过敏', '用户随身带着抗过敏药'],
         unrel=['用户在查过敏原检测的机构', '用户最近在补充维生素']),
    dict(key='peanut', lang=EN, known=['The user has a severe peanut allergy',
                                       'The user breaks out in hives from peanuts',
                                       'Peanuts are a serious allergen for the user'],
         items=['peanut butter cookies', 'a salad topped with peanuts', 'peanut-crunch ice cream'],
         intent=['The user wants to try {X}', 'The user plans to taste {X}', 'The user says they will order {X}'],
         act=['The user ate {X} yesterday', 'The user just ordered {X}'],
         safe=['The user bought nut-free cookies', 'The user noted the peanut allergy on the order',
               'The user always carries an epipen'],
         unrel=['The user is looking into allergy testing clinics', 'The user started a vitamin routine']),
    dict(key='spicy', lang=ZH, known=['用户不吃辣', '用户碰一点辣就胃疼', '用户是无辣不欢的反面——完全忌辣'],
         items=['微辣火锅', '辣子鸡', '麻辣香锅'],
         intent=['用户今天想尝试一下{X}', '用户说想点{X}', '用户在想要不要来一份{X}'],
         act=['用户昨晚吃了{X}', '用户午餐点了{X}'],
         safe=['用户点菜时选了不辣的做法', '用户让厨房完全别放辣椒', '用户点的都是清汤锅底'],
         unrel=['用户在研究养胃食谱', '用户最近爱吃清淡的淮扬菜']),
    dict(key='spicy', lang=EN, known=['The user cannot eat spicy food', 'Spicy food upsets the user stomach',
                                      'The user strictly avoids chili'],
         items=['extra-spicy hotpot', 'spicy diced chicken', 'mala xiang guo'],
         intent=['The user is thinking of trying {X}', 'The user says they might order {X}',
                 'The user considers getting {X}'],
         act=['The user had {X} last night', 'The user ordered {X} for lunch'],
         safe=['The user asked for the non-spicy version', 'The user told the kitchen to skip chili entirely',
               'The user ordered the clear broth base'],
         unrel=['The user is researching stomach-friendly recipes', 'The user has been into mild dishes lately']),
    dict(key='lactose', lang=ZH, known=['用户乳糖不耐受', '用户喝牛奶就拉肚子', '用户对乳糖不耐受'],
         items=['芝士蛋糕', '拿铁', '奶油浓汤'],
         intent=['用户想试试{X}', '用户在想要不要来份{X}', '用户说想喝{X}'],
         act=['用户昨天吃了{X}', '用户下午点了{X}'],
         safe=['用户喝的是无乳糖牛奶', '用户点的都是燕麦奶基底', '用户备着乳糖酶'],
         unrel=['用户在查植物奶品牌', '用户最近在补钙']),
    dict(key='lactose', lang=EN, known=['The user is lactose intolerant', 'Milk makes the user sick',
                                        'The user cannot digest lactose'],
         items=['cheesecake', 'a latte', 'cream soup'],
         intent=['The user wants to try {X}', 'The user is considering {X}', 'The user feels like having {X}'],
         act=['The user had {X} yesterday', 'The user ordered {X} this afternoon'],
         safe=['The user drinks lactose-free milk', 'The user orders oat-milk bases only',
               'The user carries lactase pills'],
         unrel=['The user is comparing plant-milk brands', 'The user has been taking calcium supplements']),
    dict(key='alcohol', lang=ZH, known=['用户对酒精过敏', '用户沾酒就脸红心跳快', '用户滴酒不能沾'],
         items=['一杯白酒', '啤酒', '鸡尾酒'],
         intent=['用户饭局上想硬着头皮喝{X}', '用户说想尝一口{X}', '用户在想要不要来{X}'],
         act=['用户昨晚喝了{X}', '用户刚才碰了{X}'],
         safe=['用户以茶代酒', '用户聚餐时只点无醇啤酒', '用户提前说明了自己酒精过敏'],
         unrel=['用户在研究无醇饮品', '用户最近在调整作息']),
    dict(key='alcohol', lang=EN, known=['The user is allergic to alcohol', 'Alcohol makes the user flush and racing-heart',
                                        'The user cannot touch alcohol'],
         items=['a glass of baijiu', 'beer', 'a cocktail'],
         intent=['The user is thinking of drinking {X} at the dinner', 'The user says they might sip {X}',
                 'The user considers having {X}'],
         act=['The user drank {X} last night', 'The user just had some {X}'],
         safe=['The user substitutes tea for alcohol', 'The user orders non-alcoholic beer only',
               'The user told everyone about the alcohol allergy in advance'],
         unrel=['The user is exploring zero-proof drinks', 'The user has been fixing their sleep schedule']),
    dict(key='vegetarian', lang=ZH, known=['用户是素食者', '用户吃素五年了', '用户不碰任何肉类'],
         items=['红烧肉', '牛肉汉堡', '烤羊排'],
         intent=['用户想破例尝一口{X}', '用户在想要不要试试{X}', '用户说想点{X}'],
         act=['用户昨天吃了{X}', '用户聚餐时点了{X}'],
         safe=['用户点的全是素菜', '用户把{X}换成了豆腐版本', '用户只吃植物肉版本'],
         unrel=['用户在收藏素食餐厅', '用户最近在研究豆制品菜谱']),
    dict(key='vegetarian', lang=EN, known=['The user is vegetarian', 'The user has been vegan-curious and eats no meat for five years',
                                           'The user avoids all meat'],
         items=['braised pork belly', 'a beef burger', 'lamb chops'],
         intent=['The user is tempted to try {X}', 'The user is considering {X}', 'The user says they want {X}'],
         act=['The user ate {X} yesterday', 'The user ordered {X} at the dinner'],
         safe=['The user ordered all-vegetable dishes', 'The user swapped {X} for a tofu version',
               'The user only eats the plant-meat version'],
         unrel=['The user is bookmarking vegetarian restaurants', 'The user is exploring tofu recipes']),
    dict(key='shellfish', lang=ZH, known=['用户对海鲜过敏', '用户吃虾蟹就起风团', '用户对贝类严重过敏'],
         items=['麻辣小龙虾', '清蒸大闸蟹', '蒜蓉扇贝'],
         intent=['用户想试试{X}', '用户说想吃{X}', '用户在想要不要点{X}'],
         act=['用户昨晚吃了{X}', '用户刚点了{X}'],
         safe=['用户点的是素养殖水产的素食替代', '用户让后厨确认无海鲜成分', '用户只点了鸡肉类菜品'],
         unrel=['用户在备抗组胺药', '用户最近在学做家常菜']),
    dict(key='shellfish', lang=EN, known=['The user is allergic to shellfish', 'Shrimp and crab give the user hives',
                                          'The user has a severe shellfish allergy'],
         items=['spicy crayfish', 'steamed hairy crab', 'garlic scallops'],
         intent=['The user wants to try {X}', 'The user says they crave {X}', 'The user is considering ordering {X}'],
         act=['The user ate {X} last night', 'The user just ordered {X}'],
         safe=['The user asked the kitchen to confirm no shellfish', 'The user ordered chicken dishes only',
               'The user double-checked the ingredients list'],
         unrel=['The user stocked up on antihistamines', 'The user has been learning home cooking']),
    dict(key='sugar', lang=ZH, known=['用户在控糖,医生叮嘱忌糖', '用户血糖偏高要忌甜', '用户正在严格控糖'],
         items=['奶茶', '提拉米苏', '全糖气泡水'],
         intent=['用户想喝一杯{X}', '用户说想吃{X}', '用户在想要不要来份{X}'],
         act=['用户昨天喝了{X}', '用户下午吃了{X}'],
         safe=['用户点的都是无糖版本', '用户用代糖做了低糖甜品', '用户随身带着血糖仪'],
         unrel=['用户在研究粗粮食谱', '用户最近在跳绳锻炼']),
    dict(key='sugar', lang=EN, known=['The user is on a strict low-sugar plan', 'The user has high blood sugar and must avoid sweets',
                                      'The user is cutting sugar per doctor orders'],
         items=['bubble tea', 'tiramisu', 'full-sugar soda'],
         intent=['The user wants a {X}', 'The user says they crave {X}', 'The user is considering {X}'],
         act=['The user had {X} yesterday', 'The user ate {X} in the afternoon'],
         safe=['The user orders sugar-free versions', 'The user bakes low-sugar desserts with sweeteners',
               'The user carries a glucose meter'],
         unrel=['The user is exploring whole-grain recipes', 'The user has been jumping rope lately']),
    dict(key='gluten', lang=ZH, known=['用户对麸质过敏', '用户乳糜泻要无麸质饮食', '用户吃面食就过敏'],
         items=['韭菜盒子', '意面', '普通面包'],
         intent=['用户想尝尝{X}', '用户在想要不要来份{X}', '用户说想吃{X}'],
         act=['用户昨天吃了{X}', '用户午餐点了{X}'],
         safe=['用户只吃无麸质面包', '用户自带无麸质意面', '用户点菜前都确认面粉成分'],
         unrel=['用户在收集无麸质食谱', '用户最近在练核心力量']),
    dict(key='gluten', lang=EN, known=['The user has a gluten allergy', 'The user has celiac disease and eats gluten-free',
                                       'Wheat makes the user sick'],
         items=['chive pockets', 'regular pasta', 'normal bread'],
         intent=['The user wants to taste {X}', 'The user is considering {X}', 'The user says they miss {X} and wants it'],
         act=['The user ate {X} yesterday', 'The user ordered {X} for lunch'],
         safe=['The user only eats gluten-free bread', 'The user brings their own gluten-free pasta',
               'The user always confirms flour ingredients first'],
         unrel=['The user is collecting gluten-free recipes', 'The user has been doing core workouts']),
]


# space patch: safe templates carry no item variable, so their combinatorial space
# is known x safe (144 for 8 constraints x 2 langs) -- below the 180 control quota.
# Add 2 safe + 1 unrelated per constraint-lang to lift it above demand.
SAFE4 = {
    (ZH, 'peanut'): ['用户刚和餐厅确认了每道菜都不含花生', '用户把零食换成了无坚果混合装'],
    (ZH, 'spicy'): ['用户外食都会提前说明忌辣', '用户自带了一份清淡小菜'],
    (ZH, 'lactose'): ['用户的咖啡都用豆奶替代', '用户买甜品前会先看乳糖含量'],
    (ZH, 'alcohol'): ['用户聚会时杯里永远是酸梅汤', '用户应酬只举杯不饮'],
    (ZH, 'vegetarian'): ['用户的荤菜那份总会转给同事', '用户点披萨只选素配料'],
    (ZH, 'shellfish'): ['用户吃火锅只涮蔬菜豆制品', '用户连蟹肉棒都会先看成分'],
    (ZH, 'sugar'): ['用户的饮品单一律写无糖或少糖', '用户的下午茶换成了原味坚果'],
    (ZH, 'gluten'): ['用户的主食以米类为主', '用户的饼干选的是米制脆饼'],
    (EN, 'peanut'): ['The user double-checks every snack label for peanut traces', 'The user sticks to nut-free snack packs'],
    (EN, 'spicy'): ['The user always orders mild by default', 'The user carries a chili note when eating out'],
    (EN, 'lactose'): ['The user swaps in soy milk everywhere', 'The user reads dessert labels for lactose first'],
    (EN, 'alcohol'): ['The user toasts with juice at dinners', 'The user only holds the glass, never drinks'],
    (EN, 'vegetarian'): ['The user always passes the meat dish to friends', 'The user orders veggie-only pizza toppings'],
    (EN, 'shellfish'): ['The user hotpots only vegetables and tofu', 'The user even checks surimi ingredients'],
    (EN, 'sugar'): ['The user writes light-sugar-or-less on every order', 'The user keeps afternoon snacks to plain nuts'],
    (EN, 'gluten'): ['The user sticks to rice-based staples', 'The user picks rice crackers over wheat ones'],
}
UNREL4 = {
    (ZH, 'peanut'): '用户最近在整理厨房储物柜', (ZH, 'spicy'): '用户学会了煲汤',
    (ZH, 'lactose'): '用户在挑保温杯', (ZH, 'alcohol'): '用户喜欢研究菜品摆盘',
    (ZH, 'vegetarian'): '用户养了一盆罗勒', (ZH, 'shellfish'): '用户厨房换了口新锅',
    (ZH, 'sugar'): '用户的冰箱上贴满了便签', (ZH, 'gluten'): '用户在学做米糕',
    (EN, 'peanut'): 'The user is reorganizing the kitchen shelves', (EN, 'spicy'): 'The user learned to make soup',
    (EN, 'lactose'): 'The user is picking a new thermos', (EN, 'alcohol'): 'The user enjoys plating dishes nicely',
    (EN, 'vegetarian'): 'The user grows a basil plant', (EN, 'shellfish'): 'The user got a new wok',
    (EN, 'sugar'): 'The fridge is covered in sticky notes', (EN, 'gluten'): 'The user is learning to make rice cakes',
}


def patch_hc():
    for h in HC:
        key = (h['lang'], h['key'])
        if key in SAFE4:
            h['safe'] = h['safe'] + SAFE4[key]
        if key in UNREL4:
            h['unrel'] = h['unrel'] + [UNREL4[key]]
    return HC


def render_hc(rng, hc, kind):
    known = rng.choice(hc['known'])
    if kind == 'hardconstraint':
        x = rng.choice(hc['items'])
        tmpl = rng.choice(hc['intent'] if rng.random() < 0.6 else hc['act'])
        new = tmpl.format(X=x)
        gold = GOLD_TRUE
    elif kind == 'hc_safe':
        new = rng.choice(hc['safe'])
        gold = GOLD_FALSE
    elif kind == 'hc_unrelated':
        new = rng.choice(hc['unrel'])
        gold = GOLD_FALSE
    else:
        raise ValueError(kind)
    return json.dumps({'known': known, 'new': new}, ensure_ascii=False), gold


def make_hc_rows(rng, n_true, n_control, lang_mix, seen, label_rng, hc_pool):
    rows, made = [], Counter()
    quota = {'hardconstraint': n_true, 'hc_safe': int(n_control * 0.6), 'hc_unrelated': n_control - int(n_control * 0.6)}
    for kind, n in quota.items():
        made[kind] = 0
        guard = 0
        while made[kind] < n and guard < n * 800:
            guard += 1
            lang = rng.choices(list(lang_mix), weights=list(lang_mix.values()))[0]
            pool = [h for h in hc_pool if h['lang'] == lang]
            if not pool:
                continue
            state, gold = render_hc(rng, rng.choice(pool), kind)
            if state in seen:
                continue
            seen.add(state)
            ls = label_rng.choice([1, 2, 3, 4])
            rows.append({'state': state, 'questions': question_for(ls),
                         'gold': {'conflict': dict(gold)},
                         '_meta': {'cat': 'hardconstraint', 'kind': kind, 'lang': lang, 'labelset': ls}})
            made[kind] += 1
        assert made[kind] == n, (kind, made[kind], n)
    return rows


def render_soft(rng, c, kind):
    """Render one pair; returns (state_json, gold, cat_key, kind)."""
    a, b = rng.sample(c['vals'], 2)
    known = rng.choice(c['known']).format(A=a)
    if kind == 'true':
        new, gold = rng.choice(c['conflict']).format(A=a, B=b), GOLD_TRUE
    elif kind == 'compat':
        new, gold = rng.choice(c['compat']).format(A=a, B=b), GOLD_FALSE
    elif kind == 'consid':
        new, gold = rng.choice(c['consid']).format(A=a, B=b), GOLD_FALSE
    elif kind == 'unrelated':
        new, gold = rng.choice(c['unrelated']).format(A=a, B=b), GOLD_FALSE
    else:
        raise ValueError(kind)
    state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
    return state, gold, c['key'], kind


def make_soft_rows(rng, cats, quota, lang_mix, seen, label_rng):
    rows = []
    for kind, n in quota.items():
        made, guard = 0, 0
        while made < n and guard < n * 800:
            guard += 1
            lang = rng.choices(list(lang_mix), weights=list(lang_mix.values()))[0]
            cand = [c for c in cats if c['lang'] == lang]
            if not cand:
                continue
            state, gold, catkey, k = render_soft(rng, rng.choice(cand), kind)
            if state in seen:
                continue
            seen.add(state)
            ls = label_rng.choice([1, 2, 3, 4])
            rows.append({'state': state, 'questions': question_for(ls),
                         'gold': {'conflict': dict(gold)},
                         '_meta': {'cat': catkey, 'kind': k, 'lang': lang, 'labelset': ls}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


def main():
    rng = random.Random(SEED)
    label_rng = random.Random(SEED + 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # val_soft states must never leak into the train block
    for p in ('nli_conflict_val_soft.jsonl',):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: extract from v3 train by exact state match against the canonical MNLI file
    v3_rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, 'nli_conflict_train_v3.jsonl'),
                                           encoding='utf-8') if l.strip()]
    mnli_states = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, 'nli_conflict_train.jsonl'),
                                                        encoding='utf-8') if l.strip()}
    mnli_rows = [r for r in v3_rows if r['state'] in mnli_states]
    assert len(mnli_rows) == 6000, f'MNLI extraction got {len(mnli_rows)}'
    mnli_kind = {'false': 'entailment/neutral', 'true': 'contradiction'}
    for r in mnli_rows:
        lab = r['gold']['conflict']['label']
        r['_meta'] = {'cat': 'mnli', 'kind': mnli_kind[lab], 'lang': 'en', 'labelset': 1}

    # synthetic block
    cats = patch_categories()
    train_pool = [c for c in cats if not c['holdout']]
    soft_rows = make_soft_rows(rng, train_pool,
                               {'true': 1800, 'compat': 1440, 'consid': 1080, 'unrelated': 1080},
                               {ZH: 0.72, EN: 0.28}, seen, label_rng)
    hc_rows = make_hc_rows(rng, 300, 300, {ZH: 0.72, EN: 0.28}, seen, label_rng, patch_hc())

    v4_train = mnli_rows + soft_rows + hc_rows
    rng.shuffle(v4_train)

    # ---- hard invariants before writing
    assert len(v4_train) == 12000, len(v4_train)
    n_true = sum(1 for r in v4_train if r['gold']['conflict']['label'] == 'true')
    assert n_true == 4100, n_true  # 2000 mnli + 1800 soft + 300 hard
    n_false = len(v4_train) - n_true
    states = [r['state'] for r in v4_train]
    assert len(set(states)) == 12000, 'duplicate states'
    val_soft_states = {json.loads(l)['state'] for l in
                       open(os.path.join(OUT_DIR, 'nli_conflict_val_soft.jsonl'), encoding='utf-8') if l.strip()}
    assert set(states).isdisjoint(val_soft_states), 'val_soft leak into train'
    label_mix = Counter(r['_meta']['labelset'] for r in v4_train)

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v4.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v4_train:
            meta = r.pop('_meta')
            r['meta'] = meta  # 改动2: every train row carries meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v4_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false/n_true:.2f})')
    print('meta.kind:', dict(Counter(r["meta"]["kind"] for r in v4_train)))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v4_train)))
    print('meta.labelset:', dict(sorted(label_mix.items())))


if __name__ == '__main__':
    main()
