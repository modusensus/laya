# -*- coding: utf-8 -*-
"""v6 data arm (HANDOFF_NLI_V6.md §3): data-hygiene reset + unrelated controls
+ negation shape families, on top of the v5 recipe (pure-CE main arm trains on
this in §4).

Changes vs gen_soft_conflicts_v5.py ("照着 v5 改"):

  1. LEAK CLEARANCE (§3.1, first priority): the round-5 re-verification's
     leak_audit found 12/35 acceptance cases with verbatim sentences inside the
     training corpus (171/13400 rows, incl. ONE full pair from new-10; v4 had
     6 legacy cases).  v6 clears ALL of them (incl. v3/v4 legacy):
       - value-pool swaps on the v3 category families (values only, families
         and shapes untouched): city 上海->南京/北京->重庆, coffee 美式->澳白,
         cloud 阿里云->百度智能云/腾讯云->京东云
       - literal known rewrites: v4 HC spicy '用户不吃辣'->'用户碰不得辣';
         v5 neg families spicy '用户向来不吃辣', peanut '用户对花生严重过敏',
         smoke '用户戒烟好几个月了'
       - v5 shape-cat value swaps: meeting 周一上午十点->周三上午十点 /
         周四下午三点->周四下午四点, commute 地铁->轻轨, delivery 公司前台->单位前台,
         injury 膝盖韧带拉伤->膝盖半月板损伤
     Negation-operator and allergen-shape coverage is NOT narrowed -- only the
     literal values move.  The §3.6 leak assert (hard invariant) + leak_audit.py
     re-run enforce 0/35 from here on.

  2. UNRELATED CONTROL ROWS (§3.2, root cause #1): both new blocks lacked
     same-topic unrelated rows, so both v5 arms false-alarmed the old-20
     '无关补充' case (19/20).  neg block +400 (neg_unrelated), cf block +300
     (cf_unrelated): same-family topic + no logical conflict (supplement /
     mention / parallel fact) => target false.  Per family >=30% rendered as
     the '同话题无关补充' angle (user action around the topic, orthogonal to
     the known attribute) -- implemented as an exact 75%/25% supp/mention
     quota per family+language.

  3. NEGATION FIVE-SHAPE FAMILIES (§3.3, root cause #2): one family per
     NEGATION_CASES shape (pet/property reversal, never-X daily habit,
     relapse, double negation, no-car compatible commute), +80 rows each
     = +400.  Shapes are borrowed, verbatim case text never enters (the v6
     assert rejects it; two shapes had NO coverage at all in v5).  Rows map
     onto existing kinds by shape semantics (neg_true / neg_false /
     neg_false_boundary) and carry meta.shape for per-shape reporting.

  4. GRADED SOFT TARGETS carried from v5 (§3.4 table) + two new tiers:
     neg_unrelated / cf_unrelated = 0.96 ± 0.01.

v6 train = 6000 MNLI + 5400 synthetic + 600 hard-constraint + 1600 negation
         (800 base + 400 five-shape + 400 unrelated) + 900 perturbation
         (600 + 300 unrelated) = 14500 rows.  val (1000) / val_soft (300) are
         NOT regenerated (byte-identical, directly comparable).

Known cost carried from v5: val/val_soft gold stays 0.95 while training
targets are graded => those sets carry accuracy/error-count/polarity criteria
only (ECE rewrite per V5 §5.2 / V6 §5.2).

Deterministic: seed 20260930 (same as v4/v5; blocks consume the stream in
fixed order).
"""
import json
import os
import random
from collections import Counter

from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import (SEED, OUT_DIR, patch_categories, patch_hc,
                                   render_hc, render_soft, HC)
from gen_soft_conflicts_v5 import (question_for_v5 as question_for,
                                   make_ls_pool, next_ls, TARGETS as V5_TARGETS,
                                   GOLD_LABEL as V5_GOLD_LABEL)

SEED_V6 = 20260930  # == SEED from v4/v5; named locally for clarity

# --- §3.1 leak clearance: value swaps + literal rewrites ----------------------
# Only values move; family structure / ratios / tone stay.  No replacement
# value may equal any acceptance sentence (the §3.6 assert is the backstop).
VAL_SWAPS = {
    (ZH, 'city'): {'上海': '南京', '北京': '重庆'},
    (ZH, 'coffee'): {'美式': '澳白'},
    (ZH, 'cloud'): {'阿里云': '百度智能云', '腾讯云': '京东云'},
}


def swap_vals(cats):
    for c in cats:
        m = VAL_SWAPS.get((c['lang'], c['key']))
        if m:
            c['vals'] = [m.get(v, v) for v in c['vals']]
    return cats


def rewrite_hc():
    # v4 legacy leak: HC spicy ZH known[0] is verbatim the 饮食禁忌松动 case known
    for h in HC:
        if h['lang'] == ZH and h['key'] == 'spicy':
            h['known'] = ['用户碰不得辣' if k == '用户不吃辣' else k for k in h['known']]
    return HC


# --- graded soft targets: v5 tiers + the two new unrelated tiers (§3.4) -------
TARGETS = {**V5_TARGETS, 'neg_unrelated': (0.96, 0.01), 'cf_unrelated': (0.96, 0.01)}
GOLD_LABEL = {**V5_GOLD_LABEL, 'neg_unrelated': 'false', 'cf_unrelated': 'false'}


def gold_for(kind, target_rng):
    """v5's gold_for but resolving against the v6 TARGETS/GOLD_LABEL tables."""
    base, j = TARGETS[kind]
    p = base if j == 0 else base + target_rng.uniform(-j, j)
    p = round(min(max(p, 0.7), 0.98), 4)
    assert abs(p - 0.5) >= 0.2, (kind, p)
    lab = GOLD_LABEL[kind]
    other = 'true' if lab == 'false' else 'false'
    return {'label': lab, 'probabilities': {lab: round(p, 4), other: round(1 - p, 4)}}


def cat_v5(key, vals, known, conflict, compat, consid, unrelated):
    return dict(key=key, lang=None, vals=vals, known=known, conflict=conflict,
                compat=compat, consid=consid, unrelated=unrelated, holdout=False)


# --- block A: negation-operator families (v5's, 3 knowns reworded §3.1,
#     each family + supp/mention unrelated lists for §3.2) ----------------------
# unrel['supp'] = user-action supplements around the topic (无关补充 angle);
# unrel['mention'] = third-party / venue / parallel-fact mentions.
NEG_FAMILIES = [
    dict(key='spicy', lang=ZH, A=['辣的', '麻辣香锅', '辣子鸡'],
         known=['用户向来不吃辣', '用户从不吃辣', '用户一点辣都不碰'],
         act=['用户昨天点了{A}', '用户午餐吃了{A}', '用户刚下单了一份{A}'],
         avoid=['用户昨天没吃辣', '用户点的是不辣的做法', '用户今天也没碰辣的'],
         known_b=['用户已经忌辣半年了', '用户早就不吃辣了'],
         bnd=['用户没有点{A}', '用户一直没再碰{A}'],
         supp=['用户{T}在整理本周菜单,想找几家清淡的馆子', '用户{T}研究了几道蒸菜的做法',
               '用户在给部门聚餐订清淡口的餐厅', '用户{T}把家里的辣椒酱送给了同事'],
         mention=['用户家附近的川菜馆{T}搞周年庆', '用户的同事午餐点了{A},用户自己没点',
                  '用户{T}刷到一条辣度挑战的视频']),
    dict(key='spicy', lang=EN, A=['the spicy version', 'extra chili', 'mala xiang guo'],
         known=['The user cannot eat spicy food', 'The user never eats chili',
                'The user strictly avoids spice'],
         act=['The user ordered {A} yesterday', 'The user had {A} for lunch', 'The user just ordered {A}'],
         avoid=['The user did not eat spicy food yesterday', 'The user ordered the mild version',
                'The user skipped chili again today'],
         known_b=['The user has been off spice for months', 'The user quit spicy food long ago'],
         bnd=['The user did not order {A}', 'The user has not touched {A} since'],
         supp=['The user is planning a mild-food lineup for the week {T}', 'The user looked up steamed-dish recipes {T}',
               'The user is booking a mild-flavored place for the team lunch', 'The user gave their chili sauce to a coworker {T}'],
         mention=['The Sichuan place nearby is running an anniversary promo {T}',
                  "The user's coworker ordered {A} at lunch; the user did not",
                  'The user scrolled past a spice-challenge video {T}']),
    dict(key='peanut', lang=ZH, A=['花生酱饼干', '加了花生的沙律', '花生碎冰淇淋'],
         known=['用户对花生严重过敏', '用户碰花生就起疹子', '用户花生重度过敏'],
         act=['用户昨天吃了{A}', '用户刚才点了{A}', '用户下午尝了{A}'],
         avoid=['用户昨天没碰花生', '用户点的是无花生版本', '用户今天也没吃坚果'],
         known_b=['用户早就查明确认花生过敏', '用户一直对花生过敏'],
         bnd=['用户没有点含花生的东西', '用户没让厨房加花生'],
         supp=['用户{T}整理了家里的零食柜,把坚果类都清点了', '用户在研究过敏原检测要预约哪家机构',
               '用户{T}看了一期讲食物过敏急救的科普', '用户随身的包里常备着抗过敏药'],
         mention=['超市的{A}这周在打折', '用户{T}帮同事带过一次{A}', '公司茶歇订了{A},用户没领']),
    dict(key='peanut', lang=EN, A=['peanut butter cookies', 'a peanut-topped salad', 'peanut ice cream'],
         known=['The user has a severe peanut allergy', 'Peanuts give the user hives',
                'The user is highly allergic to peanuts'],
         act=['The user ate {A} yesterday', 'The user just ordered {A}', 'The user tried {A} this afternoon'],
         avoid=['The user avoided peanuts yesterday', 'The user ordered the peanut-free version',
                'The user has not touched nuts today either'],
         known_b=['The user has a confirmed peanut allergy on file', 'The user has always been allergic to peanuts'],
         bnd=['The user did not order anything with peanuts', 'The user asked the kitchen to skip peanuts'],
         supp=['The user inventoried the snack cabinet and sorted the nuts {T}', 'The user is booking an allergen test',
               'The user watched a food-allergy first-aid explainer {T}', 'The user always carries an antihistamine'],
         mention=['The {A} is on sale at the supermarket this week', 'The user picked up {A} for a coworker once {T}',
                  'The office ordered {A} for the break; the user skipped it']),
    dict(key='lactose', lang=ZH, A=['芝士蛋糕', '拿铁', '奶油浓汤'],
         known=['用户乳糖不耐受', '用户喝牛奶就拉肚子', '用户对乳糖不耐受'],
         act=['用户昨天吃了{A}', '用户下午点了{A}', '用户刚才喝了一份{A}'],
         avoid=['用户昨天没喝牛奶', '用户点的是燕麦奶基底', '用户今天也没碰奶制品'],
         known_b=['用户一直乳糖不耐受', '用户早就查出乳糖不耐受'],
         bnd=['用户没有点奶制的{A}', '用户没加牛奶'],
         supp=['用户{T}把常点的咖啡都换成燕麦奶基底', '用户在比较几款无奶植物奶的价格',
               '用户{T}收藏了一篇讲乳糖不耐受的文章', '用户包里放着乳糖酶片'],
         mention=['楼下咖啡店的{A}推出了新口味', '用户{T}帮同事带过一杯{A}', '部门下午茶团购了{A},用户没参加']),
    dict(key='lactose', lang=EN, A=['cheesecake', 'a latte', 'cream soup'],
         known=['The user is lactose intolerant', 'Milk makes the user sick',
                'The user cannot digest lactose'],
         act=['The user had {A} yesterday', 'The user ordered {A} in the afternoon', 'The user just drank {A}'],
         avoid=['The user did not drink milk yesterday', 'The user ordered the oat-milk base',
                'The user skipped dairy today as well'],
         known_b=['The user has always been lactose intolerant', 'The user was diagnosed lactose intolerant'],
         bnd=['The user did not order the dairy {A}', 'The user skipped the milk'],
         supp=['The user switched their usual coffee to the oat-milk base {T}', 'The user is comparing plant-milk prices',
               'The user saved an article about lactose intolerance {T}', 'The user keeps lactase pills in their bag'],
         mention=['The coffee shop downstairs launched a new {A} flavor', 'The user grabbed {A} for a coworker {T}',
                  'The team ordered {A} for afternoon tea; the user opted out']),
    dict(key='alcohol', lang=ZH, A=['一杯白酒', '啤酒', '鸡尾酒'],
         known=['用户对酒精过敏', '用户滴酒不能沾', '用户沾酒就脸红心跳快'],
         act=['用户昨晚喝了{A}', '用户刚才碰了{A}', '用户饭局上喝了{A}'],
         avoid=['用户昨晚没喝酒', '用户以茶代酒', '用户今天也没碰酒'],
         known_b=['用户早就戒酒了', '用户一直不喝酒'],
         bnd=['用户没有喝{A}', '用户一直没再沾酒'],
         supp=['用户在帮年会采购清单核对饮品,把酒类一栏划掉了', '用户{T}收藏了几家清吧的探店笔记,准备推荐给朋友',
               '用户{T}看了个讲酒精代谢的视频', '用户包里常备着解酒药,是替老板拿的'],
         mention=['年会酒水单里的{A}换了供应商', '用户{T}帮同事搬过一箱{A}', '用户刷到一条{A}的测评视频']),
    dict(key='alcohol', lang=EN, A=['a glass of baijiu', 'beer', 'a cocktail'],
         known=['The user is allergic to alcohol', 'The user cannot touch alcohol',
                'Alcohol makes the user flush'],
         act=['The user drank {A} last night', 'The user just had {A}', 'The user drank {A} at the dinner'],
         avoid=['The user did not drink last night', 'The user substituted tea for alcohol',
                'The user has not touched alcohol today either'],
         known_b=['The user quit drinking long ago', 'The user never drinks'],
         bnd=['The user did not have {A}', 'The user has not drunk since quitting'],
         supp=['The user is checking the party drink list and struck the alcohol line {T}',
               'The user saved lounge-bar notes to recommend to friends {T}',
               'The user watched a video about alcohol metabolism {T}'],
         mention=['The party menu swapped the {A} supplier', 'The user carried a case of {A} for a coworker {T}',
                  'A {A} review video popped up in the feed {T}']),
    dict(key='vegetarian', lang=ZH, A=['红烧肉', '牛肉汉堡', '烤羊排'],
         known=['用户是素食者', '用户吃素五年了', '用户不碰任何肉类'],
         act=['用户昨天吃了{A}', '用户聚餐时点了{A}', '用户刚才吃了{A}'],
         avoid=['用户昨天没吃肉', '用户把{A}换成了豆腐版本', '用户今天也没碰荤菜'],
         known_b=['用户吃素已经很多年', '用户早就开始吃素'],
         bnd=['用户没有点{A}', '用户一直没再吃肉'],
         supp=['用户{T}整理了一轮素食菜谱,存了十几道', '用户在给家人选餐厅,自己看的是素菜档口',
               '用户{T}给阳台的几盆香草浇了水', '用户办了张素食自助的次卡'],
         mention=['食堂的{A}窗口排起了长队', '用户{T}帮同事捎过一次{A}', '公司聚餐订的桌上有{A}']),
    dict(key='vegetarian', lang=EN, A=['braised pork belly', 'a beef burger', 'lamb chops'],
         known=['The user is vegetarian', 'The user has eaten no meat for five years',
                'The user avoids all meat'],
         act=['The user ate {A} yesterday', 'The user ordered {A} at the dinner', 'The user just ate {A}'],
         avoid=['The user did not eat meat yesterday', 'The user swapped {A} for the tofu version',
                'The user skipped meat today as well'],
         known_b=['The user has been vegetarian for years', 'The user quit meat long ago'],
         bnd=['The user did not order {A}', 'The user has not eaten meat since'],
         supp=['The user saved a dozen vegetarian recipes {T}', 'The user was picking a restaurant for family and browsed the veggie section',
               'The user watered the balcony herbs {T}', 'The user got a veggie-buffet punch card'],
         mention=['The {A} counter had a long line at the canteen', 'The user picked up {A} for a coworker once {T}',
                  'The team dinner table came with {A}']),
    dict(key='sugar', lang=ZH, A=['奶茶', '提拉米苏', '全糖气泡水'],
         known=['用户在控糖,医生叮嘱忌糖', '用户血糖偏高要忌甜', '用户正在严格控糖'],
         act=['用户昨天喝了{A}', '用户下午吃了{A}', '用户刚才来了一份{A}'],
         avoid=['用户昨天没碰甜的', '用户点的都是无糖版本', '用户今天也没吃甜食'],
         known_b=['用户控糖已经三个月', '用户早就开始控糖'],
         bnd=['用户没有点{A}', '用户一直没再吃甜的'],
         supp=['用户{T}研究了一轮代糖烘焙配方', '用户在帮同事选无糖的生日蛋糕',
               '用户{T}给爸妈寄了两盒无糖点心', '用户办了张轻食店的储值卡'],
         mention=['便利店的{A}摆上了第二排货架', '用户{T}路过新开的甜品店,拍了张门头', '奶茶店{T}推出了季节新品']),
    dict(key='sugar', lang=EN, A=['bubble tea', 'tiramisu', 'full-sugar soda'],
         known=['The user is on a strict low-sugar plan', 'The user must avoid sweets per doctor orders',
                'The user is cutting sugar'],
         act=['The user had {A} yesterday', 'The user ate {A} in the afternoon', 'The user just got {A}'],
         avoid=['The user did not have sweets yesterday', 'The user ordered sugar-free versions',
                'The user skipped desserts today as well'],
         known_b=['The user has been cutting sugar for months', 'The user started the low-sugar plan long ago'],
         bnd=['The user did not order {A}', 'The user has not had sweets since'],
         supp=['The user researched sugar-substitute baking recipes {T}', 'The user is picking a sugar-free birthday cake for a coworker',
               'The user mailed their parents sugar-free snacks {T}', 'The user topped up a light-meal store card'],
         mention=['The convenience store added a second shelf of {A}', 'The user walked past a new dessert shop and snapped the storefront {T}',
                  'The tea shop launched a seasonal item {T}']),
    dict(key='gluten', lang=ZH, A=['韭菜盒子', '意面', '普通面包'],
         known=['用户对麸质过敏', '用户乳糜泻要无麸质饮食', '用户吃面食就过敏'],
         act=['用户昨天吃了{A}', '用户午餐点了{A}', '用户刚才吃了{A}'],
         avoid=['用户昨天没吃面食', '用户只吃无麸质面包', '用户今天也没碰小麦'],
         known_b=['用户确诊乳糜泻已经两年', '用户一直对麸质过敏'],
         bnd=['用户没有点{A}', '用户没让上含麸质的主食'],
         supp=['用户{T}在研究无麸质烘焙的配方', '用户收藏了一家做无麸质面包的店',
               '用户{T}把常去的意面店从收藏夹移到了「不能去」分组', '用户在留意超市的无麸质专区上新'],
         mention=['面包房的{A}进了新批次', '用户{T}帮同事捎过一次{A}', '部门订下午茶时统计到有人点{A},用户没点']),
    dict(key='gluten', lang=EN, A=['chive pockets', 'regular pasta', 'normal bread'],
         known=['The user has a gluten allergy', 'The user has celiac disease',
                'Wheat makes the user sick'],
         act=['The user ate {A} yesterday', 'The user ordered {A} for lunch', 'The user just had {A}'],
         avoid=['The user did not eat wheat yesterday', 'The user only eats gluten-free bread',
                'The user skipped gluten today as well'],
         known_b=['The user was diagnosed with celiac two years ago', 'The user has always avoided gluten'],
         bnd=['The user did not order {A}', 'The user skipped the gluten staples'],
         supp=['The user is researching gluten-free baking recipes {T}', 'The user saved a gluten-free bakery to their list',
               'The user moved their usual pasta place to a "cannot go" folder {T}', 'The user tracks new gluten-free aisles at the supermarket'],
         mention=['The bakery restocked {A} with a new batch', 'The user picked up {A} for a coworker once {T}',
                  'The team tea-order tally listed {A}; the user skipped it']),
    dict(key='smoke', lang=ZH, A=['烟', '一根烟', '电子烟'],
         known=['用户戒烟好几个月了', '用户戒烟半年了', '用户早就把烟戒了'],
         act=['用户昨天又抽上{A}了', '用户刚才抽了{A}', '用户聚会时抽了{A}'],
         avoid=['用户一直没有复吸', '用户昨天也没抽烟', '用户到现在都没再碰{A}'],
         known_b=['用户戒烟很久了', '用户把烟戒掉一年多了'],
         bnd=['用户没有复吸', '用户一直没再抽{A}'],
         supp=['用户{T}在给家里挑空气净化器,看了半天测评', '用户把工位上的烟灰缸收进抽屉吃灰',
               '用户{T}帮同事跑了趟腿买打火机,自己没用上', '用户给小区提了条意见,希望吸烟区离儿童区远一点'],
         mention=['公司在推楼道禁烟周', '用户{T}看了部讲戒烟门诊的纪录片', '便利店把{A}的货架挪到了门口']),
    dict(key='smoke', lang=EN, A=['cigarettes', 'a cigarette', 'the vape'],
         known=['The user quit smoking months ago', 'The user has been smoke-free for half a year',
                'The user gave up smoking long ago'],
         act=['The user started smoking {A} again yesterday', 'The user just smoked {A}',
              'The user smoked {A} at the party'],
         avoid=['The user has not relapsed', 'The user did not smoke yesterday either',
                'The user has not touched {A} since quitting'],
         known_b=['The user quit smoking over a year ago', 'The user has been off cigarettes for ages'],
         bnd=['The user has not relapsed', 'The user has not smoked {A} since'],
         supp=['The user spent {T} comparing air purifier reviews', 'The user stashed the desk ashtray into a drawer',
               'The user once bought a lighter for a coworker {T}', 'The user suggested moving the smoking area away from the kids zone'],
         mention=['The office is running a smoke-free week', 'The user watched a documentary about cessation clinics {T}',
                  'The store moved the {A} shelf to the entrance']),
    dict(key='barber', lang=ZH, A=['阿杰', 'Andy', 'Tony'],
         known=['用户不找{A}剪发了', '用户已经不在{A}那里剪发了', '用户再也没去{A}那剪过'],
         act=['用户昨天又去找{A}剪了头发', '用户刚预约了{A}', '用户上次还是{A}剪的'],
         avoid=['用户昨天没去找{A}', '用户现在固定找别的发型师', '用户一直没再约{A}'],
         known_b=['用户换掉{A}很久了', '用户早就不找{A}了'],
         bnd=['用户没有再约{A}', '用户一直没回去找{A}'],
         supp=['用户{T}给自己买了套理发工具,打算自己修刘海', '用户收藏了几个发型博主的视频',
               '用户{T}把理发店的会员卡余额查了一遍', '用户在研究适合自己脸型的短发样式'],
         mention=['用户常去的那家理发店换了新价目表', '发型师{A}上了本地的探店推荐', '理发店大堂{T}换了新的等待沙发']),
    dict(key='barber', lang=EN, A=['Jeremy', 'Kevin', 'Tony'],
         known=['The user no longer gets haircuts from {A}', 'The user stopped seeing {A} months ago',
                'The user never went back to {A}'],
         act=['The user went back to {A} for a cut yesterday', 'The user just booked {A}',
              'The user last had {A} cut their hair'],
         avoid=['The user did not see {A} yesterday', 'The user now sees a different stylist regularly',
                'The user has not booked {A} since'],
         known_b=['The user switched away from {A} long ago', 'The user stopped seeing {A} ages ago'],
         bnd=['The user has not rebooked {A}', 'The user never went back to {A}'],
         supp=['The user bought a home hair-trimming kit {T}', 'The user saved several hairstylist videos',
               'The user checked their salon gift-card balance {T}', 'The user is researching short cuts for their face shape'],
         mention=['The salon the user visits refreshed its price list', 'Stylist {A} made a local recommendation list',
                  'The salon lounge got new waiting sofas {T}']),
]

TT = {ZH: ['上周', '昨天', '前两天', '上周末', '最近'],
      EN: ['last week', 'yesterday', 'a few days ago', 'last weekend', 'recently']}

# carry over v5's process fixes: 4th act for ZH neg_true space + widened avoids
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
                rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                             'gold': {'conflict': gold_for(kind, target_rng)},
                             '_meta': {'cat': 'negation', 'kind': kind, 'lang': lang, 'labelset': ls}})
                made += 1
            assert made == n_lang, (kind, lang, made, n_lang)
    return rows


# --- §3.2 unrelated controls for the neg block (exact per-family quotas) ------
UNREL_SUPP_SHARE = 0.75  # per family supp(同话题无关补充) >= 30% required; we fix 75%


def neg_unrel_quota(total, fams_zh, fams_en, lang_mix):
    """Allocate exact per-(family, lang, angle) quotas; returns
    {(fkey, lang): {'supp': n, 'mention': n}}."""
    n_en = round(total * lang_mix[EN])
    per_lang = {ZH: total - n_en, EN: n_en}
    out = {}
    for lang, n_lang in per_lang.items():
        fams = fams_zh if lang == ZH else fams_en
        base, rem = divmod(n_lang, len(fams))
        for i, f in enumerate(fams):
            n = base + (1 if i < rem else 0)
            s = round(n * UNREL_SUPP_SHARE)
            out[(f['key'], lang)] = {'supp': s, 'mention': n - s}
    return out


def make_neg_unrel_rows(rng, total, lang_mix, seen, label_rng, target_rng, ls_pool):
    quota = neg_unrel_quota(total, [f for f in NEG_FAMILIES if f['lang'] == ZH],
                            [f for f in NEG_FAMILIES if f['lang'] == EN], lang_mix)
    rows = []
    for (fkey, lang), q in quota.items():
        fam = next(f for f in NEG_FAMILIES if f['key'] == fkey and f['lang'] == lang)
        known_pool = fam['known'] + fam.get('known_b', [])
        for angle, n in q.items():
            made, guard = 0, 0
            while made < n and guard < n * 2000:
                guard += 1
                known = rng.choice(known_pool).format(A=rng.choice(fam['A']))
                new = rng.choice(fam[angle]).format(A=rng.choice(fam['A']), T=rng.choice(TT[lang]))
                state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                if state in seen:
                    continue
                seen.add(state)
                ls = next_ls(ls_pool)
                rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                             'gold': {'conflict': gold_for('neg_unrelated', target_rng)},
                             '_meta': {'cat': 'negation', 'kind': 'neg_unrelated', 'lang': lang,
                                       'labelset': ls, 'angle': angle, 'family': fkey}})
                made += 1
            assert made == n, (fkey, lang, angle, made, n)
    return rows


# --- block B: known-side one-step perturbation (v5's, shape values swapped) ---
SHAPE_CATS = [
    cat_v5('shape_meeting', ['周三上午十点', '周四下午四点', '周五早上九点', '周二下午两点'],
           known=['用户的周会固定在{A}', '用户例会时间是{A}'],
           conflict=['周会已经改到{B}了', '用户把例会挪到了{B}'],
           compat=['用户{A}开完会都会整理纪要', '会议纪要里写的是{A}的议程'],
           consid=['用户在考虑把周会挪到{B}', '用户在纠结要不要改到{B}'],
           unrelated=['会议室的投影仪{T}换了新的', '行政在统计下周的订水数量', '公司在统计各会议室的使用率']),
    cat_v5('shape_meeting_en', ['Wednesday 10am', 'Thursday 4pm', 'Friday 9am', 'Tuesday 2pm'],
           known=['The user has a standing meeting at {A}', 'The weekly sync is at {A}'],
           conflict=['The meeting has moved to {B}', 'The user moved the sync to {B}'],
           compat=['The user takes notes right after the {A} meeting', 'The agenda still lists {A} items'],
           consid=['The user is considering moving the sync to {B}', 'The user is debating a {B} slot'],
           unrelated=['The meeting-room projector was replaced {T}', 'Office is counting water orders',
                      'The company is tracking meeting-room utilization']),
    cat_v5('shape_streaming', ['视频会员', '音乐会员', '网盘会员'],
           known=['用户没有订阅{A}', '用户没开{A}'],
           conflict=['用户昨晚开通了{A}', '用户刚续费了{A}'],
           compat=['用户在{A}的活动页领过优惠券', '用户收藏过{A}的片单'],
           consid=['用户在想要不要开{A}', '用户有点心动{A}的年卡'],
           unrelated=['视频平台的会员价{T}调整了', '家里的宽带{T}升级到了千兆', '电视系统{T}推送了更新']),
    cat_v5('shape_streaming_en', ['a video membership', 'a music subscription', 'cloud storage plan'],
           known=['The user has not subscribed to {A}', 'The user never signed up for {A}'],
           conflict=['The user signed up for {A} last night', 'The user just renewed {A}'],
           compat=['The user once grabbed a coupon from the {A} promo page', 'The user saved playlists from {A}'],
           consid=['The user is considering {A}', 'The user is tempted by the {A} annual plan'],
           unrelated=['The streaming platform adjusted membership prices {T}', 'The home broadband got upgraded {T}',
                      'The TV system pushed an update {T}']),
    cat_v5('shape_commute', ['轻轨', '公交', '骑车', '开车'],
           known=['用户通勤坐{A}', '用户上下班靠{A}'],
           conflict=['用户通勤改成{B}了', '用户现在上下班改{B}了'],
           compat=['用户{A}上会听播客', '用户的{A}卡刚充了值'],
           consid=['用户在考虑改成{B}通勤', '用户在纠结要不要换{B}'],
           unrelated=['地铁沿线的商场{T}办了市集', '早高峰的公交{T}加了班次', '公司楼下的共享单车换新了一批']),
    cat_v5('shape_commute_en', ['the light rail', 'the bus', 'a bike', 'driving'],
           known=['The user commutes by {A}', 'The user gets to work via {A}'],
           conflict=['The user switched their commute to {B}', 'The user now commutes by {B}'],
           compat=['The user listens to podcasts on {A}', 'The user just topped up the {A} card'],
           consid=['The user is considering switching to {B}', 'The user is debating {B} for the commute'],
           unrelated=['A market popped up along the line {T}', 'Morning buses got extra runs {T}',
                      'The shared bikes downstairs were rotated out {T}']),
    cat_v5('shape_delivery', ['单位前台', '家门口的驿站', '物业代收点'],
           known=['用户的默认收货地址是{A}', '用户快递默认放{A}'],
           conflict=['用户把默认收货地址改成了新驿站的地址', '用户的收货偏好换地方了,现在写的是公司楼下的柜子'],
           compat=['用户让{A}的同事帮忙代收过一个包裹', '用户的退货件也是走{A}'],
           consid=['用户在想要不要把收货地址改到公司附近', '用户在考虑换个代收点'],
           unrelated=['快递柜{T}搞了寄件优惠', '小区驿站{T}延长了营业时间', '物业{T}贴了新的代收须知']),
    cat_v5('shape_delivery_en', ['the office front desk', 'the pickup point downstairs', 'the doorman'],
           known=['The user has packages delivered to {A}', 'The default delivery spot is {A}'],
           conflict=['The user changed the default delivery address to a new pickup point',
                     'The delivery preference now points to a different locker address'],
           compat=['A colleague at {A} once signed for a parcel for the user', 'Returns also go through {A}'],
           consid=['The user is considering moving deliveries near the office', 'The user is weighing a new pickup point'],
           unrelated=['The parcel locker ran a shipping discount {T}', 'The pickup point extended its hours {T}',
                      'Property management posted new pickup rules {T}']),
    cat_v5('shape_injury', ['膝盖半月板损伤', '脚踝扭伤', '腰肌劳损'],
           known=['用户{A},医生要求静养一个月', '用户{A}还在恢复期'],
           conflict=['用户周末打算去踢球,不顾{A}的医嘱', '用户不等{A}好利索就去跑步了'],
           compat=['用户{A}期间在 gym 做上肢训练', '用户按医嘱复查了{A}'],
           consid=['用户在想要不要提前结束{A}的静养', '用户有点想提前恢复训练'],
           unrelated=['康复科{T}来了位新治疗师', '运动医学的科普号{T}更新了一篇推文', '体育馆{T}检修了器械']),
    cat_v5('shape_injury_en', ['a torn meniscus', 'a twisted ankle', 'a strained back'],
           known=['The user has {A} and the doctor ordered a month of rest', 'The user is recovering from {A}'],
           conflict=['The user plans to play soccer this weekend against medical advice about {A}',
                     'The user went running before the {A} healed'],
           compat=['The user does upper-body gym work during the {A} recovery', 'The user had the {A} reviewed as told'],
           consid=['The user is considering cutting the {A} rest short', 'The user is tempted to resume training early'],
           unrelated=['A new therapist joined the rehab clinic {T}', 'The sports-medicine account posted a new article {T}',
                      'The gym inspected its equipment {T}']),
]

# supp (同话题无关补充, user-action angle) per shape family for cf_unrelated
SHAPE_SUPP = {
    'shape_meeting': ['用户{T}在周会纪要里补了一条待办', '用户给会议室投影换了根HDMI线', '用户把例会材料的模板更新了一版'],
    'shape_meeting_en': ['The user added a follow-up item to the meeting notes {T}', 'The user swapped the meeting-room projector cable',
                         'The user refreshed the sync agenda template'],
    'shape_streaming': ['用户{T}整理了一下视频App的收藏夹', '用户给电视会员的支付方式换了张卡', '用户把片单分享给了同事'],
    'shape_streaming_en': ['The user tidied up their watchlist {T}', 'The user changed the card tied to the TV membership',
                           'The user shared a playlist with a coworker'],
    'shape_commute': ['用户{T}给通勤包换了根背带', '用户在App里查了下一班的时刻表', '用户把通勤耳机换了副新的'],
    'shape_commute_en': ['The user replaced the strap on their commute bag {T}', 'The user checked the timetable in the transit app',
                         'The user got a new pair of commute earbuds'],
    'shape_delivery': ['用户{T}在快递App里开了会员', '用户给门禁卡充了值', '用户把纸箱攒了一摞留着寄东西'],
    'shape_delivery_en': ['The user got a membership in the courier app {T}', 'The user topped up the building access card',
                          'The user stacked some boxes for future shipping'],
    'shape_injury': ['用户{T}给护具换了副新的', '用户把复查预约改到了下周三', '用户收藏了一组合康复动作视频'],
    'shape_injury_en': ['The user got a new brace {T}', 'The user moved the follow-up appointment to next Wednesday',
                        'The user saved a rehab-exercise video collection'],
}

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
                rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                             'gold': {'conflict': gold_for(kind, target_rng)},
                             '_meta': {'cat': 'perturb', 'kind': kind, 'lang': lang, 'labelset': ls}})
                made += 1
            assert made == n_lang, (kind, lang, made, n_lang)
    return rows


def make_cf_unrel_rows(rng, cats_shape, total, lang_mix, seen, label_rng, target_rng, ls_pool):
    """cf-unrelated controls: same shape family + no logical conflict.
    supp/mention 75%/25% per family+language (same construction as neg)."""
    n_en = round(total * lang_mix[EN])
    per_lang = {ZH: total - n_en, EN: n_en}
    rows = []
    for lang, n_lang in per_lang.items():
        fams = [c for c in cats_shape if c['lang'] == lang]
        base, rem = divmod(n_lang, len(fams))
        for i, c in enumerate(fams):
            n = base + (1 if i < rem else 0)
            s = round(n * UNREL_SUPP_SHARE)
            for angle, n_angle in (('supp', s), ('mention', n - s)):
                made, guard = 0, 0
                while made < n_angle and guard < n_angle * 2000:
                    guard += 1
                    a, b = rng.sample(c['vals'], 2)
                    known = rng.choice(c['known']).format(A=a)
                    if angle == 'supp':
                        new = rng.choice(SHAPE_SUPP[c['key']]).format(A=a, T=rng.choice(TT[lang]))
                    else:
                        new = rng.choice(c['unrelated']).format(A=a, B=b, T=rng.choice(TT[lang]))
                    state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                    if state in seen:
                        continue
                    seen.add(state)
                    ls = next_ls(ls_pool)
                    rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                                 'gold': {'conflict': gold_for('cf_unrelated', target_rng)},
                                 '_meta': {'cat': 'perturb', 'kind': 'cf_unrelated', 'lang': lang,
                                           'labelset': ls, 'angle': angle, 'family': c['key']}})
                    made += 1
                assert made == n_angle, (c['key'], lang, angle, made, n_angle)
    return rows


# --- §3.3 negation five-shape families (+80 rows each, shapes not literals) ---
# Shape semantics -> kind mapping (targets already in the §3.4 tier table):
#   pet    : negated possession + the owned thing shows up   -> 48 neg_true / 32 neg_false_boundary
#   never  : "never X" + daily X behavior                    -> 48 neg_true / 32 neg_false
#   relapse: quit-X + resumed X                              -> 48 neg_true / 32 neg_false
#   dneg   : negated preference + sustained negation         -> 80 neg_false_boundary
#   nocar  : no car + compatible commute                     -> 24 neg_true / 40 neg_false / 16 neg_false_boundary
# None of the rendered strings may equal a case sentence (§3.6 assert enforces).

PETS_ZH = ['狗', '猫', '鹦鹉', '仓鼠', '兔子']
# NB: no template may render a verbatim case sentence -- e.g. '用户的{P}昨天打了疫苗'
# with P=狗 IS the negation-case new, hence 这周/前天 below (the §3.6 assert backstops).
PET_NEWS_ZH = ['用户的{P}这周打了疫苗', '用户的{P}上周末走丢了', '用户的{P}刚生了幼崽',
               '用户带{P}去做了体检', '用户的{P}把沙发抓花了']
PET_BND_ZH = ['用户帮邻居遛过{P}', '用户去宠物咖啡馆撸过猫', '用户给楼下的流浪{P}搭了个窝',
              '用户朋友的{P}寄养到用户家两天']
PETS_EN_P = ['dog', 'cat', 'parrot', 'hamster', 'rabbit']
PETS_EN_A = ['a dog', 'a cat', 'a parrot', 'a hamster', 'a rabbit']
PET_NEWS_EN = ["The user's {P} got vaccinated yesterday", "The user's {P} went missing last weekend",
               "The user's {P} just had a litter", 'The user took the {P} in for a checkup',
               "The user's {P} scratched up the sofa"]
PET_BND_EN = ["The user once walked a neighbor's dog", 'The user petted cats at a cat café',
              'The user built a shelter for a stray {P}', "A friend's {P} stayed over for two days"]

NEVER_CATS_ZH = [('咖啡', ['拿铁', '美式', '冷萃', '摩卡']), ('奶茶', ['珍珠奶茶', '杨枝甘露', '烧仙草']),
                 ('碳酸饮料', ['可乐', '雪碧', '气泡水']), ('酒', ['啤酒', '红酒', '清酒'])]
NEVER_KNOWN_ZH = ['用户从来不碰{A}', '用户从不沾{A}', '用户基本不喝{A}']
NEVER_TRUE_ZH = ['用户每天早上都来一杯{S}', '用户这周天天在喝{S}', '用户每天下午都要点{S}']
NEVER_FALSE_ZH = ['用户平时只喝豆浆和椰汁', '用户办公室常备的是矿泉水和酸奶', '用户{T}点过果茶和气泡水']
NEVER_CATS_EN = [('coffee', ['latte', 'americano', 'cold brew', 'mocha']),
                 ('bubble tea', ['brown sugar milk tea', 'taro milk tea']),
                 ('soda', ['cola', 'lemon-lime soda', 'sparkling water']),
                 ('alcohol', ['beer', 'wine', 'sake'])]
NEVER_KNOWN_EN = ['The user never touches {A}', 'The user never drinks {A} at all', 'The user basically never drinks {A}']
NEVER_TRUE_EN = ['The user starts every morning with a {S}', 'The user has been drinking {S} every day this week',
                 'The user orders a {S} every afternoon']
NEVER_FALSE_EN = ['The user usually drinks soy milk and coconut water', 'The office stash is bottled water and yogurt',
                  'The user ordered fruit tea and sparkling water {T}']

RELAPSE_ZH = [('奶茶', ['点了一杯奶茶', '连喝了三天奶茶', '又开了个新口味']),
              ('游戏', ['通宵打了一晚上游戏', '重新下载了游戏', '连着打了三天排位']),
              ('夜宵', ['半夜点了烧烤', '连着一周都在吃夜宵', '又吃起了炸鸡']),
              ('网购', ['又下单了一堆东西', '这周清空了三次购物车', '连着三天都在收快递'])]
RELAPSE_KNOWN_ZH = ['用户已经把{A}戒了', '用户戒{A}有一阵子了', '用户{A}都戒了大半年']
RELAPSE_AVOID_ZH = ['用户一直没再碰{A}', '用户{T}也没碰{A}', '用户早就完全断了{A}']
RELAPSE_EN = [('bubble tea', ['ordered a bubble tea again', 'has had bubble tea three days straight', 'tried a new flavor']),
              ('gaming', ['pulled an all-night gaming session', 'reinstalled the game', 'ranked three nights in a row']),
              ('late-night snacks', ['ordered BBQ at midnight', 'has been snacking every night this week', 'went back to fried chicken']),
              ('online shopping', ['placed another big order', 'emptied the cart three times this week', 'has been getting parcels daily'])]
RELAPSE_KNOWN_EN = ['The user quit {A} three months ago', 'The user has been off {A} for a while', 'The user gave up {A} months ago']
RELAPSE_AVOID_EN = ["The user hasn't touched {A} since", 'The user skipped {A} again {T}', 'The user cut {A} out completely']

DNEG_ITEMS_ZH = [('科幻电影', ['看', '点开']), ('恐怖片', ['看', '点开']), ('说唱', ['听']),
                 ('香菜', ['吃', '加']), ('摇滚现场', ['去'])]
DNEG_KNOWN_ZH = ['用户对{A}不感兴趣', '用户一直不爱{A}', '用户谈不上喜欢{A}']
DNEG_NEW_ZH = ['用户从来不{V}{A}', '用户也没{V}过{A}', '用户连{A}相关的节目都不点开', '用户从来没{V}过{A}']
DNEG_ITEMS_EN = [('sci-fi movies', 'watch', 'watched'), ('horror films', 'watch', 'watched'),
                 ('rap music', 'listen to', 'listened to'), ('cilantro', 'eat', 'eaten'),
                 ('rock shows', 'attend', 'attended')]
DNEG_KNOWN_EN = ["The user isn't into {A}", 'The user has never liked {A}', "The user wouldn't call themselves a fan of {A}"]
DNEG_NEW_EN = ['The user never {V} {A}', "The user hasn't {V2} {A} either", 'The user never {V2} {A}, not once']

NOCAR_KNOWN_ZH = ['用户名下没车', '用户家里没买车', '用户还没买车', '用户到现在都没买过车']
NOCAR_TRUE_ZH = ['用户昨天提了辆新车', '用户刚订了台代步车', '用户家里刚添了辆车']
NOCAR_FALSE_ZH = ['用户每天搭地铁通勤', '用户上下班都坐公交', '用户{T}骑共享单车去上班', '用户平时打车上下班']
NOCAR_BND_ZH = ['用户周末租了辆车全家出游', '用户借朋友的车搬过一次家', '用户考了驾照但一直没买车', '用户{T}开过一次共享汽车']
NOCAR_KNOWN_EN = ['The user does not own a car', "The user's household has no car", 'The user has not bought a car yet',
                  'The user still has no car']
NOCAR_TRUE_EN = ['The user just picked up a new car yesterday', 'The user just ordered a runabout', 'The family just added a car']
NOCAR_FALSE_EN = ['The user commutes by subway every day', 'The user takes the bus to work',
                  'The user bikes to work {T}', 'The user usually takes a ride-hail to work']
NOCAR_BND_EN = ['The user rented a car for a family trip last weekend', 'The user borrowed a friend\'s car once for moving',
                "The user has a license but has never bought a car", 'The user drove a shared car once {T}']


def render_neg5(rng, shape, side, lang):
    """side in {'true','false','bnd'}; returns (known, new)."""
    if shape == 'pet':
        if lang == ZH:
            p = rng.choice(PETS_ZH)
            known = rng.choice(['用户没有养' + p, '用户家里没养' + p, '用户从来没养过' + p])
            new = rng.choice({'true': PET_NEWS_ZH, 'bnd': PET_BND_ZH}[side]).format(P=p)
        else:
            i = rng.randrange(len(PETS_EN_P))
            known = rng.choice([f'The user does not have {PETS_EN_A[i]}',
                                f"The user doesn't keep {PETS_EN_A[i]}",
                                f'The user has never owned {PETS_EN_A[i]}'])
            new = rng.choice({'true': PET_NEWS_EN, 'bnd': PET_BND_EN}[side]).format(P=PETS_EN_P[i])
        return known, new
    if shape == 'never':
        cats = NEVER_CATS_ZH if lang == ZH else NEVER_CATS_EN
        a, subs = rng.choice(cats)
        s = rng.choice(subs)
        t = rng.choice(TT[lang])
        if lang == ZH:
            known = rng.choice(NEVER_KNOWN_ZH).format(A=a)
            new = rng.choice({'true': NEVER_TRUE_ZH, 'false': NEVER_FALSE_ZH}[side]).format(S=s, T=t)
        else:
            known = rng.choice(NEVER_KNOWN_EN).format(A=a)
            new = rng.choice({'true': NEVER_TRUE_EN, 'false': NEVER_FALSE_EN}[side]).format(S=s, T=t)
        return known, new
    if shape == 'relapse':
        if lang == ZH:
            a, verbs = rng.choice(RELAPSE_ZH)
            known = rng.choice(RELAPSE_KNOWN_ZH).format(A=a)
            # verb phrases carry their own time words (昨天/连着三天/又), no extra T
            new = ('用户' + rng.choice(verbs)) if side == 'true' else rng.choice(RELAPSE_AVOID_ZH).format(A=a, T=rng.choice(TT[lang]))
        else:
            a, verbs = rng.choice(RELAPSE_EN)
            known = rng.choice(RELAPSE_KNOWN_EN).format(A=a)
            new = ('The user ' + rng.choice(verbs)) if side == 'true' \
                else rng.choice(RELAPSE_AVOID_EN).format(A=a, T=rng.choice(TT[lang]))
        return known, new
    if shape == 'dneg':
        if lang == ZH:
            a, verbs = rng.choice(DNEG_ITEMS_ZH)
            known = rng.choice(DNEG_KNOWN_ZH).format(A=a)
            new = rng.choice(DNEG_NEW_ZH).format(V=rng.choice(verbs), A=a)
        else:
            a, v1, v2 = rng.choice(DNEG_ITEMS_EN)
            known = rng.choice(DNEG_KNOWN_EN).format(A=a)
            new = rng.choice(DNEG_NEW_EN).format(V=v1, V2=v2, A=a)
        return known, new
    if shape == 'nocar':
        if lang == ZH:
            known = rng.choice(NOCAR_KNOWN_ZH)
            new = rng.choice({'true': NOCAR_TRUE_ZH, 'false': NOCAR_FALSE_ZH, 'bnd': NOCAR_BND_ZH}[side]).format(T=rng.choice(TT[lang]))
        else:
            known = rng.choice(NOCAR_KNOWN_EN)
            new = rng.choice({'true': NOCAR_TRUE_EN, 'false': NOCAR_FALSE_EN, 'bnd': NOCAR_BND_EN}[side]).format(T=rng.choice(TT[lang]))
        return known, new
    raise ValueError(shape)


# side quotas per shape: {side: (n_zh, n_en)}
NEG5_SIDES = {
    'pet': {'true': (24, 24), 'bnd': (16, 16)},
    'never': {'true': (24, 24), 'false': (16, 16)},
    'relapse': {'true': (24, 24), 'false': (16, 16)},
    'dneg': {'bnd': (40, 40)},
    'nocar': {'true': (12, 12), 'false': (20, 20), 'bnd': (8, 8)},
}
NEG5_KIND = {'true': 'neg_true', 'false': 'neg_false', 'bnd': 'neg_false_boundary'}


def make_neg5_rows(rng, seen, label_rng, target_rng, ls_pool):
    rows = []
    for shape, sides in NEG5_SIDES.items():
        for side, (n_zh, n_en) in sides.items():
            for lang, n_lang in ((ZH, n_zh), (EN, n_en)):
                made, guard = 0, 0
                while made < n_lang and guard < n_lang * 2000:
                    guard += 1
                    known, new = render_neg5(rng, shape, side, lang)
                    state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                    if state in seen:
                        continue
                    seen.add(state)
                    ls = next_ls(ls_pool)
                    kind = NEG5_KIND[side]
                    rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                                 'gold': {'conflict': gold_for(kind, target_rng)},
                                 '_meta': {'cat': 'negation', 'kind': kind, 'lang': lang,
                                           'labelset': ls, 'shape': shape}})
                    made += 1
                assert made == n_lang, (shape, side, lang, made, n_lang)
    return rows


def main():
    rng = random.Random(SEED_V6)
    label_rng = random.Random(SEED_V6 + 1)
    target_rng = random.Random(SEED_V6 + 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # acceptance sets must never leak into train (V6 §3.1/§8)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: same extraction as v4/v5 (exact state match), name-rotated too
    ls_pool = make_ls_pool(label_rng, 14500)
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
        r['questions'] = question_for(r['_meta']['labelset'], ZH, label_rng)  # v4 kept zh instructions on MNLI

    # §3.1: value swaps + HC literal rewrite, THEN build the synthetic blocks
    cats = swap_vals(patch_categories())
    rewrite_hc()
    train_pool = [c for c in cats if not c['holdout']]
    soft_rows = make_soft_rows_v6(rng, train_pool,
                                  {'true': 1800, 'compat': 1440, 'consid': 1080, 'unrelated': 1080},
                                  {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    hc_rows = make_hc_rows_v6(rng, 300, 300, {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng,
                              patch_hc(), ls_pool)

    # negation block: 800 base + 400 five-shape + 400 unrelated = 1600
    neg_rows = make_neg_rows(rng, {'neg_true': 400, 'neg_false': 300, 'neg_false_boundary': 100},
                             {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    neg5_rows = make_neg5_rows(rng, seen, label_rng, target_rng, ls_pool)
    neg_unrel_rows = make_neg_unrel_rows(rng, 400, {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)

    # perturbation block: 600 + 300 unrelated = 900
    shape_cats = []
    for c in SHAPE_CATS:
        shape_cats.append(dict(c, lang=EN if c['key'].endswith('_en') else ZH))
    cf_rows = make_cf_rows(rng, train_pool, shape_cats,
                           {'cf_true': 300, 'cf_false': 300},
                           {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    cf_unrel_rows = make_cf_unrel_rows(rng, shape_cats, 300, {ZH: 0.72, EN: 0.28}, seen,
                                       label_rng, target_rng, ls_pool)

    v6_train = mnli_rows + soft_rows + hc_rows + neg_rows + neg5_rows + neg_unrel_rows + cf_rows + cf_unrel_rows
    rng.shuffle(v6_train)

    # ---- §3.5 hard invariants -------------------------------------------------
    assert len(v6_train) == 14500, len(v6_train)
    n_true = sum(1 for r in v6_train if r['gold']['conflict']['label'] == 'true')
    n_false = len(v6_train) - n_true
    states = [r['state'] for r in v6_train]
    assert len(set(states)) == 14500, 'duplicate states'
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        other = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, p), encoding='utf-8') if l.strip()}
        assert set(states).isdisjoint(other), f'leak into {p}'

    kind_counts = Counter(r['_meta']['kind'] for r in v6_train)
    expect_kinds = {'contradiction': 2000, 'entailment/neutral': 4000, 'true': 1800, 'compat': 1440,
                    'consid': 1080, 'unrelated': 1080, 'hardconstraint': 300, 'hc_safe': 180,
                    'hc_unrelated': 120, 'neg_true': 568, 'neg_false': 404, 'neg_false_boundary': 228,
                    'neg_unrelated': 400, 'cf_true': 300, 'cf_false': 300, 'cf_unrelated': 300}
    assert dict(kind_counts) == expect_kinds, (kind_counts, expect_kinds)

    label_mix = Counter(r['_meta']['labelset'] for r in v6_train)
    assert label_mix[5] == 2900 and label_mix[6] == 2900, label_mix
    assert sum(label_mix[s] for s in (1, 2, 3, 4)) == 14500 - 5800, label_mix
    for ls in range(1, 7):
        kinds_with = {r['_meta']['kind'] for r in v6_train if r['_meta']['labelset'] == ls}
        assert len(kinds_with) >= 3, (ls, len(kinds_with))

    ls_rng = random.Random(777)  # re-derive each row's label strings for the audit check
    for r in v6_train:
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

    # ---- §3.6 leak assert (hard invariant; case texts never enter training) ----
    assert_no_case_leak(v6_train)

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v6.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v6_train:
            meta = r.pop('_meta')
            meta['name_a'] = r['questions']['conflict']['labels']['false']
            meta['name_b'] = r['questions']['conflict']['labels']['true']
            r['meta'] = meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v6_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false / n_true:.2f})')
    print('meta.kind:', dict(sorted(Counter(r["meta"]["kind"] for r in v6_train).items())))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v6_train)))
    print('meta.labelset:', dict(sorted(Counter(r["meta"]["labelset"] for r in v6_train).items())))
    sem = sum(Counter(r["meta"]["labelset"] for r in v6_train)[s] for s in (1, 2, 3, 4))
    print(f'labelset ratio: semantic {sem / 14500:.1%}, LS5 {label_mix[5] / 14500:.1%}, LS6 {label_mix[6] / 14500:.1%}')
    shapes = Counter(r['meta'].get('shape') for r in v6_train if 'shape' in r['meta'])
    print('neg5 per-shape:', dict(sorted(shapes.items())))
    shape_by_kind = Counter((r['meta']['shape'], r['meta']['kind']) for r in v6_train if 'shape' in r['meta'])
    print('neg5 shape x kind:', {f'{s}/{k}': n for (s, k), n in sorted(shape_by_kind.items())})
    angles = Counter((r['meta']['kind'], r['meta'].get('angle')) for r in v6_train if 'angle' in r['meta'])
    print('unrel angle split:', {f'{k}/{a}': n for (k, a), n in sorted(angles.items(), key=lambda x: str(x[0]))})
    fam_angle = Counter((r['meta']['kind'], r['meta'].get('family'), r['meta'].get('angle')) for r in v6_train if 'angle' in r['meta'])
    supp_share = {}
    for (kind, fam, angle), n in fam_angle.items():
        key = (kind, fam)
        supp_share.setdefault(key, {})[angle] = n
    bad = {k: v for k, v in supp_share.items() if v.get('supp', 0) < 0.3 * sum(v.values())}
    assert not bad, f'family supp share <30%: {bad}'
    print(f'unrel per-family supp share: min {min(v["supp"] / sum(v.values()) for v in supp_share.values()):.0%}, '
          f'{len(supp_share)} families, all >=30%: {not bad}')


def assert_no_case_leak(rows):
    # ---- §3.6 leak assert (V6 新增;硬不变量) ----
    # 口径:35 条验收句(realtest_v2 CASES/NEGATION_CASES + realtest_v4 NEW_CASES)与训练语料
    # known/new 逐字段等值(不做子串 —— 子串会假阳性:某超集句式含验收句不等于同一句)。
    import sys
    from realtest_v2 import CASES, NEGATION_CASES
    from realtest_v4 import NEW_CASES

    ks, ns = set(), set()
    for (state, _e), _note in list(CASES) + list(NEGATION_CASES) + list(NEW_CASES):
        ks.add(state['known']); ns.add(state['new'])
    hits = []
    for r in rows:
        st = r['state']
        st = json.loads(st) if isinstance(st, str) else st
        if st.get('known') in ks or st.get('new') in ns:
            hits.append((st.get('known'), st.get('new')))
    assert not hits, f'验收句泄漏 {len(hits)} 行,示例: {hits[:2]}'


# v6 wrappers around v4's row builders: 6-labelset rotation + graded targets ----
def make_soft_rows_v6(rng, cats, quota, lang_mix, seen, label_rng, target_rng, ls_pool):
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
            rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                         'gold': {'conflict': gold_for(k, target_rng)},
                         '_meta': {'cat': catkey, 'kind': k, 'lang': lang, 'labelset': ls}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


def make_hc_rows_v6(rng, n_true, n_control, lang_mix, seen, label_rng, target_rng, hc_pool, ls_pool):
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
            rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                         'gold': {'conflict': gold_for(kind, target_rng)},
                         '_meta': {'cat': 'hardconstraint', 'kind': kind, 'lang': lang, 'labelset': ls}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


if __name__ == '__main__':
    main()
