# -*- coding: utf-8 -*-
"""v7 data arm (HANDOFF_NLI_V7.md §3): two missing hard-constraint shapes +
alt_praise + metric transfer, on top of the v6 recipe (pure-CE main arm).

Changes vs gen_soft_conflicts_v6.py ("照着 v6 改"):

  1. HC PET FAMILY (§3.1, +200): constraint subject = the user's PET (third
     party) -- known "user's dog is allergic to X" + BARE intent ("planning to
     swap its treats to X-flavored") / act ("fed it X yesterday") => true.
     NO defiance markers (不顾/偏要) -- the two new-10 misses were bare-intent
     hard constraints (V6 §6.4-4).  Split into per-allergen sub-families so
     known/items stay semantically coherent (a beef-allergic dog fed chicken
     would be mislabeled).  safe -> hc_safe, unrel -> hc_unrelated.
  2. HC DOCTOR FAMILY (§3.1, +200): constraint source = medical order
     (rest / no strenuous exercise / no spicy / low-salt), again BARE intent.
  3. ALT_PRAISE SHAPE (§3.2, +240 over 6 non-holdout families): hearsay praise
     of an ALTERNATIVE ("heard B is good / coworker recommends B") with no user
     state change => false (consid tier), meta.shape='alt_praise'.  Excludes
     the hair/barber domain on purpose (V6 §2.2 note: barber holdout overlaps
     the training domain since v5 -- report-only, data untouched).
  4. METRIC FAMILY (§3.3, +160, exploratory): quantified checkup metrics
     (weight / glucose / lipids / heart rate; NOT the eyeglasses domain)
     rewritten to a new value => conflict; four sides x40 with
     meta.shape='metric_shift' on the conflict side.  Tests abstract transfer
     to the degree-side val_soft errors; report must give before/after.

Carried from v6: leak clearance (value swaps + literal rewrites), 6-labelset
rotation, graded soft targets, exact quota pools, unrelated controls
(supp/mention), five-shape negation families, §3.6 leak assert.

v7 train = 6000 MNLI + 5800 synthetic (5400 + alt_praise 240 + metric 160)
         + 1000 hard-constraint (600 + pet 200 + doctor 200) + 1600 negation
         + 900 perturbation = 15300 rows.  val (1000) / val_soft (300) are NOT
         regenerated (byte-identical, directly comparable).

Deterministic: seed 20260930 (same family as v4/v5/v6; blocks consume the
stream in fixed order).
"""
import json
import os
import random
from collections import Counter

from gen_soft_conflicts_v3 import ZH, EN
from gen_soft_conflicts_v4 import OUT_DIR, patch_categories, patch_hc, render_hc
from gen_soft_conflicts_v6 import (
    SEED_V6, question_for, make_ls_pool, next_ls, TARGETS, gold_for,
    swap_vals, rewrite_hc, make_neg_rows, make_neg_unrel_rows, make_neg5_rows,
    SHAPE_CATS, make_cf_rows, make_cf_unrel_rows, make_soft_rows_v6, assert_no_case_leak)

SEED_V7 = 20260930  # == SEED from v4..v6; named locally for clarity


def cat_v5(key, vals, known, conflict, compat, consid, unrelated):
    return dict(key=key, lang=None, vals=vals, known=known, conflict=conflict,
                compat=compat, consid=consid, unrelated=unrelated, holdout=False)


# --- §3.1 HC pet family: per-allergen sub-families (coherence: items must
#     match the known allergen, so one sub-family per allergen x species) -----
PET_HC = [
    dict(key='pet_beef', lang=ZH,
         known=['用户家的狗对牛肉过敏', '用户养的狗一吃牛肉就吐'],
         items=['牛肉味的零食', '牛肉罐头', '牛柳粒磨牙棒'],
         intent=['用户打算给它换{X}', '用户想试试给它喂{X}', '用户准备下次囤点{X}', '用户在考虑给它买{X}'],
         act=['用户昨天喂了它{X}', '用户刚才给它碗里加了{X}'],
         safe=['用户给它换成了羊肉配方的粮', '用户买零食前都会把成分表看一遍', '用户把牛肉类零食都送给了同事的狗',
               '用户给它挑的都是单一蛋白的鸭肉零食', '用户给它订了一个月的低敏粮', '用户喂它之前先问了医生能不能吃',
               '用户把它的零食柜贴了过敏清单'],
         unrel=['用户在给狗窝换垫子', '用户预约了下个月的宠物体检', '用户给狗买了新的牵引绳', '用户在比较两款宠物空气净化器']),
    dict(key='pet_beef', lang=EN,
         known=["The user's dog is allergic to beef", 'The user\'s dog throws up whenever it eats beef'],
         items=['beef-flavored treats', 'a can of beef dog food', 'beef jerky chews'],
         intent=['The user is planning to switch it to {X}', 'The user wants to try feeding it {X}',
                 'The user is going to stock up on {X}', 'The user is considering buying {X}'],
         act=['The user fed it {X} yesterday', 'The user just added {X} to its bowl'],
         safe=['The user switched it to a lamb-formula food', 'The user reads the ingredient list before every purchase',
               'The user gave all the beef treats to a coworker\'s dog', 'The user only buys single-protein duck treats',
               'The user subscribed to a hypoallergenic food plan', 'The user asks the vet before every new treat',
               'The user taped an allergy list to the treat cabinet'],
         unrel=['The user is replacing the dog-bed cushion', 'The user booked next month\'s vet checkup',
                'The user bought a new leash', 'The user is comparing pet air purifiers']),
    dict(key='pet_chicken', lang=ZH,
         known=['用户家的猫对鸡肉过敏', '用户的猫一碰鸡肉就挠痒'],
         items=['鸡肉冻干', '鸡肉罐头', '鸡胸肉条'],
         intent=['用户打算给它换{X}', '用户想给它买点{X}', '用户准备囤一些{X}'],
         act=['用户昨天喂了它{X}', '用户刚才拆了一包{X}喂它'],
         safe=['用户给它换成了深海鱼配方', '用户把鸡肉零食都锁进了柜子', '用户给它买的是兔肉冻干',
               '用户下单前都让客服确认配方', '用户给它办了过敏原复检', '用户把含鸡肉的粮都退了',
               '用户只买鱼肉罐头'],
         unrel=['用户给猫换了新的猫抓板', '用户囤了两袋猫砂', '用户给猫窝加了个软垫', '用户在看自动喂食器的测评']),
    dict(key='pet_chicken', lang=EN,
         known=["The user's cat is allergic to chicken", "The user's cat scratches nonstop after chicken"],
         items=['freeze-dried chicken treats', 'a can of chicken cat food', 'chicken breast strips'],
         intent=['The user is planning to switch it to {X}', 'The user wants to get it some {X}',
                 'The user is going to stock up on {X}'],
         act=['The user fed it {X} yesterday', 'The user just opened a pack of {X} for it'],
         safe=['The user switched it to a deep-sea-fish formula', 'The user locked all chicken treats away',
               'The user buys rabbit freeze-dried instead', 'The user has support confirm the formula before ordering',
               'The user booked an allergen re-test for the cat', 'The user returned all the chicken-based food',
               'The user only buys fish cans'],
         unrel=['The user got the cat a new scratching post', 'The user stocked up on cat litter',
                'The user added a soft pad to the cat bed', 'The user is reading automatic-feeder reviews']),
    dict(key='pet_dairy', lang=ZH,
         known=['用户家的狗对乳制品过敏', '用户的狗一喝奶就拉肚子'],
         items=['奶片零食', '奶酪棒', '酸奶冻干'],
         intent=['用户想试试给它买{X}', '用户打算给它添点{X}', '用户准备下次给它带{X}'],
         act=['用户昨天给它吃了{X}', '用户刚才给它泡了{X}'],
         safe=['用户只给它喝宠物羊奶', '用户把奶制品都收出了它的活动区', '用户给它选的是无乳糖配方',
               '用户给它换成了肉冻干零食', '用户看了三款零食的配料表才下单', '用户把酸奶都留给自己吃了',
               '用户给它备的是洁齿骨'],
         unrel=['用户在给狗挑雨衣', '用户给狗洗了个澡', '用户带狗打了疫苗', '用户在整理宠物的用药盒']),
    dict(key='pet_dairy', lang=EN,
         known=["The user's dog is allergic to dairy", "The user's dog gets diarrhea from milk"],
         items=['milk-tablet treats', 'cheese sticks', 'freeze-dried yogurt bites'],
         intent=['The user wants to try buying it {X}', 'The user is planning to add {X} to its treats',
                 'The user is going to bring it {X} next time'],
         act=['The user gave it {X} yesterday', 'The user just mixed {X} into its food'],
         safe=['The user only gives it goat milk for pets', 'The user moved all dairy out of its reach',
               'The user picked a lactose-free formula', 'The user swapped to meat freeze-dried treats',
               'The user compared three ingredient lists before ordering', 'The user kept the yogurt for themselves',
               'The user stocked dental chews instead'],
         unrel=['The user is picking a raincoat for the dog', 'The user gave the dog a bath',
                'The user took the dog for its vaccine', 'The user is organizing the pet medicine box']),
    dict(key='pet_avocado', lang=ZH,
         known=['用户的鹦鹉对牛油果过敏', '用户家的鹦鹉不能碰牛油果'],
         items=['牛油果拌粮', '牛油果味的小食', '加了牛油果的滋养丸'],
         intent=['用户想给它买{X}', '用户打算给它掺点{X}', '用户准备试试{X}'],
         act=['用户昨天给它喂了{X}', '用户刚才往食盆加了{X}'],
         safe=['用户把牛油果都锁进了柜子', '用户给鹦鹉买的是无果配方的滋养丸', '用户查过鸟类禁忌食物清单',
               '用户只给它喂小米和蔬果', '用户把水果篮挪离了鸟笼', '用户给鸟食单独备了密封罐',
               '用户叮嘱家人别在鸟笼边吃牛油果'],
         unrel=['用户给鸟笼换了新的栖木', '用户在给鹦鹉挑玩具', '用户记录了鹦鹉的换羽期', '用户给鸟笼套了防尘罩']),
    dict(key='pet_avocado', lang=EN,
         known=["The user's parrot is allergic to avocado", "The user's parrot must never touch avocado"],
         items=['avocado-blend feed', 'avocado-flavored snacks', 'pellets with avocado oil'],
         intent=['The user wants to buy it {X}', 'The user is planning to mix in some {X}',
                 'The user is going to try {X}'],
         act=['The user fed it {X} yesterday', 'The user just added {X} to the bowl'],
         safe=['The user locked all avocados in the cabinet', 'The user buys avocado-free pellets',
               'The user checked the list of foods toxic to birds', 'The user only feeds millet and veggies',
               'The user moved the fruit basket away from the cage', 'The user keeps bird food in a sealed jar',
               'The user told the family not to eat avocado near the cage'],
         unrel=['The user replaced the cage perches', 'The user is picking toys for the parrot',
                'The user logged the parrot\'s molting season', 'The user fitted a dust cover on the cage']),
    dict(key='pet_grape', lang=ZH,
         known=['用户家的狗不能吃葡萄', '用户的狗对葡萄过敏'],
         items=['葡萄干面包', '含葡萄干的零食', '葡萄味软糖'],
         intent=['用户想掰点{X}给它', '用户打算给它尝尝{X}', '用户准备把手边的{X}分它一点'],
         act=['用户昨天喂了它{X}', '用户刚才丢了一颗{X}给它'],
         safe=['用户把葡萄干都收上了高层架子', '用户查过狗不能吃的水果清单', '用户买零食都挑无果干的',
               '用户给狗备的是苹果片零食', '用户提醒过家里别给狗喂葡萄', '用户把面包都换成了无添加的',
               '用户给狗办了宠物保险'],
         unrel=['用户在给狗选梳子', '用户带狗去修剪了毛', '用户给狗拍了写真', '用户在比较两款狗粮桶']),
    dict(key='pet_grape', lang=EN,
         known=["The user's dog cannot eat grapes", "The user's dog is allergic to grapes"],
         items=['raisin bread', 'snacks with raisins', 'grape-flavored gummies'],
         intent=['The user wants to share some {X} with it', 'The user is going to let it taste {X}',
                 'The user plans to give it a bit of {X}'],
         act=['The user fed it {X} yesterday', 'The user just dropped a piece of {X} for it'],
         safe=['The user put all raisins on a high shelf', 'The user checked the list of fruits dogs cannot eat',
               'The user buys only fruit-free treats', 'The user stocks apple-slice treats for the dog',
               'The user warned the family not to feed grapes', 'The user switched to additive-free bread',
               'The user got the dog pet insurance'],
         unrel=['The user is picking a brush for the dog', 'The user took the dog for a trim',
                'The user had a photo shoot for the dog', 'The user is comparing dog-food bins']),
]

# --- §3.1 HC doctor family: per-constraint sub-families (matched acts) --------
DOCTOR_HC = [
    dict(key='doctor_rest', lang=ZH,
         known=['用户腰肌劳损,医生让卧床休息两周', '用户韧带拉伤,医嘱是静养一个月'],
         items=['爬山', '打羽毛球', '去健身房练腿'],
         intent=['用户准备周末去{X}', '用户打算这两天去{X}', '用户计划下周去{X}', '用户在想要不要去{X}'],
         act=['用户昨天去{X}了', '用户刚才去{X}了'],
         safe=['用户在家做康复拉伸', '用户把健身卡暂停了一个月', '用户改约了线上会议没去现场',
               '用户按医嘱复查了一次', '用户请了康复师上门', '用户这周只散步十分钟',
               '用户把爬山计划推迟到了复查之后'],
         unrel=['用户在查康复科的评价', '用户给护腰换了新的', '用户在比较两款按摩仪', '用户网购了护膝']),
    dict(key='doctor_rest', lang=EN,
         known=['The user strained their back and the doctor ordered two weeks of bed rest',
                'The user sprained a ligament and the doctor ordered a month of rest'],
         items=['go hiking', 'play badminton', 'hit the gym for leg day'],
         intent=['The user is planning to {X} this weekend', 'The user intends to {X} in the next couple of days',
                 'The user scheduled {X}ing next week'],
         act=['The user went to {X} yesterday', 'The user just went {X}ing'],
         safe=['The user does rehab stretches at home', 'The user paused the gym membership for a month',
               'The user switched the meeting to a call and stayed home', 'The user had the follow-up as told',
               'The user booked a home-visit physiotherapist', 'The user only walked ten minutes this week',
               'The user postponed the hiking plan until after the checkup'],
         unrel=['The user is reading rehab-clinic reviews', 'The user got a new back brace',
                'The user is comparing massage devices', 'The user ordered knee pads online']),
    dict(key='doctor_postop', lang=ZH,
         known=['用户术后恢复期,医生叮嘱忌剧烈运动', '用户刚做完小手术,医嘱两周内不能运动'],
         items=['剧烈运动', '球类活动', '长跑'],
         intent=['用户打算去参加{X}', '用户准备恢复{X}', '用户计划周末做点{X}'],
         act=['用户昨天就去做了{X}', '用户今天上午干了场{X}'],
         safe=['用户每天只散步十分钟', '用户按医嘱复查了一次', '用户把球局都推给了同事',
               '用户改看比赛直播了', '用户办了康复科的上门评估', '用户这两周都坐车通勤',
               '用户遵医嘱拆线后才慢慢走动'],
         unrel=['用户在给病房挑靠枕', '用户复查挂了下周的号', '用户在整理病假条', '用户网购了容易穿脱的开衫']),
    dict(key='doctor_postop', lang=EN,
         known=['The user is recovering from surgery and the doctor said no strenuous exercise',
                'The user just had a minor procedure and cannot exercise for two weeks'],
         items=['intense workouts', 'ball games', 'long-distance running'],
         intent=['The user is planning to join {X}', 'The user intends to resume {X}',
                 'The user scheduled some {X} for the weekend'],
         act=['The user already did {X} yesterday', 'The user spent the morning on {X}'],
         safe=['The user only walks ten minutes a day', 'The user had the follow-up as told',
               'The user gave all their game slots to coworkers', 'The user switched to watching game streams',
               'The user booked a home rehab assessment', 'The user commutes by car these two weeks',
               'The user started walking slowly only after the stitches came out'],
         unrel=['The user is picking a lumbar pillow', 'The user booked a follow-up for next week',
                'The user is organizing their sick-leave notes', 'The user ordered easy-to-wear cardigans']),
    dict(key='doctor_gastritis', lang=ZH,
         known=['用户胃炎,医生要求忌辛辣', '用户胃不舒服,医嘱忌刺激饮食'],
         items=['火锅', '麻辣烫', '重辣烧烤'],
         intent=['用户想约朋友去吃{X}', '用户打算晚上点{X}', '用户准备聚餐去吃{X}'],
         act=['用户昨晚去吃了{X}', '用户中午点了{X}'],
         safe=['用户这周都吃的清粥小菜', '用户把聚餐改到了粤菜馆', '用户点餐都备注微辣改不辣',
               '用户囤了养胃的面条', '用户按医嘱吃了两周胃药', '用户把辣酱都送人了',
               '用户聚餐时只喝温水'],
         unrel=['用户在挑保温饭盒', '用户预约了胃镜复查', '用户囤了两盒苏打饼干', '用户在研究养胃食谱']),
    dict(key='doctor_gastritis', lang=EN,
         known=['The user has gastritis and the doctor said no spicy food',
                'The user\'s stomach is upset and the doctor ordered a bland diet'],
         items=['hotpot', 'spicy malatang', 'extra-spicy BBQ'],
         intent=['The user wants to invite friends out for {X}', 'The user plans to order {X} tonight',
                 'The user is arranging a {X} dinner'],
         act=['The user went out for {X} last night', 'The user ordered {X} at lunch'],
         safe=['The user has been eating congee and light dishes this week', 'The user moved the dinner to a Cantonese place',
               'The user always notes "no spice" on orders', 'The user stocked stomach-friendly noodles',
               'The user took the stomach medicine for two weeks as told', 'The user gave all the chili sauce away',
               'The user only drinks warm water at dinners'],
         unrel=['The user is picking an insulated lunch box', 'The user booked a gastroscopy follow-up',
                'The user stocked two boxes of soda crackers', 'The user is researching bland recipes']),
    dict(key='doctor_bp', lang=ZH,
         known=['用户血压偏高,医嘱低盐饮食', '用户在控血压,医生叮嘱少盐忌酒'],
         items=['腌菜', '咸鱼', '白酒'],
         intent=['用户想开一罐{X}', '用户打算晚饭配{X}', '用户准备学着腌{X}'],
         act=['用户晚饭就着{X}吃了两碗饭', '用户昨晚喝了{X}'],
         safe=['用户做饭改用限盐勺', '用户把家里的腌菜都送了人', '用户聚餐时只以茶代酒',
               '用户每天早晚各测一次血压', '用户按医嘱开始吃降压药', '用户囤了低钠盐',
               '用户戒掉了宵夜的重口卤味'],
         unrel=['用户在挑血压仪', '用户给爸妈寄了两盒低盐酱料', '用户预约了心内科门诊', '用户在研究DASH饮食法']),
    dict(key='doctor_bp', lang=EN,
         known=['The user has high blood pressure and the doctor ordered a low-salt diet',
                'The user is managing blood pressure and was told less salt, no alcohol'],
         items=['pickled vegetables', 'salted fish', 'baijiu'],
         intent=['The user wants to open a jar of {X}', 'The user plans to pair dinner with {X}',
                 'The user is thinking of pickling {X} at home'],
         act=['The user had two bowls of rice with {X} at dinner', 'The user drank {X} last night'],
         safe=['The user switched to a measured salt spoon', 'The user gave all the pickles away',
               'The user substitutes tea for alcohol at dinners', 'The user measures blood pressure twice a day',
               'The user started blood-pressure medication as told', 'The user stocked low-sodium salt',
               'The user quit the salty late-night snacks'],
         unrel=['The user is picking a blood-pressure monitor', 'The user mailed their parents low-salt sauces',
                'The user booked a cardiology appointment', 'The user is reading about the DASH diet']),
]

NEW_HC_SUBFAMS = PET_HC + DOCTOR_HC


# --- §3.2 alt_praise shape: hearsay praise of an alternative => false ---------
# 6 non-holdout families (city / job / gym_brand / coffee / car / cat_food),
# hair/barber domain deliberately excluded (V7 §2.2 note is report-only).
ALT_PRAISE_T = {
    (ZH, 'city'): ['用户刷到不少{B}的定居攻略,收藏了几篇', '用户听同事说{B}这几年发展不错', '用户刷到博主夸{B}适合生活',
                   '用户看到群里有讨论{B}的落户政策', '用户听朋友提过一句{B}的通勤挺方便'],
    (EN, 'city'): ['The user saved several guides about settling in {B}', 'A coworker mentioned {B} has grown nicely',
                   'The user saw a blogger praise {B} for daily life', 'The user saw a group chat about {B} residency policy',
                   'A friend once said commuting in {B} is easy'],
    (ZH, 'job'): ['用户听说{B}方向的机会挺多', '用户刷到有人分享转{B}的经历', '用户听前同事说{B}岗的团队氛围不错',
                  '用户看到招聘贴里{B}的岗位在涨', '用户听朋友夸{B}方向的前景'],
    (EN, 'job'): ['The user heard there are lots of {B} openings', 'The user saw someone share their switch to {B}',
                  'A former coworker praised the {B} team culture', 'The user noticed {B} postings rising',
                  'A friend spoke highly of the {B} track'],
    (ZH, 'gym_brand'): ['用户的朋友推荐了{B},说团课不错', '用户刷到{B}的探店视频,环境看着挺好', '用户听同事说{B}最近有优惠',
                        '用户看到群里有人夸{B}的教练', '用户听邻居提过{B}离家近'],
    (EN, 'gym_brand'): ['A friend recommended {B} for the group classes', 'The user saw a tour video of {B}; looks nice',
                        'A coworker mentioned {B} is running promos', 'The user saw group-chat praise for the {B} coaches',
                        'A neighbor said {B} is close to home'],
    (ZH, 'coffee'): ['用户刷到{B}的探店评价挺高', '用户听同事说楼下了家店{B}很正', '用户看到博主排雷后推荐了{B}',
                     '用户听朋友说{B}的豆子不错', '用户刷到{B}的拉花比赛视频'],
    (EN, 'coffee'): ['The user saw good reviews for a {B} place', 'A coworker said the shop downstairs pours a proper {B}',
                     'The user saw a blogger recommend the {B}', 'A friend said the {B} beans are solid',
                     'The user watched a latte-art video featuring the {B}'],
    (ZH, 'car'): ['用户听说{B}最近优惠力度不小', '用户刷到{B}的试驾视频,评价还行', '用户听同事说{B}的能耗不错',
                  '用户看到车友群在讨论{B}的新款', '用户听朋友提过{B}的售后挺好'],
    (EN, 'car'): ['The user heard the {B} has big discounts lately', 'The user saw a test-drive video of the {B}',
                  'A coworker said the {B} is efficient', 'The user saw owner-group chatter about the new {B}',
                  'A friend mentioned the {B} after-sales is good'],
    (ZH, 'cat_food'): ['用户看到养猫群里都在推荐{B}', '用户刷到博主测评后种草了{B}', '用户听猫友说{B}的适口性好',
                       '用户看到{B}上了个靠谱榜单', '用户听同事说她家猫吃{B}挺好'],
    (EN, 'cat_food'): ['The user saw the cat-owner group all recommend {B}', 'The user got sold on {B} after a review video',
                       'A cat-friend said {B} is palatable', 'The user saw {B} on a reputable list',
                       'A coworker said her cat does well on {B}'],
}
ALT_PRAISE_FAMS = ['city', 'job', 'gym_brand', 'coffee', 'car', 'cat_food']


def make_alt_praise_rows(rng, cats, quota, lang_mix, seen, label_rng, target_rng, ls_pool):
    """One praise-of-alternative row per (family, lang) with exact per-family quotas.
    gym_brand / cat_food exist ZH-only in the v3 category set => those families
    render ZH only (lang skew reported; per-family totals unchanged)."""
    rows = []
    per_fam = quota // len(ALT_PRAISE_FAMS)
    rem = quota - per_fam * len(ALT_PRAISE_FAMS)
    for i, fam in enumerate(ALT_PRAISE_FAMS):
        n_fam = per_fam + (1 if i < rem else 0)
        langs = [(ZH, EN)] if any(cc['key'] == fam and cc['lang'] == EN for cc in cats) else [(ZH, None)]
        n_en = round(n_fam * lang_mix[EN]) if len(langs) == 2 else 0
        plan = {ZH: n_fam - n_en, EN: n_en} if len(langs) == 2 else {ZH: n_fam}
        for lang, n_lang in plan.items():
            if n_lang == 0:
                continue
            c = next(cc for cc in cats if cc['key'] == fam and cc['lang'] == lang)
            made, guard = 0, 0
            while made < n_lang and guard < n_lang * 2000:
                guard += 1
                a, b = rng.sample(c['vals'], 2)
                known = rng.choice(c['known']).format(A=a)
                new = rng.choice(ALT_PRAISE_T[(lang, fam)]).format(B=b)
                state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                if state in seen:
                    continue
                seen.add(state)
                ls = next_ls(ls_pool)
                rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                             'gold': {'conflict': gold_for('consid', target_rng)},
                             '_meta': {'cat': fam, 'kind': 'consid', 'lang': lang,
                                       'labelset': ls, 'shape': 'alt_praise', 'family': fam}})
                made += 1
            assert made == n_lang, (fam, lang, made, n_lang)
    return rows


# --- §3.3 metric family: quantified checkup metrics rewritten => conflict -----
METRIC_CATS = [
    cat_v5('metric_weight_zh', ['65kg', '68kg', '70kg', '72kg'],
           known=['用户上次体检体重是{A}', '用户档案里的体重记录是{A}'],
           conflict=['这次体检涨到了{B}', '用户现在体重变成了{B}', '最新复查结果是{B},比之前高了'],
           compat=['用户按{A}时的食谱在吃', '体检其他项目正常,体重还是{A}', '用户这个月一直按{A}的标准控制'],
           consid=['用户有点担心涨到{B}', '用户担心下次体检会到{B}'],
           unrelated=['体脂秤换了电池', '用户办了张体检套餐', '体检中心搬了新址']),
    cat_v5('metric_weight_en', ['65kg', '68kg', '70kg', '72kg'],
           known=['The user\'s last checkup weight was {A}', 'The user\'s file lists a weight of {A}'],
           conflict=['This checkup it rose to {B}', 'The user now weighs {B}', 'The latest result is {B}, up from before'],
           compat=['The user still eats per the {A} meal plan', 'Everything else is normal and the weight is still {A}',
                   'The user has been holding the {A} target all month'],
           consid=['The user is a bit worried about rising to {B}', 'The user fears hitting {B} at the next checkup'],
           unrelated=['The smart scale got a new battery', 'The user bought a checkup package', 'The clinic moved to a new site']),
    cat_v5('metric_glucose_zh', ['5.4mmol/L', '5.8mmol/L', '6.3mmol/L', '6.6mmol/L'],
           known=['用户上次空腹血糖是{A}', '用户记录里的空腹血糖值是{A}'],
           conflict=['这次复查血糖升到了{B}', '用户最新的空腹血糖是{B}', '体检报告显示血糖{B},比上次高'],
           compat=['用户这月都按{A}时的饮食走', '复查时其他指标正常,血糖还是{A}', '用户的血糖一直稳定在{A}'],
           consid=['用户有点担心血糖涨到{B}', '用户怕下次复查血糖到{B}'],
           unrelated=['血糖仪换了试纸', '用户约了下月的体检', '用户给爸妈也买了血糖仪']),
    cat_v5('metric_glucose_en', ['5.4mmol/L', '5.8mmol/L', '6.3mmol/L', '6.6mmol/L'],
           known=['The user\'s last fasting glucose was {A}', 'The user\'s log lists a fasting glucose of {A}'],
           conflict=['This recheck the glucose rose to {B}', 'The user\'s latest fasting glucose is {B}',
                     'The report shows glucose at {B}, up from last time'],
           compat=['The user kept the {A} diet all month', 'Other markers were fine and glucose is still {A}',
                   'The user\'s glucose has stayed steady at {A}'],
           consid=['The user is a bit worried glucose will climb to {B}', 'The user fears a {B} reading next recheck'],
           unrelated=['The glucose meter got new strips', 'The user booked next month\'s checkup',
                      'The user bought a meter for their parents too']),
    cat_v5('metric_lipid_zh', ['1.7mmol/L', '2.3mmol/L', '2.8mmol/L', '3.1mmol/L'],
           known=['用户上次血脂(甘油三酯)是{A}', '用户档案里的血脂记录是{A}'],
           conflict=['这次体检血脂升到了{B}', '用户最新的血脂结果是{B}', '复查显示血脂{B},超标了'],
           compat=['用户一直按{A}时的食谱吃', '这次其他项目正常,血脂还是{A}', '用户的血脂半年都稳定在{A}'],
           consid=['用户有点担心血脂涨到{B}', '用户担心复查时血脂到{B}'],
           unrelated=['用户办了张健身房次卡', '体检套餐打了折', '用户囤了燕麦当早餐']),
    cat_v5('metric_lipid_en', ['1.7mmol/L', '2.3mmol/L', '2.8mmol/L', '3.1mmol/L'],
           known=['The user\'s last triglyceride reading was {A}', 'The user\'s file lists triglycerides at {A}'],
           conflict=['This checkup the triglycerides rose to {B}', 'The user\'s latest lipid result is {B}',
                     'The recheck shows lipids at {B}, over the limit'],
           compat=['The user still eats per the {A} plan', 'The other panels are normal and lipids remain {A}',
                   'The user\'s lipids have been steady at {A} for half a year'],
           consid=['The user is a bit worried lipids will rise to {B}', 'The user fears a {B} lipid reading at the recheck'],
           unrelated=['The user got a gym punch card', 'The checkup package was on sale', 'The user stocked oats for breakfast']),
    cat_v5('metric_hr_zh', ['72bpm', '85bpm', '95bpm', '102bpm'],
           known=['用户上次静息心率是{A}', '用户手环记录的静息心率是{A}'],
           conflict=['这次测出静息心率{B}', '用户现在的静息心率是{B}', '复查时心率到了{B},比记录高'],
           compat=['用户近期静息心率还是{A}', '手环这周平均还是{A}', '用户的心率一直在{A}附近'],
           consid=['用户有点担心心率升到{B}', '用户担心下次测出{B}'],
           unrelated=['手环换了表带', '用户续费了运动App会员', '用户给手环充了电']),
    cat_v5('metric_hr_en', ['72bpm', '85bpm', '95bpm', '102bpm'],
           known=['The user\'s last resting heart rate was {A}', 'The user\'s band logs a resting rate of {A}'],
           conflict=['This reading shows {B} at rest', 'The user\'s resting rate is now {B}',
                     'At the recheck the rate hit {B}, above the log'],
           compat=['The user\'s resting rate is still {A}', 'The band averaged {A} this week',
                   'The user\'s heart rate stays around {A}'],
           consid=['The user is a bit worried the rate will climb to {B}', 'The user fears reading {B} next time'],
           unrelated=['The band got a new strap', 'The user renewed the fitness app', 'The user charged the band']),
]

for _c in METRIC_CATS:  # cat_v5 leaves lang to the caller (SHAPE_CATS convention)
    _c['lang'] = EN if _c['key'].endswith('_en') else ZH


def make_metric_rows(rng, cats, quota, lang_mix, seen, label_rng, target_rng, ls_pool):
    """quota = {kind: n} over the metric sub-cats; conflict side gets shape='metric_shift'."""
    rows = []
    for kind, n in quota.items():
        n_en = round(n * lang_mix[EN])
        for lang, n_lang in ((ZH, n - n_en), (EN, n_en)):
            made, guard = 0, 0
            while made < n_lang and guard < n_lang * 2000:
                guard += 1
                c = rng.choice([cc for cc in cats if cc['lang'] == lang])
                vals = c['vals']
                if kind in ('true', 'consid'):
                    # §3.3: conflict = 上调为新值 (and consid worries about the rise)
                    # -- templates carry "up from before" wording, so enforce b > a
                    i = rng.randrange(len(vals) - 1)
                    a, b = vals[i], rng.choice(vals[i + 1:])
                else:
                    a, b = rng.sample(vals, 2)
                known = rng.choice(c['known']).format(A=a)
                # the conflict side lives under key 'conflict' (v3 cat interface)
                side = c['conflict'] if kind == 'true' else c[kind]
                new = rng.choice(side).format(A=a, B=b)
                state = json.dumps({'known': known, 'new': new}, ensure_ascii=False)
                if state in seen:
                    continue
                seen.add(state)
                ls = next_ls(ls_pool)
                meta = {'cat': 'metric', 'kind': kind, 'lang': lang, 'labelset': ls, 'family': c['key']}
                if kind == 'true':
                    meta['shape'] = 'metric_shift'
                rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                             'gold': {'conflict': gold_for(kind, target_rng)}, '_meta': meta})
                made += 1
            assert made == n_lang, (kind, lang, made, n_lang)
    return rows


# --- §3.1 HC row builder: exact per-kind quota over a family subset; lang
#     probabilistic (v6 convention -- the base pool's ZH safe space is tight,
#     exact per-lang quotas would exhaust it) ------------------------------------
def make_hc_rows_pool(rng, quota, lang_mix, seen, label_rng, target_rng, hc_pool, ls_pool):
    rows = []
    for kind, n in quota.items():
        made, guard = 0, 0
        while made < n and guard < n * 2000:
            guard += 1
            lang = rng.choices(list(lang_mix), weights=list(lang_mix.values()))[0]
            pool = [h for h in hc_pool if h['lang'] == lang]
            if not pool:
                continue
            h = rng.choice(pool)
            state, _gold = render_hc(rng, h, kind)
            if state in seen:
                continue
            seen.add(state)
            ls = next_ls(ls_pool)
            rows.append({'state': state, 'questions': question_for(ls, lang, label_rng),
                         'gold': {'conflict': gold_for(kind, target_rng)},
                         '_meta': {'cat': 'hardconstraint', 'kind': kind, 'lang': lang,
                                   'labelset': ls, 'family': h['key']}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


def main():
    rng = random.Random(SEED_V7)
    label_rng = random.Random(SEED_V7 + 1)
    target_rng = random.Random(SEED_V7 + 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # acceptance sets must never leak into train (V7 §3.1/§9)
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        for l in open(os.path.join(OUT_DIR, p), encoding='utf-8'):
            if l.strip():
                seen.add(json.loads(l)['state'])

    # MNLI 6000: same extraction as v4..v6 (exact state match), name-rotated too
    ls_pool = make_ls_pool(label_rng, 15300)
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

    # §3.1/§3.2 of v6 carry-over: value swaps + HC literal rewrite, then blocks
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

    # hard-constraint: base 600 (original families) + pet 200 + doctor 200 (70/20/10)
    base_pool = [h for h in patch_hc() if h['key'] not in {f['key'] for f in NEW_HC_SUBFAMS}]
    hc_base_rows = make_hc_rows_pool(rng, {'hardconstraint': 300, 'hc_safe': 180, 'hc_unrelated': 120},
                                     {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, base_pool, ls_pool)
    pet_rows = make_hc_rows_pool(rng, {'hardconstraint': 140, 'hc_safe': 40, 'hc_unrelated': 20},
                                 {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, PET_HC, ls_pool)
    doctor_rows = make_hc_rows_pool(rng, {'hardconstraint': 140, 'hc_safe': 40, 'hc_unrelated': 20},
                                    {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, DOCTOR_HC, ls_pool)

    # negation (1600) + perturbation (900), carried from v6
    neg_rows = make_neg_rows(rng, {'neg_true': 400, 'neg_false': 300, 'neg_false_boundary': 100},
                             {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    neg5_rows = make_neg5_rows(rng, seen, label_rng, target_rng, ls_pool)
    neg_unrel_rows = make_neg_unrel_rows(rng, 400, {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    shape_cats = [dict(c, lang=EN if c['key'].endswith('_en') else ZH) for c in SHAPE_CATS]
    cf_rows = make_cf_rows(rng, train_pool, shape_cats, {'cf_true': 300, 'cf_false': 300},
                           {ZH: 0.72, EN: 0.28}, seen, label_rng, target_rng, ls_pool)
    cf_unrel_rows = make_cf_unrel_rows(rng, shape_cats, 300, {ZH: 0.72, EN: 0.28}, seen,
                                       label_rng, target_rng, ls_pool)

    v7_train = (mnli_rows + soft_rows + alt_praise_rows + metric_rows
                + hc_base_rows + pet_rows + doctor_rows
                + neg_rows + neg5_rows + neg_unrel_rows + cf_rows + cf_unrel_rows)
    rng.shuffle(v7_train)

    # ---- §3.5 hard invariants -------------------------------------------------
    assert len(v7_train) == 15300, len(v7_train)
    n_true = sum(1 for r in v7_train if r['gold']['conflict']['label'] == 'true')
    n_false = len(v7_train) - n_true
    states = [r['state'] for r in v7_train]
    assert len(set(states)) == 15300, 'duplicate states'
    for p in ('nli_conflict_val_soft.jsonl', 'nli_conflict_val.jsonl'):
        other = {json.loads(l)['state'] for l in open(os.path.join(OUT_DIR, p), encoding='utf-8') if l.strip()}
        assert set(states).isdisjoint(other), f'leak into {p}'

    kind_counts = Counter(r['_meta']['kind'] for r in v7_train)
    expect_kinds = {'contradiction': 2000, 'entailment/neutral': 4000, 'true': 1840, 'compat': 1480,
                    'consid': 1360, 'unrelated': 1120, 'hardconstraint': 580, 'hc_safe': 260,
                    'hc_unrelated': 160, 'neg_true': 568, 'neg_false': 404, 'neg_false_boundary': 228,
                    'neg_unrelated': 400, 'cf_true': 300, 'cf_false': 300, 'cf_unrelated': 300}
    assert dict(kind_counts) == expect_kinds, (kind_counts, expect_kinds)

    label_mix = Counter(r['_meta']['labelset'] for r in v7_train)
    assert label_mix[5] == 3060 and label_mix[6] == 3060, label_mix
    assert sum(label_mix[s] for s in (1, 2, 3, 4)) == 15300 - 6120, label_mix
    for ls in range(1, 7):
        kinds_with = {r['_meta']['kind'] for r in v7_train if r['_meta']['labelset'] == ls}
        assert len(kinds_with) >= 3, (ls, len(kinds_with))

    for r in v7_train:
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

    # ---- §3.6 leak assert (hard invariant; case texts never enter training) ----
    assert_no_case_leak(v7_train)

    path = os.path.join(OUT_DIR, 'nli_conflict_train_v7.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in v7_train:
            meta = r.pop('_meta')
            meta['name_a'] = r['questions']['conflict']['labels']['false']
            meta['name_b'] = r['questions']['conflict']['labels']['true']
            r['meta'] = meta
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(v7_train)} rows')
    print(f'gold: true {n_true} / false {n_false} (= 1:{n_false / n_true:.2f})')
    print('meta.kind:', dict(sorted(Counter(r["meta"]["kind"] for r in v7_train).items())))
    print('meta.lang:', dict(Counter(r["meta"]["lang"] for r in v7_train)))
    print('meta.labelset:', dict(sorted(Counter(r["meta"]["labelset"] for r in v7_train).items())))
    sem = sum(Counter(r["meta"]["labelset"] for r in v7_train)[s] for s in (1, 2, 3, 4))
    print(f'labelset ratio: semantic {sem / 15300:.1%}, LS5 {label_mix[5] / 15300:.1%}, LS6 {label_mix[6] / 15300:.1%}')
    fam_counts = Counter((r['meta'].get('family'), r['meta']['kind']) for r in v7_train
                         if r['meta'].get('family') in {f['key'] for f in NEW_HC_SUBFAMS})
    pet_n = sum(n for (fam, _), n in fam_counts.items() if fam.startswith('pet_'))
    doc_n = sum(n for (fam, _), n in fam_counts.items() if fam.startswith('doctor_'))
    print(f'HC new families: pet {pet_n} / doctor {doc_n}; per family x kind:')
    for (fam, kind), n in sorted(fam_counts.items()):
        print(f'   {fam:16s} {kind:16s} {n}')
    ap = Counter(r['meta'].get('family') for r in v7_train if r['meta'].get('shape') == 'alt_praise')
    print('alt_praise per family:', dict(sorted(ap.items())), '| total', sum(ap.values()))
    mt = Counter((r['meta'].get('kind'), r['meta'].get('shape')) for r in v7_train if r['meta'].get('cat') == 'metric')
    print('metric kind x shape:', {f'{k}/{s}': n for (k, s), n in sorted(mt.items(), key=lambda x: str(x[0]))})


if __name__ == '__main__':
    main()
