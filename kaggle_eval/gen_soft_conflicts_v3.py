# -*- coding: utf-8 -*-
"""v3 soft-conflict augmentation (HANDOFF_NLI_V2.md -> v3 data arm).

v2's residual misses are all one data-distribution gap: MNLI contradictions are
lexical negations, so temporal/state-update conflicts ("moved to Beijing" vs
"lives in Shanghai") and change-intent non-conflicts ("considering a music
festival" vs "likes quiet") are never seen. This generator produces those pairs
with matched controls, on top of the untouched v2 MNLI data:

  v3 train = v2 train (6000, byte-identical) + synthetic 3000
             (1000 soft-conflict true + 2000 controls)  -> global ratio 1:2
  val_soft = 300 held-out pairs; 4 attribute categories are NEVER trained on
             (generalization slice), the rest are in-distribution.

Pair shapes per attribute category:
  conflict      known asserts value A, new asserts value B (update verbs) -> true
  compatible    new adds consistent detail about A                          -> false
  consideration new mentions *considering* the change to B (no commitment)  -> false
  unrelated     new talks about B as a neutral side topic                   -> false

Deterministic: seed 20260929. Output structure is byte-compatible with v2 JSONL
(same question JSON, 0.95/0.05 gold). One global dedup set spans train and both
val slices, so no pair ever repeats across files.
"""
import json
import os
import random
from collections import Counter

SEED = 20260929
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_local')

QUESTION = {
    'conflict': {
        'type': 'noul',
        'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
        'labels': {'false': '兼容', 'true': '冲突'},
    }
}
GOLD_FALSE = {'label': 'false', 'probabilities': {'false': 0.95, 'true': 0.05}}
GOLD_TRUE = {'label': 'true', 'probabilities': {'false': 0.05, 'true': 0.95}}

ZH, EN = 'zh', 'en'


def cat(key, vals, known, conflict, compat, consid, unrelated, lang=ZH, holdout=False):
    return dict(key=key, lang=lang, vals=vals, known=known, conflict=conflict,
                compat=compat, consid=consid, unrelated=unrelated, holdout=holdout)


CATEGORIES = [
    cat('city', ['上海', '北京', '广州', '深圳', '杭州', '成都'],
        known=['用户住在{A}', '用户目前定居在{A}', '用户家就在{A}'],
        conflict=['用户最近搬到{B}工作了', '用户下个月就去{B}定居了', '用户的工作调到了{B},准备长住'],
        compat=['用户在{A}上班', '用户对{A}的地铁线路很熟', '用户的社保一直交在{A}'],
        consid=['用户在考虑搬到{B}发展', '用户有点想去{B}试试机会'],
        unrelated=['{B}这两年发展很快', '公司团建去了{B}', '{B}有很多历史古迹']),
    cat('lang', ['Python', 'Java', 'Rust', 'Go', 'C++'],
        known=['用户主力编程语言是{A}', '用户平时主要用{A}写代码', '用户写后端一直用{A}'],
        conflict=['用户现在主力语言换成了{B}', '用户的新项目全部改用{B}了', '用户把主力语言切换到了{B}'],
        compat=['用户昨天还用{A}修了个小bug', '用户觉得{A}用着挺顺手', '用户的书架上全是{A}的书'],
        consid=['用户在考虑把主力语言换成{B}', '用户有点想学{B}来替代{A}'],
        unrelated=['{B}的编译器最近更新了', '公司代码库正在迁移仓库', '用户给电脑加装了内存条']),
    cat('gym_day', ['周三', '周五', '周一', '周六'],
        known=['用户每周{A}去健身房', '用户固定每周{A}锻炼', '用户的锻炼日是每周{A}'],
        conflict=['用户的健身时间改到了每周{B}', '用户把锻炼日调整到每周{B}了', '用户健身改成每周{B}了'],
        compat=['用户每周{A}健身时会带运动饮料', '用户上周{A}刚练完肩', '用户{A}下班的教练课约满'],
        consid=['用户在想把健身挪到每周{B}', '用户在纠结要不要把锻炼日改到{B}'],
        unrelated=['健身房的空调坏了', '用户的运动鞋刚换了新鞋垫', '教练推荐了新的训练计划']),
    cat('coffee', ['美式', '拿铁', '摩卡', '燕麦拿铁'],
        known=['用户喜欢喝{A}咖啡', '用户平时点{A}', '用户的咖啡固定喝{A}'],
        conflict=['用户现在只喝{B}了', '用户改喝{B}了', '用户的咖啡口味换成了{B}'],
        compat=['用户今天早上喝的还是{A}', '用户常去楼下的{A}折扣时段', '用户的咖啡卡绑的是{A}'],
        consid=['用户想尝尝{B}', '用户有点好奇{B}的味道'],
        unrelated=['咖啡豆涨价了', '公司咖啡机换了新的', '用户买了个随行杯']),
    cat('cloud', ['阿里云', '腾讯云', '华为云', 'AWS'],
        known=['用户的项目部署在{A}', '用户的线上服务跑在{A}', '用户的测试环境搭在{A}'],
        conflict=['项目已经迁移到{B}了', '用户的部署环境切换到了{B}', '用户把服务迁到了{B}'],
        compat=['项目上周在{A}发布了新版本', '用户很熟悉{A}的控制台', '用户的备案信息挂在{A}'],
        consid=['用户在评估迁移到{B}的方案', '用户在对比{A}和{B}的价格'],
        unrelated=['{B}的区域节点又新增了', '用户给服务器打了个补丁', '监控告警阈值调高了']),
    cat('phone', ['iPhone', '华为', '小米', 'OPPO'],
        known=['用户用的是{A}手机', '用户的手机是{A}', '用户随身带的是{A}'],
        conflict=['用户手机换成{B}了', '用户最近入手了{B}手机,旧的闲置了', '用户把手机换成了{B}'],
        compat=['用户刚给{A}换了新手机壳', '用户的{A}内存还挺够用', '用户的云相册同步在{A}上'],
        consid=['用户在考虑换{B}', '用户在纠结要不要入手{B}'],
        unrelated=['手机贴膜便宜了不少', '用户给平板下了个新游戏', '充电器找不到了']),
    cat('job', ['产品经理', '交互设计师', '后端工程师', '数据分析师'],
        known=['用户是一名{A}', '用户的职位是{A}', '用户在公司做{A}'],
        conflict=['用户转岗做了{B}', '用户换工作当{B}了', '用户的新职位是{B}'],
        compat=['用户在做一款记账App', '用户上周的工作是写评审文档', '用户的OKR里全是{A}的活'],
        consid=['用户在考虑转岗去做{B}', '用户在看{B}方向的机会'],
        unrelated=['公司下午茶换了供应商', '用户工位加了块显示器', '部门在招实习生']),
    cat('home_city', ['杭州', '武汉', '长沙', '西安'],
        known=['用户老家在{A}', '用户是{A}人', '用户从小在{A}长大'],
        conflict=['用户老家其实是{B},一直说错了', '用户户口在{B},不是本地人', '用户祖籍是{B}'],
        compat=['用户很想念{A}的家常菜', '用户每年春节回{A}', '用户说话还带{A}口音'],
        consid=['用户想带朋友回{B}看看', '用户在纠结十一去不去{B}'],
        unrelated=['{B}的小吃很有名', '高铁票不好买了', '用户最近在看游记']),
    cat('hair', ['长发', '短发', '齐肩发'],
        known=['用户留的是{A}', '用户现在是{A}', '用户的发型一直是{A}'],
        conflict=['用户上周剪成了{B}', '用户把头发改成了{B}', '用户烫了{B}的新造型'],
        compat=['用户今天刚洗了头发', '用户觉得{A}比较好打理', '用户的发型师也夸{A}合适'],
        consid=['用户在想要不要剪成{B}', '用户有点想换个{B}造型'],
        unrelated=['理发店会员卡涨价了', '用户买了瓶新的洗发水', '吹风机坏了']),
    cat('team', ['曼联', '皇马', '拜仁', '利物浦'],
        known=['用户的主队是{A}', '用户是{A}的球迷', '用户一直粉{A}'],
        conflict=['用户现在主队换成{B}了', '用户改支持{B}了', '用户现在是{B}的死忠'],
        compat=['用户昨晚看了{A}的比赛', '用户收藏了{A}的围巾', '用户的手机壳还是{A}主题'],
        consid=['用户有点想转投{B}门下', '用户在犹豫要不要粉{B}'],
        unrelated=['这轮联赛爆了好几个冷门', '球票又涨价了', '用户买了双新球鞋']),
    cat('gym_brand', ['乐刻', '超级猩猩', '威尔仕', 'Keep线下店'],
        known=['用户常去的健身房是{A}', '用户在{A}办了卡', '用户固定在{A}练'],
        conflict=['用户常去的健身房换成了{B}', '用户退了卡,改去{B}了', '用户现在固定去{B}'],
        compat=['用户在{A}约了私教课', '用户觉得{A}的团课不错', '用户的储物柜还租在{A}'],
        consid=['用户在对比{A}和{B}哪个划算', '用户考虑换到{B}去练'],
        unrelated=['蛋白粉促销了', '用户买了副新耳机健身用', '体测数据出来了']),
    cat('work_start', ['九点', '十点', '八点半'],
        known=['用户的公司{A}上班', '用户每天{A}到岗', '用户的到岗时间是{A}'],
        conflict=['公司上班时间改成了{B}', '用户现在{B}就要到岗了', '用户公司调成{B}上班制了'],
        compat=['用户今天{A}准时到了公司', '用户{A}前会先买杯咖啡', '用户的通勤正好够{A}到岗'],
        consid=['公司在讨论改成{B}上班', '用户希望以后能{B}再到岗'],
        unrelated=['早高峰地铁限流了', '用户换了条通勤路线', '会议室预约系统升级了']),
    cat('rent', ['3500', '4500', '2800', '5200'],
        known=['用户房租每月{A}', '用户的月租是{A}', '用户的房租稳定在{A}'],
        conflict=['房租涨到了每月{B}', '用户续租时月租调成了{B}', '用户搬到月租{B}的房子了'],
        compat=['这个月房租照常转了{A}', '用户觉得{A}的性价比还行', '用户的租房合同写的还是{A}'],
        consid=['房东暗示可能涨到{B}', '用户在看月租{B}左右的新房源'],
        unrelated=['小区绿化翻新了', '水电费单出来了', '用户买了盆绿萝']),
    cat('car', ['特斯拉', '比亚迪', '小鹏', '大众'],
        known=['用户的车是{A}', '用户开的是{A}', '用户家里那台车是{A}'],
        conflict=['用户的车换成了{B}', '用户把旧车卖了,换了{B}', '用户提了台{B}'],
        compat=['用户给{A}做了保养', '用户的车载支架装在{A}上', '用户的充电桩是配套{A}装的'],
        consid=['用户在看{B},考虑换车', '用户在试驾{B}'],
        unrelated=['停车场收费上调了', '用户换了行车记录仪', '路上电动车越来越多了']),
    cat('cat_food', ['皇家', '渴望', '网易严选', '巅峰'],
        known=['用户家猫粮吃的是{A}', '用户给猫喂{A}', '用户囤的猫粮是{A}'],
        conflict=['猫粮换成{B}了', '用户给猫改喂{B}了', '用户家现在吃的是{B}猫粮'],
        compat=['用户囤了两袋{A}', '猫挺爱吃{A}', '用户按{A}的喂食表定时喂'],
        consid=['用户在考虑换{B}试试', '用户纠结要不要换成{B}'],
        unrelated=['猫抓板又被抓烂了', '用户给猫买了逗猫棒', '猫明天要打疫苗']),
    cat('alarm', ['七点', '六点半', '八点'],
        known=['用户每天{A}起床', '用户的闹钟定在{A}', '用户的起床时间一直是{A}'],
        conflict=['用户的闹钟改到了{B}', '用户现在每天{B}就起了', '用户的起床时间调成了{B}'],
        compat=['用户今天{A}准时起床了', '用户起床后先喝杯温水', '用户{A}起床后先拉伸十分钟'],
        consid=['用户想在冬天改成{B}起床', '用户打算试试{B}起床'],
        unrelated=['闹钟App换了新的', '用户睡前在听播客', '窗帘遮光效果不错']),
    # ---- holdout: never appear in v3 training, only in val_soft (generalization slice)
    cat('email', ['gmail', 'QQ邮箱', 'Outlook', '163邮箱'],
        known=['用户常用邮箱是{A}', '用户主要用{A}收邮件', '用户的注册邮箱一直是{A}'],
        conflict=['用户的常用邮箱换成{B}了', '用户现在邮件都转到{B}了', '用户把主邮箱换成了{B}'],
        compat=['用户的{A}收到了会议邀请', '用户用{A}注册了新服务', '用户在{A}上订了机票'],
        consid=['用户在考虑把主邮箱换成{B}', '用户想迁移到{B}'],
        unrelated=['垃圾邮件变多了', '用户给邮箱设了新签名', '邮件客户端更新了'], holdout=True),
    cat('desk_floor', ['5楼', '12楼', '3楼', '8楼'],
        known=['用户的工位在{A}', '用户坐在{A}的办公区', '用户的座位一直在{A}'],
        conflict=['用户的工位搬到了{B}', '用户换到{B}的工位了', '用户部门整体挪到了{B}'],
        compat=['用户在{A}的打印机打印材料', '用户{A}的靠窗工位视野不错', '用户的储物柜就在{A}'],
        consid=['用户听说可能要搬到{B}', '用户在打听{B}还有没有空位'],
        unrelated=['{B}的会议室刚装修好', '电梯要检修了', '茶水间换了咖啡豆'], holdout=True),
    cat('degree', ['400度', '500度', '600度', '300度'],
        known=['用户近视{A}', '用户眼镜是{A}的', '用户验光度数是{A}'],
        conflict=['用户近视涨到了{B}', '用户新配的眼镜是{B},度数加深了', '用户复查结果是{B}'],
        compat=['用户戴着眼镜看书', '用户的眼镜片有防蓝光镀膜', '用户的眼镜框是按{A}配的'],
        consid=['用户担心度数涨到{B}', '用户在想要不要重新验光'],
        unrelated=['眼镜盒摔裂了', '用户买了副太阳镜夹片', '验光师说姿势要调整'], holdout=True),
    cat('barber', ['Tony', 'Kevin', 'Andy', '阿杰'],
        known=['用户常找发型师{A}', '用户的固定理发师是{A}', '用户剪发一直找{A}'],
        conflict=['用户的固定理发师换成了{B}', '用户现在都找{B}剪', '用户改让{B}做头发了'],
        compat=['用户约了{A}下周剪发', '用户觉得{A}剪得细致', '用户的会员卡绑定在{A}名下'],
        consid=['用户在想要不要换{B}试试', '用户听说{B}手艺不错'],
        unrelated=['理发店换了新价目表', '用户办了张洗剪吹次卡', '店里换了新洗发水'], holdout=True),
]

CATEGORIES += [
    cat('city', ['Shanghai', 'Beijing', 'Shenzhen', 'Hangzhou', 'Chengdu'],
        known=['The user lives in {A}', 'The user is based in {A}', "The user's home is in {A}"],
        conflict=['The user recently moved to {B} for work', 'The user is relocating to {B} next month',
                  'The user got transferred to {B} and is settling there'],
        compat=['The user works in an office near {A}', 'The user knows the subway lines in {A} well',
                "The user's social insurance is registered in {A}"],
        consid=['The user is considering moving to {B}', 'The user is thinking about trying {B} for a change'],
        unrelated=['{B} has grown fast in recent years', 'The company offsite was held in {B}',
                   '{B} has many historic sites'], lang=EN),
    cat('lang', ['Python', 'Java', 'Rust', 'Go'],
        known=['The user mainly codes in {A}', "The user's primary language is {A}",
               'The user has been writing backends in {A}'],
        conflict=['The user switched their main language to {B}', 'The user now writes everything in {B}',
                  'The user moved from {A} to {B} as their main language'],
        compat=['The user fixed a small bug in {A} yesterday', 'The user finds {A} quite handy',
                "The user's bookshelf is full of {A} books"],
        consid=['The user is considering switching to {B}', 'The user is tempted to learn {B} to replace {A}'],
        unrelated=['The {B} compiler shipped a new release', 'The company repo is being migrated',
                   'The user added more RAM to their laptop'], lang=EN),
    cat('gym_day', ['Mondays', 'Wednesdays', 'Fridays', 'Saturdays'],
        known=['The user works out on {A}', 'The user goes to the gym on {A}',
               'The user trains every {A}'],
        conflict=['The user moved their gym days to {B}', 'The user now trains on {B}',
                  'The user changed their workout day to {B}'],
        compat=['The user brings a sports drink on {A}', 'The user trained shoulders last {A}',
                'The {A} class with their coach is fully booked'],
        consid=['The user is thinking of moving workouts to {B}', 'The user is considering {B} sessions instead'],
        unrelated=['The gym AC is broken', 'The user got new insoles', 'The coach suggested a new program'], lang=EN),
    cat('coffee', ['americano', 'latte', 'mocha', 'oat latte'],
        known=['The user usually drinks {A}', "The user's coffee order is {A}",
               'The user always orders {A}'],
        conflict=['The user only drinks {B} now', 'The user switched to {B}', "The user's order changed to {B}"],
        compat=['The user had {A} this morning', 'The user likes the {A} discount hours downstairs',
                "The user's coffee card is tied to {A}"],
        consid=['The user wants to try {B}', 'The user is curious about {B}'],
        unrelated=['Coffee bean prices went up', 'The office got a new coffee machine',
                   'The user bought a travel mug'], lang=EN),
    cat('cloud', ['Alibaba Cloud', 'Tencent Cloud', 'AWS', 'Azure'],
        known=["The user's project runs on {A}", 'The user deploys on {A}',
               "The user's staging environment sits on {A}"],
        conflict=['The project has been migrated to {B}', 'The user switched deployments to {B}',
                  'The user moved their services to {B}'],
        compat=['The project shipped a new release on {A} last week', 'The user knows the {A} console well',
                "The user's ICP filing is registered on {A}"],
        consid=['The user is evaluating a move to {B}', 'The user is comparing prices between {A} and {B}'],
        unrelated=['{B} opened a new region', 'The user patched the server', 'The alert threshold was raised'], lang=EN),
    cat('job', ['product manager', 'UX designer', 'backend engineer', 'data analyst'],
        known=['The user is a {A}', 'The user works as a {A}', 'The user does {A} work at their company'],
        conflict=['The user transferred to become a {B}', 'The user changed jobs and is now a {B}',
                  "The user's new role is {B}"],
        compat=['The user is building a budgeting app', 'The user spent last week on review documents',
                "The user's OKR is full of {A} work"],
        consid=['The user is considering moving into {B}', 'The user is looking at {B} openings'],
        unrelated=['The office changed the snack vendor', 'The user got an extra monitor',
                   'The team is hiring interns'], lang=EN),
    cat('hair', ['long hair', 'short hair', 'shoulder-length hair'],
        known=['The user has {A}', 'The user currently wears {A}', 'The user has always kept {A}'],
        conflict=['The user cut it to {B} last week', 'The user changed their style to {B}',
                  'The user now has {B}'],
        compat=['The user just washed their hair today', 'The user finds {A} easy to manage',
                'The stylist agreed {A} suits them'],
        consid=['The user is wondering whether to cut to {B}', 'The user is tempted by a {B} look'],
        unrelated=['The salon raised membership prices', 'The user bought a new shampoo',
                   'The hairdryer broke'], lang=EN),
    cat('car', ['a Tesla', 'a BYD', 'a VW', 'an Xpeng'],
        known=['The user drives {A}', "The user's car is {A}", "The family car is {A}"],
        conflict=['The user switched to {B}', 'The user sold the old car and bought {B}', 'The user just got {B}'],
        compat=['The user took {A} for maintenance', 'The phone mount is installed in {A}',
                "The charging station was installed for {A}"],
        consid=['The user is looking at {B} for a possible upgrade', 'The user is test-driving {B}'],
        unrelated=['Parking fees went up', 'The user replaced the dashcam',
                   'Electric scooters are everywhere now'], lang=EN),
    cat('alarm', ['7 am', '6:30 am', '8 am'],
        known=['The user gets up at {A}', 'The alarm is set for {A}', "The user's wake-up time has always been {A}"],
        conflict=['The alarm was moved to {B}', 'The user now gets up at {B}', 'Wake-up time changed to {B}'],
        compat=['The user got up on time at {A} today', 'The user drinks warm water right after waking',
                'The user stretches for ten minutes after the {A} alarm'],
        consid=['The user wants to switch to {B} in winter', 'The user plans to try {B} wake-ups'],
        unrelated=['The user switched alarm apps', 'The user listens to podcasts before sleep',
                   'The blackout curtains work well'], lang=EN),
    cat('team', ['Manchester United', 'Real Madrid', 'Bayern', 'Liverpool'],
        known=["The user's team is {A}", 'The user roots for {A}', 'The user has always backed {A}'],
        conflict=['The user now supports {B}', 'The user switched allegiance to {B}',
                  "The user's main club is {B} these days"],
        compat=['The user watched the {A} match last night', 'The user owns an {A} scarf',
                'The user still uses an {A} phone case'],
        consid=['The user is tempted to start following {B}', 'The user is torn about backing {B}'],
        unrelated=['There were upsets all round this weekend', 'Ticket prices rose again',
                   'The user bought new sneakers'], lang=EN),
]


def render(rng, c, kind):
    """Render one pair for the given kind; returns (state_json, gold, cat_key, kind)."""
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


def make_rows(rng, cats, quota, lang_mix, seen):
    """Fill quota {kind: n}; every state string globally unique via `seen`."""
    rows = []
    for kind, n in quota.items():
        made, guard = 0, 0
        while made < n and guard < n * 500:
            guard += 1
            lang = rng.choices(list(lang_mix), weights=list(lang_mix.values()))[0]
            cand = [c for c in cats if c['lang'] == lang]
            if not cand:
                continue
            c = rng.choice(cand)
            state, gold, catkey, k = render(rng, c, kind)
            if state in seen:
                continue
            seen.add(state)
            rows.append({'state': state, 'questions': QUESTION,
                         'gold': {'conflict': dict(gold)},
                         '_meta': {'cat': catkey, 'kind': k, 'lang': lang}})
            made += 1
        assert made == n, (kind, made, n)
    return rows


def write_jsonl(path, rows):
    stats_lang = dict(Counter(r.get('_meta', {}).get('lang', 'mnli') for r in rows))
    stats_gold = dict(Counter(r['gold']['conflict']['label'] for r in rows))
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k != '_meta'}  # keep v2 field parity
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(rows)} rows | gold {stats_gold} | lang {stats_lang}')


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()

    # v3 synthetic TRAIN block: 1000 true + 2000 false (800 compat / 600 consid / 600 unrelated)
    train_rows = make_rows(rng, [c for c in CATEGORIES if not c['holdout']],
                           {'true': 1000, 'compat': 800, 'consid': 600, 'unrelated': 600},
                           {ZH: 0.72, EN: 0.28}, seen)
    # val_soft: holdout slice (categories never trained on) + in-distribution slice
    holdout_rows = make_rows(rng, [c for c in CATEGORIES if c['holdout']],
                             {'true': 30, 'compat': 25, 'consid': 25, 'unrelated': 20},
                             {ZH: 1.0}, seen)
    indist_rows = make_rows(rng, [c for c in CATEGORIES if not c['holdout']],
                            {'true': 70, 'compat': 60, 'consid': 40, 'unrelated': 30},
                            {ZH: 0.72, EN: 0.28}, seen)

    # assemble v3 train = v2 MNLI train (untouched) + synthetic block
    v2_train = [json.loads(l) for l in open(
        os.path.join(OUT_DIR, 'nli_conflict_train.jsonl'), encoding='utf-8') if l.strip()]
    assert len(v2_train) == 6000, len(v2_train)
    v3_train = v2_train + train_rows
    rng.shuffle(v3_train)
    val_soft = holdout_rows + indist_rows
    rng.shuffle(val_soft)

    write_jsonl(os.path.join(OUT_DIR, 'nli_conflict_train_v3.jsonl'), v3_train)
    write_jsonl(os.path.join(OUT_DIR, 'nli_conflict_val_soft.jsonl'), val_soft)
    # sidecar keeps the slice/category/language info that the clean JSONL strips
    with open(os.path.join(OUT_DIR, 'nli_conflict_val_soft.meta.json'), 'w', encoding='utf-8') as f:
        json.dump([r['_meta'] for r in val_soft], f, ensure_ascii=False, indent=1)

    true_total = sum(1 for r in v3_train if r['gold']['conflict']['label'] == 'true')
    print(f'v3 train total {len(v3_train)} | true:false = {true_total}:{len(v3_train)-true_total} '
          f'(= 1:{(len(v3_train)-true_total)/true_total:.2f})')
    print('val_soft: holdout', len(holdout_rows), '| in-dist', len(indist_rows))


if __name__ == '__main__':
    main()
