# -*- coding: utf-8 -*-
"""Real-case acceptance test for the v2 conflict head (acceptance gate #3/#4).

The 20 cases are verbatim from HANDOFF_NLI_V2.md so the result is comparable with
v1's 9/20. Answer key confirmed against predict(): answers['conflict']['noul'] is
P(冲突) — the question uses custom Chinese labels but the answer key stays 'noul'.
Negation cases (upstream #377: negation is a known model-level weakness) are scored
separately and reported but are NOT a hard gate.
"""
import json
import sys

import laya

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\石晴\Desktop\laya-kaggle-output\laya-nli-conflict-v2'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'C:\Users\石晴\Desktop\laya\data_local\memory_conflict_realtest_v2.json'

QUESTION = {'type': 'noul',
            'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
            'labels': {'false': '兼容', 'true': '冲突'}}

CASES = [  # (state, 期望, 说明) — 与 v1 实测完全同一条
    (({'known': '用户住在上海', 'new': '用户最近搬到北京工作了'}, 'true'), '地点变更'),
    (({'known': '用户主要用 Python 写代码', 'new': '用户现在主力语言换成了 Rust'}, 'true'), '偏好变更'),
    (({'known': '用户每周三去健身房', 'new': '用户的健身时间改到了每周五'}, 'true'), '时间变更'),
    (({'known': '用户对花生过敏', 'new': '用户说他可以吃花生酱了'}, 'true'), '关键事实反转'),
    (({'known': '用户的项目部署在阿里云', 'new': '项目已经迁移到腾讯云了'}, 'true'), '部署变更'),
    (({'known': '用户喜欢喝美式咖啡', 'new': '用户现在只喝拿铁'}, 'true'), '口味变更'),
    (({'known': '用户的猫叫小白', 'new': '用户的猫叫小黑'}, 'true'), '名字矛盾'),
    (({'known': '用户今年 28 岁', 'new': '用户今年 35 岁'}, 'true'), '年龄矛盾'),
    (({'known': '用户喜欢喝咖啡', 'new': '用户今天早上喝了一杯咖啡'}, 'false'), '一致行为'),
    (({'known': '用户养了一只猫', 'new': '用户的猫昨天打了疫苗'}, 'false'), '补充细节'),
    (({'known': '用户在学人工智能', 'new': '用户今天学了微积分'}, 'false'), '无关补充'),
    (({'known': '用户住在北京', 'new': '北京有很多历史古迹'}, 'false'), '无关陈述'),
    (({'known': '用户是产品经理', 'new': '用户在做一款记账 App'}, 'false'), '工作相关'),
    (({'known': '用户每天早上七点起床', 'new': '用户今天七点起床后去跑步了'}, 'false'), '习惯一致'),
    (({'known': '用户喜欢科幻电影', 'new': '用户昨晚看了一部科幻片'}, 'false'), '偏好一致'),
    (({'known': '用户用 Windows 电脑', 'new': '用户给电脑装了新键盘'}, 'false'), '兼容细节'),
    (({'known': '用户喜欢安静的环境', 'new': '用户在考虑去热闹的音乐节'}, 'false'), '临时的不违背偏好'),
    (({'known': '用户不吃辣', 'new': '用户今天想尝试一下微辣'}, 'true'), '饮食禁忌松动'),
    (({'known': '用户有一个妹妹', 'new': '用户是家里的独生子'}, 'true'), '家庭结构矛盾'),
    (({'known': '用户住在上海浦东', 'new': '用户在上海上班'}, 'false'), '子集关系'),
]

NEGATION_CASES = [  # 否定句 case:单独统计,不作硬门槛(#377 否定句失效是模型层限制)
    (({'known': '用户没有养宠物', 'new': '用户的狗昨天打了疫苗'}, 'true'), '否定-养宠物反转'),
    (({'known': '用户从不喝咖啡', 'new': '用户每天早上都喝一杯拿铁'}, 'true'), '否定-从不喝咖啡'),
    (({'known': '用户已经戒烟了', 'new': '用户昨天抽了一包烟'}, 'true'), '否定-戒烟后复吸'),
    (({'known': '用户不喜欢科幻电影', 'new': '用户从来不看科幻片'}, 'false'), '双重否定-一致'),
    (({'known': '用户没有车', 'new': '用户每天坐地铁上班'}, 'false'), '否定-无车与通勤兼容'),
]


def run(agent, cases):
    results = []
    for (state, expect), note in cases:
        a = agent.predict(state, {'conflict': QUESTION})['answers']['conflict']
        p_conflict = float(a['noul'])
        got = 'true' if p_conflict >= 0.5 else 'false'
        ok = got == expect
        results.append({'note': note, 'expect': expect, 'got': got,
                        'p_conflict': round(p_conflict, 4), 'ok': ok})
        print('OK  ' if ok else 'MISS', note, expect, '->', got, round(p_conflict, 4))
    return results


def main():
    agent = laya.load(CKPT, device='cpu')
    main_results = run(agent, CASES)
    neg_results = run(agent, NEGATION_CASES)
    n_ok = sum(r['ok'] for r in main_results)
    neg_ok = sum(r['ok'] for r in neg_results)
    print(f'TOTAL: {n_ok}/{len(main_results)}')
    print(f'NEGATION: {neg_ok}/{len(neg_results)}')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': CKPT,
                   'cases': main_results,
                   'total': f'{n_ok}/{len(main_results)}',
                   'negation_cases': neg_results,
                   'negation_total': f'{neg_ok}/{len(neg_results)}'},
                  f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
