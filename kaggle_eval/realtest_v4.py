# -*- coding: utf-8 -*-
"""v4 real-case acceptance: frozen old-20 + negation-5 + NEW 10 (HANDOFF_NLI_V3.md, 改动5).

The old 20 are imported verbatim from realtest_v2.py (跨版本可比性). The 10 new
cases avoid every synthetic attribute category (会议时间/订阅服务/语言学习/通勤方式/
宠物粮食/快递地址/音乐口味/室友/笔记本型号/运动伤病史) and include 2 hard-constraint
cases (pet-food allergy, knee injury) + 1 negation case.

Usage: python realtest_v4.py <ckpt> <out_json>
"""
import json
import sys

import laya

sys.path.insert(0, r'D:\laya\kaggle_eval')
from realtest_v2 import CASES, NEGATION_CASES, QUESTION  # noqa: E402  frozen sets, verbatim

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\memory_conflict_realtest_v4.json'

# 10 new cases, template-independent (task 改动5); tags for the report
NEW_CASES = [
    (({'known': '用户的周会固定在周一上午十点', 'new': '周会已经改到周四下午三点了'}, 'true'), '新-会议时间'),
    (({'known': '用户没有订阅任何视频会员', 'new': '用户昨晚开通了视频会员月卡'}, 'true'), '新-订阅服务(否定句)'),
    (({'known': '用户在学日语,目标是年底过N2', 'new': '用户把学习计划从日语换成了西班牙语'}, 'true'), '新-语言学习'),
    (({'known': '用户通勤坐地铁', 'new': '用户在地铁上听播客'}, 'false'), '新-通勤方式'),
    (({'known': '用户家的猫对谷物过敏', 'new': '用户想换一款含谷物的猫粮试试'}, 'true'), '新-宠物粮食(硬约束)'),
    (({'known': '用户的默认收货地址是公司前台', 'new': '用户让前台的同事帮忙代收了一个快递'}, 'false'), '新-快递地址'),
    (({'known': '用户喜欢民谣', 'new': '用户昨晚去听了场民谣现场'}, 'false'), '新-音乐口味'),
    (({'known': '用户和室友合租两居室', 'new': '室友搬走了,用户现在一个人住'}, 'true'), '新-室友'),
    (({'known': '用户用的是14寸轻薄本', 'new': '用户给笔记本加装了一条内存'}, 'false'), '新-笔记本型号'),
    (({'known': '用户膝盖韧带拉伤,医生要求静养一个月', 'new': '用户周末打算去踢球'}, 'true'), '新-运动伤病史(硬约束)'),
]


def run(agent, cases):
    results = []
    for (state, expect), note in cases:
        p = float(agent.predict(state, {'conflict': QUESTION})['answers']['conflict']['noul'])
        got = 'true' if p >= 0.5 else 'false'
        results.append({'note': note, 'expect': expect, 'got': got,
                        'p_conflict': round(p, 4), 'ok': got == expect})
        print('OK  ' if got == expect else 'MISS', note, expect, '->', got, round(p, 4))
    return results


def main():
    agent = laya.load(CKPT, device='cpu')
    main_results = run(agent, CASES)
    neg_results = run(agent, NEGATION_CASES)
    new_results = run(agent, NEW_CASES)
    n_ok = sum(r['ok'] for r in main_results)
    neg_ok = sum(r['ok'] for r in neg_results)
    new_ok = sum(r['ok'] for r in new_results)
    print(f'TOTAL old-20: {n_ok}/20 | NEGATION: {neg_ok}/5 | NEW-10: {new_ok}/10')
    json.dump({'checkpoint': CKPT,
               'cases': main_results, 'total': f'{n_ok}/20',
               'negation_cases': neg_results, 'negation_total': f'{neg_ok}/5',
               'new_cases': new_results, 'new_total': f'{new_ok}/10'},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
