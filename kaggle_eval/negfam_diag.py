# -*- coding: utf-8 -*-
"""V12 §2.2 negation-family diagnostic (preregistered gate #13).

Mirrors family_diag.py (v11 gate #12) for the OPPOSITE polarity: known carries
an explicit negation (「没有/从不/未」系) and new VIOLATES it => conflict (gold
true).  Round 11 showed the two chronic acceptance cases of this shape
(「否定-养宠物反转」「新-订阅服务(否定句)」) flip in ~1/3 of same-config runs --
a shape-level weakness, not pure sampling noise (HANDOFF_NLI_V12.md §0.1).
This 8-case family rate tests whether the "negated known + violating new =>
true" rule GENERALIZES; 4 cases cover the chronic domains (pet / subscription),
4 cross fresh domains, zh/en 4+4.  The chronic acceptance texts themselves stay
out of the instrument (and out of training, §2.1 hard constraint).

Gate (HANDOFF_NLI_V12.md §1.2 #13, preregistered; enforced by
protocol_verdict.py --protocol v12):  median over runs of family-mean
p_conflict >= 0.5, AND every run has at most 1 of the 8 cases at
p_conflict < 0.5 (confident-miss).  Template disjointness from the §2.1 lever
rows is asserted in gen_soft_conflicts_v12.py so a family pass cannot be
template recall of the lever itself.

Usage: python negfam_diag.py <ckpt> <out_json>
"""
import json
import sys

import laya

from noul_bias_diag import QUESTION

NEGFAM_CASES = [
    # (state, expect, note) -- negation-known isomorphic: explicit negation known + violating new
    (({'known': '用户家里没有养仓鼠', 'new': '用户刚下单了一整套仓鼠用品'}, 'true'), '否定家族N1 宠物', True),
    (({'known': "The user doesn't have a dog", 'new': 'The user just bought a dog bed and puppy food'}, 'true'), '否定家族N2 pet (en)', True),
    (({'known': '用户没有开通任何云存储服务', 'new': '用户昨晚买了年度云盘会员'}, 'true'), '否定家族N3 订阅', True),
    (({'known': 'The user has never subscribed to any music service', 'new': 'The user activated a premium music plan yesterday'}, 'true'), '否定家族N4 subscription (en)', True),
    (({'known': '用户从来不喝咖啡', 'new': '用户今天早上买了一杯美式'}, 'true'), '否定家族N5 饮食', False),
    (({'known': 'The user never drives to work', 'new': 'The user drove to the office this morning'}, 'true'), '否定家族N6 commute (en)', False),
    (({'known': '用户周五晚上不排会议', 'new': '用户把一个客户会约在了周五晚上'}, 'true'), '否定家族N7 日程', False),
    (({'known': 'The user has no game console at home', 'new': 'The user unboxed a new game console yesterday'}, 'true'), '否定家族N8 device (en)', False),
]


def run(agent):
    results = []
    for (state, expect), note, is_chronic in NEGFAM_CASES:
        a = agent.predict(state, {'conflict': QUESTION})['answers']['conflict']
        p_conflict = float(a['noul'])
        got = 'true' if p_conflict >= 0.5 else 'false'
        results.append({'note': note, 'expect': expect, 'got': got, 'p_conflict': round(p_conflict, 4),
                        'ok': got == expect, 'is_chronic_domain': is_chronic})
    return results


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
    out = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\negfam_diag_v12_r1.json'
    agent = laya.load(ckpt, device='cpu')
    cases = run(agent)
    fam_mean = sum(c['p_conflict'] for c in cases) / len(cases)
    n_lowconf = sum(1 for c in cases if c['p_conflict'] < 0.5)
    n_ok = sum(c['ok'] for c in cases)
    for c in cases:
        print('OK  ' if c['ok'] else 'MISS', c['note'], c['expect'], '->', c['got'], c['p_conflict'])
    print(f'negfam: {n_ok}/8 ok | mean p_conflict {fam_mean:.4f} | p<0.5 cases {n_lowconf}')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': ckpt, 'cases': cases, 'family_mean_p': round(fam_mean, 4),
                   'n_lowconf': n_lowconf, 'n_ok': n_ok}, f, ensure_ascii=False, indent=2)
    print('saved:', out)


if __name__ == '__main__':
    main()
