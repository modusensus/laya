# -*- coding: utf-8 -*-
"""V11 §2.2 isomorphic-family diagnostic (preregistered gate #12).

Round 10 showed the single B2 probe has no per-run decision power (50pp
run-variance band across same-config runs, V10 §10.4).  This instrument
replaces the single probe with a 9-case FAMILY RATE: B2 itself plus 8 fresh
same-shape cases (name/reference known + COMPATIBLE second attribute => false)
across four non-pet domains, zh/en 4+4.

Design intent (§2.2): lever 1 only upsamples the pet_name shape, so the 8
non-pet cases test CROSS-DOMAIN generalization of the "compatible attribute
=> false" rule; pet memorization alone fails them honestly.  B2 stays in the
family for continuity with the v7-v10 readings.

Gate (HANDOFF_NLI_V11.md §1.2 #12, preregistered; cross-run, enforced by
protocol_verdict.py):  median over runs of family-mean p_conflict < 0.5, AND
every run has at most 1 of the 9 cases at p_conflict >= 0.9 (confident-wrong).

Usage: python family_diag.py <ckpt> <out_json>
"""
import json
import sys

import laya

from noul_bias_diag import QUESTION, CONTROL_PAIRS

B2 = next(((s, e), n) for (s, e), n in CONTROL_PAIRS if n.startswith('对照B2'))

FAMILY_CASES = [
    # (state, expect, note) -- B2-isomorphic: name/reference + compatible attribute
    (B2[0], '家族F0 (B2) 名字一致', True),
    (({'known': '用户的车叫小钢炮', 'new': '用户的车是红色的'}, 'false'), '家族F1 车名一致', False),
    (({'known': "The user's car goes by Flash", 'new': "The user's car is red"}, 'false'), '家族F2 car name', False),
    (({'known': '用户的手机叫圆方', 'new': '用户的手机是折叠屏'}, 'false'), '家族F3 手机名一致', False),
    (({'known': "The user's phone is named Brick", 'new': "The user's phone is a foldable"}, 'false'), '家族F4 phone name', False),
    (({'known': '用户的公路车叫飞燕', 'new': '用户的公路车是蓝色的'}, 'false'), '家族F5 车名一致', False),
    (({'known': "The user's bike is named Bessie", 'new': "The user's bike is electric"}, 'false'), '家族F6 bike name', False),
    (({'known': '用户的笔记本叫老黄牛', 'new': '用户的笔记本是十四寸的'}, 'false'), '家族F7 笔记本名一致', False),
    (({'known': "The user's laptop is called Old Faithful", 'new': "The user's laptop is silver"}, 'false'), '家族F8 laptop name', False),
]


def run(agent):
    results = []
    for (state, expect), note, is_b2 in FAMILY_CASES:
        a = agent.predict(state, {'conflict': QUESTION})['answers']['conflict']
        p_conflict = float(a['noul'])
        got = 'true' if p_conflict >= 0.5 else 'false'
        results.append({'note': note, 'expect': expect, 'got': got, 'p_conflict': round(p_conflict, 4),
                        'ok': got == expect, 'is_b2': is_b2})
    return results


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
    out = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\family_diag_v11_r1.json'
    agent = laya.load(ckpt, device='cpu')
    cases = run(agent)
    fam_mean = sum(c['p_conflict'] for c in cases) / len(cases)
    n_highconf = sum(1 for c in cases if c['p_conflict'] >= 0.9)
    n_ok = sum(c['ok'] for c in cases)
    for c in cases:
        print('OK  ' if c['ok'] else 'MISS', c['note'], c['expect'], '->', c['got'], c['p_conflict'])
    print(f'family: {n_ok}/9 ok | mean p_conflict {fam_mean:.4f} | p>=0.9 cases {n_highconf}')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': ckpt, 'cases': cases, 'family_mean_p': round(fam_mean, 4),
                   'n_highconf': n_highconf, 'n_ok': n_ok}, f, ensure_ascii=False, indent=2)
    print('saved:', out)


if __name__ == '__main__':
    main()
