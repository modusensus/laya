# -*- coding: utf-8 -*-
"""Bias diagnosis for the v2 noul conflict head (acceptance gate #2, upstream #156/#220).

A model with the boolean-label-word bias answers the same label regardless of input.
This script builds minimal control pairs that share one `known` and differ only in
whether `new` contradicts it, plus literal input swaps of conflict/unrelated pairs.
PASS requires:
  1. control accuracy >= 0.9 (outputs track the input), and
  2. neither label takes >= 90% of predictions (outputs are not pinned).
"""
import json
import sys

import laya

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\石晴\Desktop\laya-kaggle-output\laya-nli-conflict-v2'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'C:\Users\石晴\Desktop\laya\data_local\noul_bias_diag_v2.json'

QUESTION = {'type': 'noul',
            'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
            'labels': {'false': '兼容', 'true': '冲突'}}

# minimal pairs: same known, new flips only the conflicted attribute
CONTROL_PAIRS = [
    # (state, expect, note)
    (({'known': '用户住在上海', 'new': '用户最近搬到北京工作了'}, 'true'), '对照A1 地点变更'),
    (({'known': '用户住在上海', 'new': '上海有很多大学'}, 'false'), '对照A2 地点无关'),
    (({'known': '用户的猫叫小白', 'new': '用户的猫叫小黑'}, 'true'), '对照B1 名字矛盾'),
    (({'known': '用户的猫叫小白', 'new': '用户的猫是白色的'}, 'false'), '对照B2 名字一致'),
    (({'known': '用户今年28岁', 'new': '用户今年35岁'}, 'true'), '对照C1 年龄矛盾'),
    (({'known': '用户今年28岁', 'new': '用户喜欢爬山'}, 'false'), '对照C2 年龄无关'),
    (({'known': '用户对花生过敏', 'new': '用户说他可以吃花生酱了'}, 'true'), '对照D1 过敏反转'),
    (({'known': '用户对花生过敏', 'new': '用户昨天吃了开心果'}, 'false'), '对照D2 其他坚果无关'),
    (({'known': '用户的项目部署在阿里云', 'new': '项目已经迁移到腾讯云了'}, 'true'), '对照E1 部署变更'),
    (({'known': '用户的项目部署在阿里云', 'new': '项目上周发布了新版本'}, 'false'), '对照E2 部署兼容'),
]

# literal input swaps: a conflicting or unrelated pair keeps its label when known/new trade places
SWAP_PAIRS = [
    (({'known': '用户的猫叫小黑', 'new': '用户的猫叫小白'}, 'true'), '交换B1'),
    (({'known': '上海有很多大学', 'new': '用户住在上海'}, 'false'), '交换A2'),
    (({'known': '用户今年35岁', 'new': '用户今年28岁'}, 'true'), '交换C1'),
    (({'known': '用户住在上海', 'new': '用户最近搬到北京工作了'}, 'true'), '交换A1'),
]


def run(agent, pairs):
    results = []
    for (state, expect), note in pairs:
        a = agent.predict(state, {'conflict': QUESTION})['answers']['conflict']
        p_conflict = float(a['noul'])          # answer key confirmed: noul = P(冲突)
        got = 'true' if p_conflict >= 0.5 else 'false'
        results.append({'note': note, 'expect': expect, 'got': got,
                        'p_conflict': p_conflict, 'ok': got == expect})
    return results


def main():
    agent = laya.load(CKPT, device='cpu')
    controls = run(agent, CONTROL_PAIRS)
    swaps = run(agent, SWAP_PAIRS)
    all_r = controls + swaps
    acc = sum(r['ok'] for r in all_r) / len(all_r)
    n_true = sum(r['got'] == 'true' for r in all_r)
    true_frac = n_true / len(all_r)
    tracks_input = acc >= 0.9
    not_pinned = 0.1 < true_frac < 0.9
    verdict = 'PASS' if (tracks_input and not_pinned) else 'FAIL'
    for r in all_r:
        print('OK  ' if r['ok'] else 'MISS', r['note'], r['expect'], '->', r['got'], round(r['p_conflict'], 4))
    print(f'control+swap accuracy: {sum(r["ok"] for r in all_r)}/{len(all_r)} '
          f'| predicted-true fraction: {n_true}/{len(all_r)}')
    print(f'tracks_input={tracks_input} not_pinned={not_pinned} => BIAS DIAG: {verdict}')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': CKPT, 'verdict': verdict,
                   'accuracy': acc, 'true_fraction': true_frac,
                   'tracks_input': tracks_input, 'not_pinned': not_pinned,
                   'controls': controls, 'swaps': swaps}, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
