# -*- coding: utf-8 -*-
"""Label-surface swap check (HANDOFF_NLI_V3.md, 改动3).

Re-runs the frozen real-case 20 and the 14 bias-diag pairs with the labels'
two display values swapped (兼容 <-> 冲突) while semantics/instructions stay
put. A model still reading the label WORDS flips its answers; a model that
learned semantics does not. Gate: accuracy difference <= 1 case per set.

Usage: python label_swap_check.py <ckpt> <out_json>
"""
import json
import sys

import laya

sys.path.insert(0, r'D:\laya\kaggle_eval')
from realtest_v2 import CASES          # noqa: E402  frozen old-20, verbatim
from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\label_swap_v4.json'

Q_STD = {'type': 'noul',
         'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
         'labels': {'false': '兼容', 'true': '冲突'}}
Q_SWAP = {'type': 'noul',
          'instructions': Q_STD['instructions'],
          'labels': {'false': '冲突', 'true': '兼容'}}  # surface values exchanged


def run(agent, pairs, question):
    out = []
    for (state, expect), note in pairs:
        p = float(agent.predict(state, {'conflict': question})['answers']['conflict']['noul'])
        got = 'true' if p >= 0.5 else 'false'
        out.append({'note': note, 'expect': expect, 'got': got,
                    'p_conflict': round(p, 4), 'ok': got == expect})
    return out


def main():
    agent = laya.load(CKPT, device='cpu')
    real_std = run(agent, CASES, Q_STD)
    real_swap = run(agent, CASES, Q_SWAP)
    diag_pairs = CONTROL_PAIRS + SWAP_PAIRS
    diag_std = run(agent, diag_pairs, Q_STD)
    diag_swap = run(agent, diag_pairs, Q_SWAP)

    def acc(rows):
        return sum(r['ok'] for r in rows)
    real_diff = acc(real_std) - acc(real_swap)
    diag_diff = acc(diag_std) - acc(diag_swap)
    ok = abs(real_diff) <= 1 and abs(diag_diff) <= 1

    print(f'real-20 : std {acc(real_std)}/20 vs swap {acc(real_swap)}/20 (diff {real_diff:+d})')
    print(f'diag-14 : std {acc(diag_std)}/14 vs swap {acc(diag_swap)}/14 (diff {diag_diff:+d})')
    print('LABEL-SWAP:', 'PASS' if ok else 'FAIL', '(gate: |diff| <= 1 per set)')
    flipped = [(a['note'], a['got'], b['got']) for a, b in zip(real_std, real_swap) if a['got'] != b['got']]
    flipped += [(a['note'], a['got'], b['got']) for a, b in zip(diag_std, diag_swap) if a['got'] != b['got']]
    for note, g_std, g_swap in flipped:
        print(f'  flipped on swap: {note}: std={g_std} swap={g_swap}')
    json.dump({'real_std': real_std, 'real_swap': real_swap,
               'diag_std': diag_std, 'diag_swap': diag_swap,
               'acc': {'real_std': acc(real_std), 'real_swap': acc(real_swap),
                       'diag_std': acc(diag_std), 'diag_swap': acc(diag_swap)},
               'diff': {'real': real_diff, 'diag': diag_diff},
               'verdict': 'PASS' if ok else 'FAIL'},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved:', OUT)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
