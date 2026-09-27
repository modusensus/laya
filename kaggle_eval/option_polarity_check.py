# -*- coding: utf-8 -*-
"""Option-name polarity check (HANDOFF_NLI_V4.1.md, 改动1 ①, NLI side).

arXiv:2609.26758 axis: the SEMANTICS of the option-name system itself
(0/1 vs no/yes vs random strings), distinct from the value-word swap axis
already covered by label_swap_check.py. Three renderings of the same rubric:
  std     labels {"false": "兼容", "true": "冲突"}
  neutral labels {"false": "A",    "true": "B"}      (fixed mapping)
  random  per-case random strings, no vowels/ambiguous chars, len 8,
          seed 20260928 (rule in V4.1 改动1)

Sets: real 20 + bias-diag 14 + val_soft 300. Answer key stays `noul`
(single-probe verified 2026-09-28); decision = p_true >= 0.5; p-shift
(mean |dp|) reported as an observation, not a gate.

Gates (V4.1): real20 & diag14 each |diff| <= 1 per variant (both directions);
val_soft error-count change <= 1.

Usage: python option_polarity_check.py <ckpt> <out_json>
"""
import json
import random
import string
import sys

import laya

sys.path.insert(0, r'D:\laya\kaggle_eval')
from realtest_v2 import CASES                                   # noqa: E402  frozen old-20
from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS            # noqa: E402  diag-14

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\polarity_nli_v4.json'
SEED = 20260928
RAND_ALPHABET = 'BCDFGHJKLMNPQRSTVWXZ2456789'  # no vowels -> no readable words by construction


def random_label(rng):
    return ''.join(rng.choices(RAND_ALPHABET, k=8))


def make_question(base_instructions, variant, rng):
    q = {'type': 'noul', 'instructions': base_instructions}
    if variant == 'std':
        q['labels'] = {'false': '兼容', 'true': '冲突'}
    elif variant == 'neutral':
        q['labels'] = {'false': 'A', 'true': 'B'}
    else:
        q['labels'] = {'false': random_label(rng), 'true': random_label(rng)}
    return q


def collect_sets():
    sets = {'real': {'expect': [(s, e) for (s, e), _ in CASES],
                     'notes': [n for _, n in CASES],
                     'instr': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关'}}
    diag = [(s, e) for (s, e), _ in (CONTROL_PAIRS + SWAP_PAIRS)]
    sets['diag'] = {'expect': diag, 'notes': [n for _, n in (CONTROL_PAIRS + SWAP_PAIRS)],
                    'instr': sets['real']['instr']}
    rows = [json.loads(l) for l in open(r'D:\laya\data_local\nli_conflict_val_soft.jsonl', encoding='utf-8')]
    assert len(rows) == 300, f'val_soft rows = {len(rows)}, expect 300'
    vs_pairs, vs_instr = [], []
    for r in rows:
        state = json.loads(r['state']) if isinstance(r['state'], str) else r['state']
        vs_pairs.append((state, r['gold']['conflict']['label']))
        vs_instr.append(r['questions']['conflict']['instructions'])
    assert len(set(vs_instr)) == 1, 'val_soft instructions not uniform?'
    sets['val_soft'] = {'expect': vs_pairs, 'notes': [f'vs{i:03d}' for i in range(len(rows))],
                        'instr': vs_instr[0]}
    return sets


def main():
    rng = random.Random(SEED)
    sets = collect_sets()
    agent = laya.load(CKPT, device='cpu')
    out = {'checkpoint': CKPT, 'seed': SEED,
           'random_rule': {'alphabet': 'consonants+digits (no vowels/ambiguous)', 'len': 8, 'per_case': True}}
    verdicts = {}
    for set_name, spec in sets.items():
        per_variant = {}
        for variant in ['std', 'neutral', 'random']:
            preds = []
            for i, (state, expect) in enumerate(spec['expect']):
                q = make_question(spec['instr'], variant, rng)
                p = float(agent.predict(state, {'conflict': q})['answers']['conflict']['noul'])
                preds.append({'note': spec['notes'][i], 'expect': expect,
                              'p_true': round(p, 4), 'got': 'true' if p >= 0.5 else 'false'})
                if (i + 1) % 100 == 0:
                    print(f'  {set_name}/{variant}: {i + 1}/{len(spec["expect"])}', flush=True)
            per_variant[variant] = preds
            errs = sum(1 for r in preds if r['got'] != r['expect'])
            print(f'{set_name}/{variant}: err={errs}', flush=True)
        std = per_variant['std']
        row = {'acc': {}, 'err': {}, 'mean_abs_dp': {}}
        for variant in ['neutral', 'random']:
            var = per_variant[variant]
            diff = sum(r['got'] == r['expect'] for r in std) - sum(r['got'] == r['expect'] for r in var)
            flips = [(a['note'], a['got'], b['got']) for a, b in zip(std, var) if a['got'] != b['got']]
            row[variant] = {'diff': diff, 'flips': flips,
                            'mean_abs_dp': round(sum(abs(a['p_true'] - b['p_true'])
                                                     for a, b in zip(std, var)) / len(std), 4)}
            row['acc'][variant] = sum(r['got'] == r['expect'] for r in var)
        row['acc']['std'] = sum(r['got'] == r['expect'] for r in std)
        row['err'] = {v: sum(r['got'] != r['expect'] for r in per_variant[v])
                      for v in ['std', 'neutral', 'random']}
        verdicts[set_name] = row
        out[set_name] = {'per_variant': per_variant, 'summary': row}
        print(f'{set_name}: acc std/neutral/random = '
              f'{row["acc"]["std"]}/{row["acc"]["neutral"]}/{row["acc"]["random"]}, '
              f'err = {row["err"]["std"]}/{row["err"]["neutral"]}/{row["err"]["random"]}', flush=True)

    g1 = all(abs(verdicts[s]['neutral']['diff']) <= 1 and abs(verdicts[s]['random']['diff']) <= 1
             for s in ['real', 'diag'])
    g2 = all(abs(verdicts['val_soft']['err']['std'] - verdicts['val_soft']['err'][v]) <= 1
             for v in ['neutral', 'random'])
    out['gates'] = {'real_diag_diff_le_1': g1, 'val_soft_err_change_le_1': g2,
                    'verdict': 'PASS' if (g1 and g2) else 'FAIL'}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('GATES:', out['gates'])
    print('saved:', OUT)
    sys.exit(0 if (g1 and g2) else 1)


if __name__ == '__main__':
    main()
