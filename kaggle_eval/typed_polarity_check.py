# -*- coding: utf-8 -*-
"""Option-name polarity check (HANDOFF_NLI_V4.1.md, 改动1 ②, multilingual side).

Official LocalLLaMA/typed-decisions `all/test`, 400 cases / 2000 decisions
(pulled via hf-mirror 2026-09-28, sha f7a2487e; data files unchanged since
2026-09-16, same data as the card's ea930645 reference).

Steps (V4.1): ① baseline rerun on the full 400 cases — must match the card
(0.7875 / choice 0.752 / noul 0.862 / score 0.759) before any variant; ② choice
questions only: rename criteria KEYS (option names) keeping rubric values and
order fixed — neutral (A,B,C,... by position) and random strings (per question,
consonants+digits len 8, seed 20260928); map predictions back through the
position bijection; ③ flips vs baseline + McNemar on paired correctness.

Gates: flips <= 2% of choice decisions, McNemar p > 0.05.
Local 100-case file (data_local/typed_decisions_test.jsonl) is smoke-only.

Usage: python typed_polarity_check.py <ckpt> <out_json>
"""
import json
import math
import random
import sys

import pyarrow.parquet as pq

import laya

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya-kaggle-output\laya_finetuned_typed_decisions'
OUT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya\data_local\polarity_typed_v4.json'
PARQUET = r'D:\laya\data_local\typed_official\test.parquet'
SEED = 20260928
RAND_ALPHABET = 'BCDFGHJKLMNPQRSTVWXZ2456789'
NEUTRAL = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def load_cases():
    t = pq.read_table(PARQUET).to_pylist()
    assert len(t) == 400, f'rows = {len(t)}, expect 400'
    cases = []
    for r in t:
        cases.append({'id': r['id'],
                      'state': json.loads(r['state']) if isinstance(r['state'], str) else r['state'],
                      'questions': json.loads(r['questions']) if isinstance(r['questions'], str) else r['questions'],
                      'gold': json.loads(r['gold']) if isinstance(r['gold'], str) else r['gold']})
    return cases


def pred_label(ans, qtype):
    if qtype == 'noul':
        return 'true' if ans['noul'] >= 0.5 else 'false'
    if 'choice' in ans:
        return ans['choice']
    if ans.get('probabilities'):
        return max(ans['probabilities'], key=ans['probabilities'].get)
    return None


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    rng = random.Random(SEED)
    cases = load_cases()
    agent = laya.load(CKPT, device='cpu')

    # ---- ① baseline (full question set, one predict per case) ----
    base = {}
    type_stats = {}
    for i, case in enumerate(cases):
        answers = agent.predict(case['state'], case['questions'])['answers']
        for key, q in case['questions'].items():
            gt = case['gold'][key]
            got = pred_label(answers[key], q['type'])
            ok = (got == gt['label'])
            base[(case['id'], key)] = {'pred': got, 'ok': ok, 'type': q['type'],
                                       'gold': gt['label']}
            st = type_stats.setdefault(q['type'], [0, 0])
            st[0] += int(ok)
            st[1] += 1
        if (i + 1) % 50 == 0:
            print(f'  baseline {i + 1}/400', flush=True)
    total_ok = sum(s[0] for s in type_stats.values())
    total_n = sum(s[1] for s in type_stats.values())
    print('BASELINE:', {k: f'{v[0]}/{v[1]} = {v[0] / v[1]:.4f}' for k, v in type_stats.items()},
          f'total {total_ok}/{total_n} = {total_ok / total_n:.4f}', flush=True)

    # ---- ② choice-only variants ----
    choice_specs = [(c['id'], key, q) for c in cases for key, q in c['questions'].items()
                    if q['type'] == 'choice']
    print(f'choice decisions: {len(choice_specs)}', flush=True)
    out_variants = {}
    for variant in ['neutral', 'random']:
        flips, rows = [], []
        n_ok_std = n_ok_var = 0
        b = c2 = 0
        for j, (cid, key, q) in enumerate(choice_specs):
            orig_keys = list(q['criteria'].keys())
            if variant == 'neutral':
                mapping = {o: NEUTRAL[i] for i, o in enumerate(orig_keys)}
            else:
                mapping = {o: ''.join(rng.choices(RAND_ALPHABET, k=8)) for o in orig_keys}
            inv = {v: k for k, v in mapping.items()}
            q2 = {'type': 'choice', 'instructions': q['instructions'],
                  'criteria': {mapping[o]: q['criteria'][o] for o in orig_keys}}
            state = next(x['state'] for x in cases if x['id'] == cid)
            ans = agent.predict(state, {key: q2})['answers'][key]
            probs = ans.get('probabilities') or {}
            pred_new = max(probs, key=probs.get) if probs else ans.get('choice')
            pred_orig = inv.get(pred_new, pred_new)
            gold = base[(cid, key)]['gold']
            ok_var = (pred_orig == gold)
            ok_std = base[(cid, key)]['ok']
            if pred_orig != base[(cid, key)]['pred']:
                flips.append({'case': cid, 'q': key, 'std': base[(cid, key)]['pred'],
                              'var': pred_orig, 'gold': gold})
            n_ok_std += int(ok_std)
            n_ok_var += int(ok_var)
            b += int(ok_std and not ok_var)
            c2 += int((not ok_std) and ok_var)
            rows.append({'case': cid, 'q': key, 'std_pred': base[(cid, key)]['pred'],
                         'var_pred': pred_orig, 'gold': gold})
            if (j + 1) % 100 == 0:
                print(f'  {variant} {j + 1}/{len(choice_specs)}', flush=True)
        n = len(choice_specs)
        p = mcnemar_exact(b, c2)
        out_variants[variant] = {
            'flips': flips, 'flip_count': len(flips), 'flip_rate': round(len(flips) / n, 4),
            'acc_std': round(n_ok_std / n, 4), 'acc_var': round(n_ok_var / n, 4),
            'mcnemar': {'b_std_right_var_wrong': b, 'c_std_wrong_var_right': c2, 'p': round(p, 4)},
            'verdict': 'PASS' if (len(flips) <= 0.02 * n and p > 0.05) else 'FAIL'}
        print(f"{variant}: flips {len(flips)}/{n} ({100 * len(flips) / n:.2f}%), "
              f"acc {n_ok_std / n:.4f} -> {n_ok_var / n:.4f}, McNemar b={b} c={c2} p={p:.4f}", flush=True)

    gate = all(v['verdict'] == 'PASS' for v in out_variants.values())
    json.dump({'checkpoint': CKPT, 'parquet': PARQUET,
               'dataset_sha': 'f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8',
               'baseline': {k: {'acc': round(v[0] / v[1], 4), 'n': v[1]} for k, v in type_stats.items()}
               | {'total': round(total_ok / total_n, 4)}},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    # rows are large; write them in a second pass key
    full = json.load(open(OUT, encoding='utf-8'))
    full['choice_variants'] = out_variants
    full['choice_variant_rows'] = {v: out_variants[v]['flips'] for v in out_variants}
    full['gates'] = {'flips_le_2pct_and_mcnemar_p_gt_0.05': gate, 'verdict': 'PASS' if gate else 'FAIL'}
    json.dump(full, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print('GATES:', full['gates'])
    print('saved:', OUT)
    sys.exit(0 if gate else 1)


if __name__ == '__main__':
    main()
