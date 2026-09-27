"""Laya typed-decisions multilingual — option-name polarity check on Kaggle GPU.

Round-4 改动1 ② (HANDOFF_NLI_V4.1.md). Inference-only (no training):
baseline rerun on the official LocalLLaMA/typed-decisions all/test (400 cases),
then choice-question option-name renames (neutral A/B..., per-question random
strings) mapped back through the position bijection; flips + McNemar.

Model pinned: Modusnsus/laya-typed-decisions-multilingual revision
d4385065972ea1f5b0a8495fac40016233a02b8e (weights sha256 8bb8cdd4…2718).
Dataset pinned: LocalLLaMA/typed-decisions all/test parquet at main
(f7a2487e…; data files unchanged since 2026-09-16).

Output: /kaggle/working/polarity_typed_v4.json
"""
import os
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'laya==0.3.20'], check=True)

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

import laya  # noqa: E402

MODEL_REV = 'd4385065972ea1f5b0a8495fac40016233a02b8e'
SEED = 20260928
RAND_ALPHABET = 'BCDFGHJKLMNPQRSTVWXZ2456789'
NEUTRAL = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
OUT = '/kaggle/working/polarity_typed_v4.json'


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pred_label(ans, qtype):
    if qtype == 'noul':
        return 'true' if ans['noul'] >= 0.5 else 'false'
    if 'choice' in ans:
        return ans['choice']
    if ans.get('probabilities'):
        return max(ans['probabilities'], key=ans['probabilities'].get)
    return None


def main():
    rng = random.Random(SEED)
    pq_path = hf_hub_download('LocalLLaMA/typed-decisions', 'all/test-00000-of-00001.parquet',
                              repo_type='dataset', revision='f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8')
    t = pq.read_table(pq_path).to_pylist()
    assert len(t) == 400, f'rows = {len(t)}'
    cases = [{'id': r['id'],
              'state': json.loads(r['state']) if isinstance(r['state'], str) else r['state'],
              'questions': json.loads(r['questions']) if isinstance(r['questions'], str) else r['questions'],
              'gold': json.loads(r['gold']) if isinstance(r['gold'], str) else r['gold']} for r in t]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, flush=True)
    # NOTE: PyPI laya 0.3.20 load() has no `revision` kwarg (local editable build does);
    # hub default resolves to the latest revision d438506…, identity pinned by the
    # weights SHA256 recorded on the card (8bb8cdd4…2718).
    agent = laya.load('Modusnsus/laya-typed-decisions-multilingual', device=device)

    base, type_stats = {}, {}
    for i, case in enumerate(cases):
        answers = agent.predict(case['state'], case['questions'])['answers']
        for key, q in case['questions'].items():
            gt = case['gold'][key]
            got = pred_label(answers[key], q['type'])
            base[(case['id'], key)] = {'pred': got, 'ok': got == gt['label'], 'gold': gt['label']}
            st = type_stats.setdefault(q['type'], [0, 0])
            st[0] += int(got == gt['label'])
            st[1] += 1
        if (i + 1) % 50 == 0:
            print(f'baseline {i + 1}/400', flush=True)
    tok = sum(s[0] for s in type_stats.values())
    ton = sum(s[1] for s in type_stats.values())
    print('BASELINE:', {k: f'{v[0]}/{v[1]}={v[0] / v[1]:.4f}' for k, v in type_stats.items()},
          f'total {tok}/{ton}={tok / ton:.4f}', flush=True)

    choice_specs = [(c['id'], key, q) for c in cases for key, q in c['questions'].items()
                    if q['type'] == 'choice']
    state_by_id = {c['id']: c['state'] for c in cases}
    out_variants = {}
    for variant in ['neutral', 'random']:
        flips, n_ok_std, n_ok_var, b, c2 = [], 0, 0, 0, 0
        for j, (cid, key, q) in enumerate(choice_specs):
            orig_keys = list(q['criteria'].keys())
            if variant == 'neutral':
                mapping = {o: NEUTRAL[i] for i, o in enumerate(orig_keys)}
            else:
                mapping = {o: ''.join(rng.choices(RAND_ALPHABET, k=8)) for o in orig_keys}
            inv = {v: k for k, v in mapping.items()}
            q2 = {'type': 'choice', 'instructions': q['instructions'],
                  'criteria': {mapping[o]: q['criteria'][o] for o in orig_keys}}
            ans = agent.predict(state_by_id[cid], {key: q2})['answers'][key]
            probs = ans.get('probabilities') or {}
            pred_new = max(probs, key=probs.get) if probs else ans.get('choice')
            pred_orig = inv.get(pred_new, pred_new)
            gold = base[(cid, key)]['gold']
            ok_var, ok_std = pred_orig == gold, base[(cid, key)]['ok']
            if pred_orig != base[(cid, key)]['pred']:
                flips.append({'case': cid, 'q': key, 'std': base[(cid, key)]['pred'],
                              'var': pred_orig, 'gold': gold})
            n_ok_std += int(ok_std)
            n_ok_var += int(ok_var)
            b += int(ok_std and not ok_var)
            c2 += int((not ok_std) and ok_var)
            if (j + 1) % 100 == 0:
                print(f'{variant} {j + 1}/{len(choice_specs)}', flush=True)
        n = len(choice_specs)
        p = mcnemar_exact(b, c2)
        out_variants[variant] = {
            'flips': flips, 'flip_count': len(flips), 'flip_rate': round(len(flips) / n, 4),
            'acc_std': round(n_ok_std / n, 4), 'acc_var': round(n_ok_var / n, 4),
            'mcnemar': {'b': b, 'c': c2, 'p': round(p, 4)},
            'verdict': 'PASS' if (len(flips) <= 0.02 * n and p > 0.05) else 'FAIL'}
        print(f"{variant}: flips {len(flips)}/{n} ({100 * len(flips) / n:.2f}%), "
              f"acc {n_ok_std / n:.4f}->{n_ok_var / n:.4f}, b={b} c={c2} p={p:.4f}", flush=True)

    gate = all(v['verdict'] == 'PASS' for v in out_variants.values())
    result = {'model_rev': MODEL_REV, 'device': device,
              'baseline': {**{k: {'acc': round(v[0] / v[1], 4), 'n': v[1]} for k, v in type_stats.items()},
                           'total': round(tok / ton, 4)},
              'choice_variants': out_variants,
              'gates': {'flips_le_2pct_and_mcnemar_p_gt_0.05': gate,
                        'verdict': 'PASS' if gate else 'FAIL'}}
    json.dump(result, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print('GATES:', result['gates'])
    print('saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
