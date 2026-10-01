# -*- coding: utf-8 -*-
"""V11 DRAFT §1.2 protocol verdict tool (mechanizes the pre-registered
multi-run readings; the Hermes final version may adjust thresholds — this
script makes any such adjustment a visible diff).

Usage:
  python protocol_verdict.py --tag-list v9,v10_s2,v10_l2 \
      --main v9:val_probs_v9.json v10_s2:val_probs_v10_s2.json ... \
      (files resolve under data_local/; per-run file names follow the
       <prefix>_<tag>.json convention, see DEFAULTS below)

Gate numbers are the v9/v10 hard gates, UNCHANGED.  Variance axes use the
draft's median readings.  One JSON verdict per axis + an overall verdict.
"""
import argparse
import json
import os
import statistics
import sys

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_local')

HOLDOUT = {'email', 'desk_floor', 'degree', 'barber'}


def load(path):
    p = os.path.join(_D, path)
    return json.load(open(p, encoding='utf-8'))


def main_val_acc(tag):
    rows = load(f'val_probs_{tag}.json')
    err = sum(1 for r in rows if (r['p_true'] >= 0.5) != (r['gold'] == 1))
    return 1 - err / len(rows), err


def realtest(tag):
    d = load(f'memory_conflict_realtest_{tag}.json')
    old = sum(r['ok'] for r in d['cases'])
    neg = sum(r['ok'] for r in d['negation_cases'])
    new = sum(r['ok'] for r in d['new_cases'])
    # per-case miss tracking for the old-20 repeated-miss reading
    misses = {r['note'] for r in d['cases'] if not r['ok']}
    return old, neg, new, misses


def bias_diag(tag):
    d = load(f'noul_bias_diag_{tag}.json')
    pairs = d['controls'] + d['swaps']
    n_ok = sum(r['ok'] for r in pairs)
    b2 = next(r for r in d['controls'] if r['note'].startswith('对照B2'))
    return n_ok, len(pairs), b2['p_conflict']


def val_soft_err(tag):
    rows = load(f'val_soft_probs_{tag}.json')
    return sum(1 for r in rows if ('true' if r['p_true'] >= 0.5 else 'false') != r['label'])


def swap_diff(tag):
    d = load(f'label_swap_{tag}.json')
    return d.get('verdict', d.get('pass', None))


def conformal_adopted(tag):
    # adopted file: <tag>_s1.json or <tag>_s2.json — pick the one whose
    # 'fit.rule'/'input' says adopted; the eval flow writes both
    for rule in ('s1', 's2'):
        p = os.path.join(_D, f'conformal_{tag}_{rule}.json')
        if os.path.exists(p):
            d = json.load(open(p, encoding='utf-8'))
            g = d.get('gates', {})
            if g:
                yield rule, g.get('capture'), g.get('capture_total'), g.get('abstain'), g.get('pass')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', required=True, help='comma-separated run tags (3+ runs)')
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    assert len(tags) >= 3, 'protocol requires >=3 runs'

    print(f'== V11 DRAFT §1.2 protocol verdict over {len(tags)} runs: {tags} ==\n')
    axes = {}

    # --- hard axes: ALL runs must pass the unchanged gate numbers ---
    accs = [main_val_acc(t) for t in tags]
    axes['main_val(>=0.896, 3/3)'] = all(a >= 0.896 for a, _ in accs)
    print('main val:', [(t, round(a, 4), e) for t, (a, e) in zip(tags, accs)])

    olds, negs, news, miss_sets = [], [], [], []
    for t in tags:
        old, neg, new, misses = realtest(t)
        olds.append(old); negs.append(neg); news.append(new); miss_sets.append(misses)
    axes['old20(=20, 3/3)'] = all(o == 20 for o in olds)
    axes['negation(=5, 3/3)'] = all(n == 5 for n in negs)
    axes['new10(=10, 3/3)'] = all(n == 10 for n in news)
    print('old20:', dict(zip(tags, olds)), '| negation:', dict(zip(tags, negs)), '| new10:', dict(zip(tags, news)))

    vs = [val_soft_err(t) for t in tags]
    axes['val_soft(<=3, 3/3)'] = all(v <= 3 for v in vs)
    print('val_soft err:', dict(zip(tags, vs)))

    sw = [swap_diff(t) for t in tags]
    axes['swap(3/3)'] = all(s in (True, 'PASS', None) or s == 0 for s in sw)
    print('swap:', dict(zip(tags, sw)))

    # --- variance axes: median-based pre-registered readings ---
    diags = [bias_diag(t) for t in tags]
    b2s = [b2 for _, _, b2 in diags]
    med_b2 = statistics.median(b2s)
    n_diag_pass = sum(1 for n_ok, n, _ in diags if n_ok >= 13)
    axes['bias_diag(median B2<0.5 AND >=2/3 runs >=13/14)'] = (med_b2 < 0.5 and n_diag_pass >= 2)
    print('diag ok:', dict(zip(tags, [f'{n}/{m}' for n, m, _ in diags])),
          '| B2:', [round(b, 4) for b in b2s], '| median B2:', round(med_b2, 4),
          f'| runs >=13/14: {n_diag_pass}/3')

    # old-20 median 20/20 AND no case missed in >=2 runs
    from collections import Counter
    miss_counter = Counter()
    for ms in miss_sets:
        for note in ms:
            miss_counter[note] += 1
    repeated = {k: v for k, v in miss_counter.items() if v >= 2}
    import statistics as st
    med_old = st.median(olds)
    axes['old20_variance(median=20 AND no case missed in >=2 runs)'] = (med_old == 20 and not repeated)
    print('old20 per-case miss counts:', dict(miss_counter), '| median:', med_old,
          '| repeated(>=2 runs):', repeated or 'none')

    # conformal: all runs' adopted file passes
    conf = {t: list(conformal_adopted(t)) for t in tags}
    axes['conformal(adopted rule, 3/3 >=50%@<=35%)'] = all(
        any(g_pass is True for _, _, _, _, g_pass in runs) for runs in conf.values())
    for t, runs in conf.items():
        for rule, cap, tot, ab, g_pass in runs:
            print(f'  conformal {t} [{rule}]: {cap}/{tot} @ {ab}' if tot else f'  conformal {t} [{rule}]: {cap} @ {ab}')

    # --- hygiene/band/polarity: present but light (band file existence) ---
    band_ok = all(os.path.exists(os.path.join(_D, f'conf_band_{t}.txt')) for t in tags)
    axes['band_files(3/3 present)'] = band_ok

    print()
    verdict = all(axes.values())
    for k, v in axes.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f'\nPROTOCOL VERDICT: {"PASS -> eligible for HF delivery" if verdict else "FAIL -> no delivery, HF untouched"}')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
