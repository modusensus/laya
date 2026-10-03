# -*- coding: utf-8 -*-
"""HANDOFF_NLI_V11.md §1.2 (FINAL) protocol verdict tool.

Supersedes the draft-criteria version (commit 138c56c): the draft->final diff
IS the preregistered change and is recorded in V11 §4 (main val all-3 -> median;
B2 single-case clause -> 9-case family axis; polarity/band/hygiene gates made
explicit).  Executors and reviewers must not modify these criteria after the
first v11 run -- that is the entire point of preregistration.

Criteria (FINAL, gate numbers unchanged from v9/v10):
  1  main val acc     : median over runs >= 0.896
  2  old 20           : median == 20/20 AND no acceptance case missed in >=2 runs
  3  new 10           : every run 10/10
  4  negation 5       : every run 5/5
  5  val_soft         : every run <=3 errors
  6  swap             : every run PASS
  7  polarity real·diag : every run <=1     (polarity_nli_<tag>.json gates)
  8  polarity val_soft  : every run <=1
  9  band             : every run >=8pp    (conf_band_<tag>.txt gate line)
  10 conformal        : every run has an adopted rule >=50% capture @ <=35% abstain
  11 bias diag        : >=2/3 runs >=13/14 (B2 single-case clause removed)
  12 family (NEW)     : median over runs of family-mean p_conflict < 0.5,
                        AND every run has <=1 case at p_conflict >= 0.9
  13 hygiene          : leak_audit_v11.txt present and reports cases touched 0/35
  -  tau(noul)        : report-only (from the ckpt's rl_agent_config.json)

Usage: python protocol_verdict.py --tags v9,v10_s2,v10_l2     (v9 backtest)
       python protocol_verdict.py --tags r1,r2,r3             (v11 verdict)
Per-run artifacts resolve under data_local/ via the <prefix>_<tag>.json
convention (v11 runs use --prefix v11_).
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_local')
_OUT = r'D:\laya-kaggle-output'


def load(path):
    return json.load(open(os.path.join(_D, path), encoding='utf-8'))


def art(name, tag, pre, ext='.json'):
    """Resolve a per-run artifact path.  The battery emits <name>_v11_<tag>.json
    while the spec convention reads <prefix><name>_<tag>.json; try both (plus
    the bare <name>_<tag>.json used by the v9 backtest).  Gate logic never
    depends on which spelling exists."""
    cands = [f'{pre}{name}_{tag}{ext}',
             f"{name}_{pre.rstrip('_')}_{tag}{ext}" if pre else f'{name}_{tag}{ext}',
             f'{name}_{tag}{ext}']
    for cand in dict.fromkeys(cands):
        p = os.path.join(_D, cand)
        if os.path.exists(p):
            return p
    return os.path.join(_D, cands[0])   # miss -> FileNotFoundError names the spec spelling


def main_val_acc(tag, prefix):
    rows = load(art('val_probs', tag, prefix))
    err = sum(1 for r in rows if (r['p_true'] >= 0.5) != (r['gold'] == 1))
    return 1 - err / len(rows), err


def realtest(tag, prefix):
    d = load(art('memory_conflict_realtest', tag, prefix))
    old = sum(r['ok'] for r in d['cases'])
    neg = sum(r['ok'] for r in d['negation_cases'])
    new = sum(r['ok'] for r in d['new_cases'])
    misses = {r['note'] for r in d['cases'] if not r['ok']}
    return old, neg, new, misses


def bias_diag(tag, prefix):
    d = load(art('noul_bias_diag', tag, prefix))
    pairs = d['controls'] + d['swaps']
    return sum(r['ok'] for r in pairs), len(pairs)


def val_soft_err(tag, prefix):
    rows = load(art('val_soft_probs', tag, prefix))
    return sum(1 for r in rows if ('true' if r['p_true'] >= 0.5 else 'false') != r['label'])


def swap_verdict(tag, prefix):
    d = load(art('label_swap', tag, prefix))
    return d.get('verdict', d.get('pass'))


def polarity_gates(tag, prefix):
    d = load(art('polarity_nli', tag, prefix))
    g = d.get('gates', {})
    return g.get('real_diag_diff_le_1'), g.get('val_soft_err_change_le_1')


def band_gate(tag, prefix):
    p = art('conf_band', tag, prefix, ext='.txt')
    if not os.path.exists(p):
        return None
    txt = open(p, encoding='utf-8').read()
    if 'val_soft band gate (>=8pp): PASS' in txt:   # v10-era stdout phrasing
        return True
    for line in txt.splitlines():                   # file phrasing (r1 observed)
        if line.startswith('val_soft:') and '≥8pp' in line:
            if '⇒ PASS' in line:
                return True
            if '⇒ FAIL' in line:
                return False
    return None


def family_diag(tag, prefix):
    p = art('family_diag', tag, prefix)
    if not os.path.exists(p):
        return None, None  # pre-v11 runs have no family artifact -> axis FAILS
    d = json.load(open(p, encoding='utf-8'))
    return d['family_mean_p'], d['n_highconf']


def conformal_adopted(tag, prefix):
    out = []
    for rule in ('s1', 's2'):
        p = art('conformal', f'{tag}_{rule}', prefix)
        if os.path.exists(p):
            g = json.load(open(p, encoding='utf-8')).get('gates', {})
            if g:
                out.append((rule, g.get('capture'), g.get('capture_total'), g.get('abstain'), g.get('pass')))
    return out


def tau_report(tag):
    for name in (f'laya-nli-conflict-v11-{tag}', f'laya-nli-conflict-{tag}'):
        p = os.path.join(_OUT, name, 'rl_agent_config.json')
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding='utf-8'))['temperature'][2]
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', required=True, help='comma-separated run tags (3 runs)')
    ap.add_argument('--prefix', default='', help='artifact filename prefix, e.g. v11_ '
                                                 '(v9 backtest uses the bare convention)')
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    pre = args.prefix
    assert len(tags) == 3, 'v11 protocol fixes n=3'

    print(f'== V11 FINAL §1.2 protocol verdict over runs: {tags} (prefix "{pre}") ==\n')
    axes = {}

    # 1 main val: median >= 0.896
    accs = [main_val_acc(t, pre) for t in tags]
    med_acc = statistics.median(a for a, _ in accs)
    axes['1 main_val(median>=0.896)'] = med_acc >= 0.896
    print('main val:', [(t, round(a, 4), e) for t, (a, e) in zip(tags, accs)], '| median:', round(med_acc, 4))

    # 2 old 20: median == 20 AND no case missed in >=2 runs
    olds, negs, news, miss_sets = [], [], [], []
    for t in tags:
        old, neg, new, misses = realtest(t, pre)
        olds.append(old); negs.append(neg); news.append(new); miss_sets.append(misses)
    miss_counter = Counter(n for ms in miss_sets for n in ms)
    repeated = {k: v for k, v in miss_counter.items() if v >= 2}
    med_old = statistics.median(olds)
    axes['2 old20(median=20 AND no case missed in >=2 runs)'] = (med_old == 20 and not repeated)
    print('old20:', dict(zip(tags, olds)), '| median:', med_old,
          '| per-case miss counts:', dict(miss_counter) or '{}', '| repeated(>=2):', repeated or 'none')

    # 3/4/5/6: every-run hard gates
    axes['3 new10(all=10)'] = all(n == 10 for n in news)
    axes['4 negation(all=5)'] = all(n == 5 for n in negs)
    print('negation:', dict(zip(tags, negs)), '| new10:', dict(zip(tags, news)))
    vs = [val_soft_err(t, pre) for t in tags]
    axes['5 val_soft(all<=3)'] = all(v <= 3 for v in vs)
    print('val_soft err:', dict(zip(tags, vs)))
    sw = [swap_verdict(t, pre) for t in tags]
    axes['6 swap(all PASS)'] = all(s in (True, 'PASS') or s == 0 for s in sw)
    print('swap:', dict(zip(tags, sw)))

    # 7/8 polarity
    pols = [polarity_gates(t, pre) for t in tags]
    axes['7 polarity real·diag(all<=1)'] = all(p[0] is True for p in pols)
    axes['8 polarity val_soft(all<=1)'] = all(p[1] is True for p in pols)
    print('polarity gates (real·diag, val_soft):', dict(zip(tags, pols)))

    # 9 band
    bands = [band_gate(t, pre) for t in tags]
    axes['9 band(all >=8pp)'] = all(b is True for b in bands)
    print('band gate:', dict(zip(tags, bands)))

    # 10 conformal: every run's adopted rule passes
    conf = {t: conformal_adopted(t, pre) for t in tags}
    axes['10 conformal(adopted rule, all >=50%@<=35%)'] = all(
        any(gp is True for _, _, _, _, gp in runs) for runs in conf.values())
    for t, runs in conf.items():
        for rule, cap, tot, ab, gp in runs:
            print(f'  conformal {t} [{rule}]: {cap}/{tot} @ {ab} -> {"PASS" if gp else "FAIL"}' if tot
                  else f'  conformal {t} [{rule}]: {cap} @ {ab} -> {"PASS" if gp else "FAIL"}')

    # 11 bias diag: >=2/3 runs >= 13/14 (B2 single-case clause removed)
    diags = [bias_diag(t, pre) for t in tags]
    n_diag_pass = sum(1 for n_ok, n in diags if n_ok >= 13)
    axes['11 bias_diag(>=2/3 runs >=13/14)'] = n_diag_pass >= 2
    print('diag ok:', dict(zip(tags, [f'{n}/{m}' for n, m in diags])), f'| runs >=13/14: {n_diag_pass}/3')

    # 12 family (NEW)
    fams = [family_diag(t, pre) for t in tags]
    if any(fm is None for fm, _ in fams):
        med_fam = None
        axes['12 family(median mean-p<0.5 AND all runs <=1 case p>=0.9)'] = False
        print('family: MISSING artifact for', [t for t, (fm, _) in zip(tags, fams) if fm is None],
              '-> axis FAIL (v11 runs must emit family_diag_<tag>.json)')
    else:
        med_fam = statistics.median(fm for fm, _ in fams)
        axes['12 family(median mean-p<0.5 AND all runs <=1 case p>=0.9)'] = (
            med_fam < 0.5 and all(nh <= 1 for _, nh in fams))
        print('family mean p:', [(t, fm) for t, (fm, _) in zip(tags, fams)], '| median:', round(med_fam, 4),
              '| p>=0.9 cases per run:', dict(zip(tags, [nh for _, nh in fams])))

    # 13 hygiene
    hyg = os.path.join(_D, 'leak_audit_v11.txt')
    ok_hyg = os.path.exists(hyg) and 'cases touched: 0/35' in open(hyg, encoding='utf-8').read()
    axes['13 hygiene(0/35 + family cases 0 overlap asserted at gen)'] = ok_hyg
    print('leak_audit_v11.txt:', 'present, 0/35' if ok_hyg else 'MISSING or nonzero')

    # tau: report-only
    print('tau(noul) report-only:', {t: tau_report(t) for t in tags})

    print()
    verdict = all(axes.values())
    for k, v in axes.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f'\nPROTOCOL VERDICT: {"PASS -> eligible for HF delivery (§1.3)" if verdict else "FAIL -> no delivery, HF untouched"}')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
