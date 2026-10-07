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
       python protocol_verdict.py --protocol v12 --tags r1,r2,r3  (v12 verdict)
Per-run artifacts resolve under data_local/ via the <prefix>_<tag>.json
convention (v11/v12 runs use --prefix v11_/v12_; the resolver accepts the
battery's actual <name>_<prefix>_<tag>.json spelling too).

v12 protocol deltas (HANDOFF_NLI_V12.md §1.2, preregistered 2026-10-06):
  #3/#4 new10/negation5: median over runs = full AND no case missed in >=2 runs
  #13 negfam (NEW, 8 cases gold=true): median family-mean p_conflict >= 0.5,
      AND every run has <=1 case at p_conflict < 0.5
  #14 hygiene: leak_audit_v12.txt 0/35 (gen asserts 0/51 with both families)
  report-only additions: automation@5% per run (max coverage at <=5% selective
      error from val_probs; v11 baseline 74.3/78.6/79.3), tau(noul)
The default protocol stays v11 so the frozen v11 verdict remains reproducible.
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
    miss_old = {r['note'] for r in d['cases'] if not r['ok']}
    miss_neg = {r['note'] for r in d['negation_cases'] if not r['ok']}
    miss_new = {r['note'] for r in d['new_cases'] if not r['ok']}
    return old, neg, new, miss_old, miss_neg, miss_new


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


def negfam_diag(tag, prefix):
    p = art('negfam_diag', tag, prefix)
    if not os.path.exists(p):
        return None, None  # runs without the v12 instrument -> axis FAILS
    d = json.load(open(p, encoding='utf-8'))
    return d['family_mean_p'], d['n_lowconf']


def automation5(tag, prefix):
    """Report-only: max coverage at <=5% selective error, from val_probs
    (confidence = distance from 0.5, descending)."""
    rows = load(art('val_probs', tag, prefix))
    conf = sorted(((max(r['p_true'], 1 - r['p_true']), (r['p_true'] >= 0.5) == (r['gold'] == 1))
                   for r in rows), reverse=True)
    best, err = 0, 0
    for k, (_c, ok) in enumerate(conf, 1):
        if not ok:
            err += 1
        if err / k <= 0.05:
            best = k
    return best / len(rows)


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
    for name in (f'laya-nli-conflict-v13-{tag}', f'laya-nli-conflict-v12-{tag}',
                 f'laya-nli-conflict-v11-{tag}', f'laya-nli-conflict-{tag}'):
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
    ap.add_argument('--protocol', default='v11', choices=('v11', 'v12', 'v13'),
                    help='v11 = frozen HANDOFF_NLI_V11 criteria; v12 = HANDOFF_NLI_V12 §1.2 '
                         '(median new10/negation5 + negfam axis + 0/51 hygiene); '
                         'v13 = HANDOFF_NLI_V13 (main-val gate re-estimated to 0.900, else v12)')
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    pre = args.prefix
    proto = args.protocol
    assert len(tags) == 3, 'v11/v12 protocol fixes n=3'

    print(f'== {proto.upper()} §1.2 protocol verdict over runs: {tags} (prefix "{pre}") ==\n')
    axes = {}

    # 1 main val: median >= 0.896 (v11/v12) / 0.900 (v13 re-estimate, HANDOFF_NLI_V13 §3-1)
    gate_val = 0.900 if proto == 'v13' else 0.896
    accs = [main_val_acc(t, pre) for t in tags]
    med_acc = statistics.median(a for a, _ in accs)
    axes[f'1 main_val(median>={gate_val})'] = med_acc >= gate_val
    print('main val:', [(t, round(a, 4), e) for t, (a, e) in zip(tags, accs)], '| median:', round(med_acc, 4),
          f'| gate: {gate_val}')

    # 2 old 20: median == 20 AND no case missed in >=2 runs
    olds, negs, news = [], [], []
    miss_old_all, miss_neg_all, miss_new_all = [], [], []
    for t in tags:
        old, neg, new, mo, mn, mnw = realtest(t, pre)
        olds.append(old); negs.append(neg); news.append(new)
        miss_old_all.append(mo); miss_neg_all.append(mn); miss_new_all.append(mnw)
    miss_counter = Counter(n for ms in miss_old_all for n in ms)
    repeated = {k: v for k, v in miss_counter.items() if v >= 2}
    med_old = statistics.median(olds)
    axes['2 old20(median=20 AND no case missed in >=2 runs)'] = (med_old == 20 and not repeated)
    print('old20:', dict(zip(tags, olds)), '| median:', med_old,
          '| per-case miss counts:', dict(miss_counter) or '{}', '| repeated(>=2):', repeated or 'none')

    # 3/4: v11 = every-run hard gates; v12 = median full + no case missed in >=2 runs
    if proto == 'v11':
        axes['3 new10(all=10)'] = all(n == 10 for n in news)
        axes['4 negation(all=5)'] = all(n == 5 for n in negs)
    else:
        cnt_new = Counter(n for ms in miss_new_all for n in ms)
        rep_new = {k: v for k, v in cnt_new.items() if v >= 2}
        cnt_neg = Counter(n for ms in miss_neg_all for n in ms)
        rep_neg = {k: v for k, v in cnt_neg.items() if v >= 2}
        axes['3 new10(median=10 AND no case missed in >=2 runs)'] = (
            statistics.median(news) == 10 and not rep_new)
        axes['4 negation(median=5 AND no case missed in >=2 runs)'] = (
            statistics.median(negs) == 5 and not rep_neg)
        print('new10 repeated(>=2):', rep_new or 'none', '| negation repeated(>=2):', rep_neg or 'none')
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

    # 12 family (v11 gate; v12 regression guard)
    fams = [family_diag(t, pre) for t in tags]
    if any(fm is None for fm, _ in fams):
        med_fam = None
        axes['12 family(median mean-p<0.5 AND all runs <=1 case p>=0.9)'] = False
        print('family: MISSING artifact for', [t for t, (fm, _) in zip(tags, fams) if fm is None],
              '-> axis FAIL (v11+ runs must emit family_diag_<tag>.json)')
    else:
        med_fam = statistics.median(fm for fm, _ in fams)
        axes['12 family(median mean-p<0.5 AND all runs <=1 case p>=0.9)'] = (
            med_fam < 0.5 and all(nh <= 1 for _, nh in fams))
        print('family mean p:', [(t, fm) for t, (fm, _) in zip(tags, fams)], '| median:', round(med_fam, 4),
              '| p>=0.9 cases per run:', dict(zip(tags, [nh for _, nh in fams])))

    if proto in ('v12', 'v13'):
        # 13 negfam (v12+ instrument, 8 cases gold=true)
        nf = [negfam_diag(t, pre) for t in tags]
        if any(fm is None for fm, _ in nf):
            axes['13 negfam(median mean-p>=0.5 AND all runs <=1 case p<0.5)'] = False
            print('negfam: MISSING artifact for', [t for t, (fm, _) in zip(tags, nf) if fm is None],
                  '-> axis FAIL (v12+ runs must emit negfam_diag_<tag>.json)')
        else:
            med_nf = statistics.median(fm for fm, _ in nf)
            axes['13 negfam(median mean-p>=0.5 AND all runs <=1 case p<0.5)'] = (
                med_nf >= 0.5 and all(nl <= 1 for _, nl in nf))
            print('negfam mean p:', [(t, fm) for t, (fm, _) in zip(tags, nf)], '| median:', round(med_nf, 4),
                  '| p<0.5 cases per run:', dict(zip(tags, [nl for _, nl in nf])))
        hyg_name = f'leak_audit_{proto}.txt'
        hyg_axis = '14 hygiene(0/35 + family 9+8=17 cases 0 overlap asserted at gen)'
    else:
        hyg_name, hyg_axis = 'leak_audit_v11.txt', '13 hygiene(0/35 + family cases 0 overlap asserted at gen)'

    # hygiene (last numbered axis of the protocol)
    hyg = os.path.join(_D, hyg_name)
    ok_hyg = os.path.exists(hyg) and 'cases touched: 0/35' in open(hyg, encoding='utf-8').read()
    axes[hyg_axis] = ok_hyg
    print(f'{hyg_name}:', 'present, 0/35' if ok_hyg else 'MISSING or nonzero')

    # report-only
    print('tau(noul) report-only:', {t: tau_report(t) for t in tags})
    if proto in ('v12', 'v13'):
        print('automation@5% report-only (v11 baseline 0.743/0.786/0.793):',
              {t: round(automation5(t, pre), 3) for t in tags})

    print()
    verdict = all(axes.values())
    for k, v in axes.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f'\nPROTOCOL VERDICT: {"PASS -> eligible for HF delivery (§1.3)" if verdict else "FAIL -> no delivery, HF untouched"}')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
