# -*- coding: utf-8 -*-
"""Split-conformal abstention + combo-rule audit (HANDOFF_NLI_V4.1.md, 改动3).

LAC split-conformal on the noul head: nonconformity s = 1 - p(gold class),
quantile fitted on val_soft-300 ONLY (never on main val — that is snooping).
Decision rule: write/decide iff max_conf >= T (T = 1 - qhat). Abstained items
go to human/stronger-model review. Red line unchanged: plugin auto-write
threshold stays <= 0.92; conformal is an ADDITIONAL layer
(write = p_true at red line AND not abstained).

Gates (main val, 99 errors, per V4.1 after precompute):
  capture >= 50/99 (50.5%) AND abstention <= 35%
Report-only: val_soft coverage (same-distribution, nominal guarantee applies),
realtest/diag coverage+abstention, 20%-budget envelope (proven unreachable:
max 48-49 errors at 20% budget, recompute note in V4.1), nominal 95/85/80
variants. Exchangeability caveat: val_soft (synthetic) fit -> main val (MNLI)
evaluation is distribution shift; main-val numbers are transfer performance,
not a finite-sample guarantee.

Usage: python conformal_abstain.py <out_json>
"""
import argparse
import json
import math
import sys

import numpy as np

_D = r'D:\laya\data_local'
_ap = argparse.ArgumentParser()
_ap.add_argument('--soft', default=_D + r'\val_soft_v4.json')
_ap.add_argument('--main', default=_D + r'\val_probs_v4.json')
_ap.add_argument('--valj', default=_D + r'\nli_conflict_val.jsonl')
_ap.add_argument('--real', default=_D + r'\memory_conflict_realtest_v4.json')
_ap.add_argument('--diag', default=_D + r'\noul_bias_diag_v4.json')
_ap.add_argument('--out', default=_D + r'\conformal_v4.json')
_ap.add_argument('--capture-gate', type=int, default=50, help='min errors to capture (v5: >=50 of that round errors)')
_ap.add_argument('--abstain-gate', type=float, default=0.35)
args = _ap.parse_args()
SOFT, MAIN, VALJ, REAL, DIAG, OUT = args.soft, args.main, args.valj, args.real, args.diag, args.out


def fit_quantile(p, g, nominal):
    """LAC: s = 1 - p(gold); finite-sample quantile at ceil((n+1)*nominal)."""
    s = np.where(g == 1, 1 - p, p)
    k = int(math.ceil(nominal * (len(s) + 1)))
    qhat = float(np.sort(s)[k - 1])
    return qhat, 1 - qhat, k


def collect_p_conflict(obj, out):
    if isinstance(obj, dict):
        for key in ('p_conflict', 'p_true'):
            v = obj.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
                return
        for v in obj.values():
            collect_p_conflict(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_p_conflict(v, out)


def audit(p, ok, t):
    mc = np.maximum(p, 1 - p)
    dec = mc >= t
    n = len(p)
    wr = float(dec.mean())
    wer = float((dec & ~ok).sum() / dec.sum()) if dec.sum() else 0.0
    return {'threshold': t, 'write_rate': round(wr, 4),
            'write_error_rate': round(wer, 4) if dec.sum() else None,
            'abstain_rate': round(1 - wr, 4), 'n_write': int(dec.sum()),
            'n_write_err': int((dec & ~ok).sum())}


def main():
    global args_capture
    args_capture = (args.capture_gate, args.abstain_gate)
    # ---- fit (val_soft only) ----
    soft = json.load(open(SOFT, encoding='utf-8'))
    sp = np.array([r['p_true'] for r in soft], float)
    sg = np.array([1 if r['label'] in (True, 'true', 1) else 0 for r in soft], int)
    qhat, T, k = fit_quantile(sp, sg, 0.90)
    print(f'fit: n=300, nominal 90% -> k={k}, qhat={qhat:.4f}, T={1 - qhat:.4f}')

    variants = {}
    for nom in (0.95, 0.85, 0.80):
        variants[f'{int(nom * 100)}%'] = fit_quantile(sp, sg, nom)

    # ---- main val ----
    probs = json.load(open(MAIN, encoding='utf-8'))
    p = np.array([r['p_true'] for r in probs], float)
    g = np.array([int(r['gold']) for r in probs], int)
    pred = (p >= 0.5).astype(int)
    err = pred != g
    mc = np.maximum(p, 1 - p)
    dec = mc >= T
    capture = int((err & ~dec).sum())
    abst = float((~dec).mean())
    gate_capture = capture >= args_capture[0]
    gate_abst = abst <= args_capture[1]

    # risk-coverage / AURC on main val (ordering = max_conf desc)
    order = np.argsort(-mc, kind='stable')
    risks = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    aurc = float(risks.mean())
    err_count = int(err.sum())
    # oracle ordering: all errors least confident
    oracle_err = np.concatenate([np.zeros(len(err) - err_count, int), np.ones(err_count, int)])
    aurc_oracle = float((np.cumsum(oracle_err) / np.arange(1, len(err) + 1)).mean())

    # ---- other sets ----
    sdec = np.maximum(sp, 1 - sp) >= T
    soft_cov = float((sdec & ((sp >= 0.5).astype(int) == sg)).mean())
    real = json.load(open(REAL, encoding='utf-8'))
    rp, rok = [], []
    for grp in ('cases', 'negation_cases', 'new_cases'):
        for c in real[grp]:
            rp.append(float(c['p_conflict']))
            rok.append(bool(c['ok']))
    rp, rok = np.array(rp), np.array(rok)
    rdec = np.maximum(rp, 1 - rp) >= T
    dp = []
    collect_p_conflict(json.load(open(DIAG, encoding='utf-8')), dp)
    dp = np.array(dp)

    # ---- per-item list: abstained errors on main val ----
    rows = [json.loads(l) for l in open(VALJ, encoding='utf-8')]
    assert len(rows) == len(g), 'val jsonl / val_probs length mismatch'
    golds = [1 if r['gold']['conflict']['label'] == 'true' else 0 for r in rows]
    assert golds == list(g), 'val jsonl order does not match val_probs golds'
    abstained_err = []
    for i in np.where(err & ~dec)[0]:
        st = json.loads(rows[i]['state'])
        abstained_err.append({'idx': int(i), 'known': st.get('known', ''), 'new': st.get('new', ''),
                              'p_true': float(p[i]), 'gold': 'true' if g[i] == 1 else 'false'})

    # ---- combo-rule audit table ----
    audit_main = {f'{t:.4f}': audit(p, ~err, t) for t in (0.92, T, 0.9225, 0.90)}
    audit_soft = {f'{t:.4f}': audit(sp, (sp >= 0.5).astype(int) == sg, t) for t in (0.92, T, 0.9225, 0.90)}

    # ---- nominal variants migrated to main val ----    var_rows = {}
    var_rows = {}
    for name, (qh2, T2, k2) in variants.items():
        d2 = mc >= T2
        var_rows[name] = {'T': round(T2, 4), 'qhat': round(qh2, 4),
                          'main_abstain': round(float((~d2).mean()), 4),
                          'main_capture': int((err & ~d2).sum())}

    out = {
        'input': {'soft': SOFT, 'main': MAIN, 'real': REAL},
        'fit': {'set': 'val_soft 300', 'nominal': 0.90, 'k_order_stat': k,
                'qhat': round(qhat, 4), 'threshold_T': round(T, 4)},
        'gates': {'capture': capture, 'capture_total': int(err.sum()), 'abstain': round(abst, 4),
                  'gate': f'>={args_capture[0]}/errors and <={args_capture[1]:.0%} abstain',
                  'pass': bool(gate_capture and gate_abst)},
        'main_val': {'decide': int(dec.sum()), 'abstain': round(abst, 4), 'capture': capture,
                     'write_error_rate': round(float((dec & err).sum() / dec.sum()), 4),
                     'aurc_maxconf_order': round(aurc, 5), 'aurc_oracle': round(aurc_oracle, 5)},
        'val_soft': {'decide': int(sdec.sum()), 'coverage': round(soft_cov, 4),
                     'abstain': round(float((~sdec).mean()), 4),
                     'note': 'same-distribution slice; nominal 90% guarantee applies here'},
        'realtest_35': {'decide': int(rdec.sum()), 'coverage_decide_and_ok': round(float((rdec & rok).mean()), 4),
                        'abstain': round(float((~rdec).mean()), 4)},
        'diag_14': {'n_p_found': int(len(dp)),
                    'decide': int((np.maximum(dp, 1 - dp) >= T).sum()) if len(dp) else None},
        'budget_20pct': {'envelope_capture': '48-49/99 (recompute note in V4.1); 50 unreachable at <=20%',
                         'verdict': 'unreachable, report-only per V4.1'},
        'variants_migrated': var_rows,
        'audit_table_main_val': audit_main,
        'audit_table_val_soft': audit_soft,
        'tau_warning': 'confidence band is a function of tau: sharpening tau 1.2176->0.7-0.8 lifts '
                       'main-val write rate 87.1%->98.5%+ and write error rate 6.66%->9.4-9.5%; '
                       'any tau change must pass the audit table first (V4.1 预核算 §5)',
        'exchangeability_note': 'val_soft(synthetic) fit -> main val(MNLI) = distribution shift; '
                                'main-val numbers are TRANSFER PERFORMANCE, not a finite-sample '
                                'guarantee; val_soft coverage 0.907 is the same-distribution check',
        'abstained_errors': abstained_err,
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"GATE: capture {capture}/{int(err.sum())} (>={args_capture[0]} {'PASS' if gate_capture else 'FAIL'}) @ "
          f"abstain {abst:.1%} (<={args_capture[1]:.0%} {'PASS' if gate_abst else 'FAIL'}) => "
          f"{'PASS' if out['gates']['pass'] else 'FAIL'}")
    print(f"val_soft coverage {soft_cov:.4f} (nominal 90%) | realtest decide {int(rdec.sum())}/35 | "
          f"AURC {aurc:.5f} vs oracle {aurc_oracle:.5f}")
    print('abstained errors listed:', len(abstained_err))
    print('saved:', OUT)
    sys.exit(0 if out['gates']['pass'] else 1)


if __name__ == '__main__':
    main()
