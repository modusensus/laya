# -*- coding: utf-8 -*-
"""Conformal rule-form pre-registered evaluation (HANDOFF_NLI_V7.md §6).

Goal: does ANY row-level uncertainty signal support an abstention rule that
captures >=50% of errors on the report half at <=35% abstention?  The v6
finding (maxconf/LAC cannot catch high-confidence errors: 52/97 errors at
conf>=0.92) motivates two non-maxconf candidates.

Pre-registered protocol (frozen before evaluation):
  * Candidates (exactly three, no post-hoc fourth):
      s1 = maxconf(p1)                     -- status-quo LAC baseline
      s2 = 1 - (max_r p_r - min_r p_r)     -- surface consistency over three
              fixed renderings of the SAME state: r1 row-original questions,
              r2 neutral A/B, r3 keep/supersede (v4 LABEL_SETS[3])
      s3 = answers.conflict.action.act_probability  -- auxiliary action-head
              slot-0 probability, read off the r1 forward (field semantics
              verified in laya.agent._decode_answers: `act` = the auxiliary
              action head output; direction NOT assumed, selected on half1)
  * Split: seed 20260928 shuffle, half1 = first 500 (fit), half2 = last 500
    (report).  Halves asserted disjoint.  Identical construction to
    conformal_abstain.py --split-half (comparable with v6 numbers).
  * On half1 only: scan BOTH rule directions per candidate (write iff
    score >= T, write iff score <= T) over all observed score values; select
    ONE variant per candidate = max capture subject to abstain <= 0.35
    (tie-break: lower abstain).  This selection is the entire fitting budget.
  * On half2: evaluate the selected variant.  Gate: capture >= 50% of half2
    errors AND abstain <= 35%.  Any candidate passing => adopt this round
    (wire into conformal_abstain.py as a mode); otherwise keep the status quo
    and report the negative conclusion.
  * half2 and the 35 realtest cases never enter any fit or selection.

Usage: python conformal_alt_probe.py <ckpt> <out_json> [--rows N] [--skip-preds]
       --skip-preds loads scores from the output file (for re-analysis).
"""
import argparse
import json
import sys

import numpy as np

_D = r'D:\laya\data_local'
_ap = argparse.ArgumentParser()
_ap.add_argument('ckpt')
_ap.add_argument('out')
_ap.add_argument('--main', default=_D + r'\nli_conflict_val.jsonl')
_ap.add_argument('--val-probs', default=_D + r'\val_probs_v7.json',
                 help='kernel val_probs for the p1 cross-check')
_ap.add_argument('--rows', type=int, default=0, help='smoke-test cap (0 = all 1000)')
_ap.add_argument('--skip-preds', action='store_true', help='re-analyze existing output')
_ap.add_argument('--sel-budget', type=float, default=0.35,
                 help='half1 SELECTION budget (V9 §6.2: pass 0.34 for a 1pp margin; '
                      'the half2 GATE stays 0.35 -- pre-registered, not a gate change)')
args = _ap.parse_args()

R2 = {'false': 'A', 'true': 'B'}
R3 = {'false': 'keep', 'true': 'supersede'}
R3_INS = ('Does the new information (new) conflict with the known (old) memory? '
          'supersede = the new information contradicts the old memory and must replace it, '
          'keep = consistent or unrelated')
R2_INS = '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关'


def rendering_questions(row, rid):
    if rid == 1:
        return row['questions']
    if rid == 2:
        return {'conflict': {'type': 'noul', 'instructions': R2_INS, 'labels': dict(R2)}}
    if rid == 3:
        return {'conflict': {'type': 'noul', 'instructions': R3_INS, 'labels': dict(R3)}}
    raise ValueError(rid)


def run_preds(ckpt, rows, cap):
    import laya
    agent = laya.load(ckpt, device='cpu')
    out = []
    n = len(rows) if not cap else min(cap, len(rows))
    for i, row in enumerate(rows[:n]):
        st = json.loads(row['state']) if isinstance(row['state'], str) else row['state']
        ps, act_p = [], None
        for rid in (1, 2, 3):
            a = agent.predict(st, rendering_questions(row, rid))['answers']['conflict']
            ps.append(float(a['noul']))
            if rid == 1:
                act_p = float(a['action']['act_probability'])
        gold = 1 if row['gold']['conflict']['label'] == 'true' else 0
        out.append({'idx': i, 'gold': gold, 'p1': ps[0], 'p2': ps[1], 'p3': ps[2], 's3': act_p,
                    's2': 1 - (max(ps) - min(ps))})
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{n}', flush=True)
    return out


def scan(h1, score, err, max_abstain=0.35):
    """Scan both directions on the fitting half; returns full curves."""
    s = np.asarray(score, float)
    curves = {}
    for direction in ('ge', 'le'):
        variants = []
        for t in np.unique(s):
            dec = (s >= t) if direction == 'ge' else (s <= t)
            abstain = float((~dec).mean())
            capture = int((err & ~dec).sum())
            variants.append({'T': round(float(t), 4), 'abstain': round(abstain, 4),
                             'capture': capture, 'n_err': int(err.sum())})
        # select: max capture s.t. abstain <= max_abstain; tie-break lower abstain
        ok = [v for v in variants if v['abstain'] <= max_abstain]
        best = min(ok, key=lambda v: (-v['capture'], v['abstain'])) if ok else None
        curves[direction] = {'variants': variants, 'selected': best}
    return curves


def eval_rule(h2, score, err, direction, t):
    s = np.asarray(score, float)
    dec = (s >= t) if direction == 'ge' else (s <= t)
    n_err = int(err.sum())
    capture = int((err & ~dec).sum())
    abstain = float((~dec).mean())
    wer = float((dec & err).sum() / dec.sum()) if dec.sum() else None
    gate = capture >= 0.5 * n_err and abstain <= 0.35
    return {'direction': direction, 'T': round(float(t), 4), 'n': len(s), 'n_err': n_err,
            'capture': capture, 'capture_rate': round(capture / n_err, 4) if n_err else None,
            'abstain': round(abstain, 4), 'write_error_rate': round(wer, 4) if wer is not None else None,
            'gate': '>=50% capture @ <=35% abstain', 'pass': bool(gate)}


def main():
    rows = [json.loads(l) for l in open(args.main, encoding='utf-8') if l.strip()]
    assert len(rows) == 1000, len(rows)

    if args.skip_preds:
        blob = json.load(open(args.out, encoding='utf-8'))
        preds = blob['rows']
    else:
        preds = run_preds(args.ckpt, rows, args.rows)
        json.dump({'rows': preds}, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False)

    # p1 cross-check against the kernel's tau-corrected dump (sanity only)
    try:
        kp = json.load(open(args.val_probs, encoding='utf-8'))
        if len(kp) == len(preds) == 1000:
            dev = max(abs(r['p1'] - k['p_true']) for r, k in zip(preds, kp))
        else:
            dev = None
    except FileNotFoundError:
        dev = None

    n = len(preds)
    idx = list(range(n))
    import random as _random
    _random.Random(20260928).shuffle(idx)
    h1, h2 = idx[:n // 2], idx[n // 2:]
    assert set(h1).isdisjoint(set(h2))

    gold = np.array([preds[i]['gold'] for i in range(n)])
    p1 = np.array([preds[i]['p1'] for i in range(n)])
    p2 = np.array([preds[i]['p2'] for i in range(n)])
    p3 = np.array([preds[i]['p3'] for i in range(n)])
    s3 = np.array([preds[i]['s3'] for i in range(n)])
    err = ((p1 >= 0.5).astype(int) != gold)
    conf = np.maximum(p1, 1 - p1)
    spread = np.maximum(np.maximum(p1, p2), p3) - np.minimum(np.minimum(p1, p2), p3)
    s2 = 1 - spread

    scores = {'s1_maxconf': conf, 's2_surface_consistency': s2, 's3_act_probability': s3}
    cand = {}
    verdict = None
    adopted = []
    for name, s in scores.items():
        c1 = scan(h1, s[h1], err[h1], max_abstain=args.sel_budget)
        sel = None
        for direction in ('ge', 'le'):
            b = c1[direction]['selected']
            if b and (sel is None or (-b['capture'], b['abstain']) < (-sel['capture'], sel['abstain'])):
                sel = dict(b, direction=direction)
        r2 = eval_rule(h2, s[h2], err[h2], sel['direction'], sel['T']) if sel else None
        cand[name] = {'half1_curves': c1, 'half1_selected': sel, 'half2_report': r2}
        if r2 and r2['pass']:
            adopted.append(name)
    verdict = ('ADOPT ' + adopted[0] if len(adopted) == 1 else
               'ADOPT (tie: ' + ','.join(adopted) + ')' if adopted else
               'KEEP STATUS QUO (no candidate reaches >=50% capture @ <=35% abstain on half2)')

    stats = {}
    for name, s in scores.items():
        stats[name] = {'half1_err_mean_score': round(float(s[h1][err[h1]].mean()), 4),
                       'half1_ok_mean_score': round(float(s[h1][~err[h1]].mean()), 4),
                       'half2_err_mean_score': round(float(s[h2][err[h2]].mean()), 4),
                       'half2_ok_mean_score': round(float(s[h2][~err[h2]].mean()), 4)}

    out = {'protocol': {'split_seed': 20260928, 'n1': len(h1), 'n2': len(h2),
                        'candidates': list(scores), 'selection': f'half1 max capture @ abstain<={args.sel_budget:.0%} '
                                  f'(V9 §6.2 sel budget; half2 gate still 0.35), both directions scanned',
                        'gate': 'half2 >=50% capture @ <=35% abstain',
                        'p1_kernel_crosscheck_maxdev': round(dev, 4) if dev is not None else None},
           'error_stats': stats,
           'main_val_err': int(err.sum()),
           'candidates': cand,
           'verdict': verdict,
           'rows': preds}  # per-row scores for all three candidates (V7 §6 deliverable)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('main val err:', int(err.sum()), '| split errs h1/h2:', int(err[h1].sum()), '/', int(err[h2].sum()))
    for name, c in cand.items():
        r2 = c['half2_report']
        if r2:
            print(f"{name}: half1 sel {c['half1_selected']['direction']} T={c['half1_selected']['T']} "
                  f"(cap {c['half1_selected']['capture']} @ {c['half1_selected']['abstain']:.0%}) -> "
                  f"half2 cap {r2['capture']}/{r2['n_err']} @ {r2['abstain']:.1%} => "
                  f"{'PASS' if r2['pass'] else 'FAIL'}")
    print('VERDICT:', verdict)
    print('saved:', args.out)
    sys.exit(0 if adopted else 1)


if __name__ == '__main__':
    main()
