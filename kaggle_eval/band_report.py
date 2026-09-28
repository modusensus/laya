# -*- coding: utf-8 -*-
"""Confidence-separation band report (V5 §5.2 / V6 §5.2 deliverable conf_band_*.txt).

Quantiles use the nearest-rank convention `sorted(xs)[round(p*(n-1))]` (V5 §2.5),
conf = max(p_true, 1 - p_true), decision true iff p_true >= 0.5.

Inputs:
  --main  val_probs_*.json          (1000 x {gold, p_true}, tau-corrected)
  --soft  val_soft_probs_*.json     (300 x {pred, p_true, label, meta[row_id]})
  --meta  nli_conflict_val_soft.meta.json  (sidecar: per-row {cat, kind, lang, ...};
           joined via meta.row_id so the consid median is reportable — the v5 gap)
  --ref-tag / --ref-main / --ref-soft   reference numbers for the对照 line

Usage: python band_report.py --main ... --soft ... --out conf_band_v6.txt
"""
import argparse
import json

_D = r'D:\laya\data_local'
_ap = argparse.ArgumentParser()
_ap.add_argument('--main', default=_D + r'\val_probs_v6.json')
_ap.add_argument('--soft', default=_D + r'\val_soft_probs_v6.json')
_ap.add_argument('--meta', default=_D + r'\nli_conflict_val_soft.meta.json')
_ap.add_argument('--out', default=_D + r'\conf_band_v6.txt')
_ap.add_argument('--round-tag', default='v6')
_ap.add_argument('--ref-main', default='v4 0.62pp / v5-A 1.62pp / v5-B 3.96pp')
_ap.add_argument('--ref-soft', default='v4 0.28pp / v5-A 16.29pp / v5-B 16.18pp')
_ap.add_argument('--ref-consid', default='v4 0.9239')
args = _ap.parse_args()

BINS = [(0.50, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.92), (0.92, 0.96), (0.96, 1.01)]


def quant(xs, p):
    s = sorted(xs)
    return s[round(p * (len(s) - 1))]


def load_main(path):
    rows = json.load(open(path, encoding='utf-8'))
    conf = [max(r['p_true'], 1 - r['p_true']) for r in rows]
    err = [0 if (r['p_true'] >= 0.5) == (r['gold'] == 1) else 1 for r in rows]
    return conf, err


def load_soft(path, meta_path):
    rows = json.load(open(path, encoding='utf-8'))
    side = json.load(open(meta_path, encoding='utf-8'))
    conf, err, kinds = [], [], []
    for r in rows:
        conf.append(max(r['p_true'], 1 - r['p_true']))
        err.append(0 if r['pred'] == r['label'] else 1)
        rid = r.get('meta', {}).get('row_id')
        kinds.append(side[rid]['kind'] if rid is not None and rid < len(side) else None)
    return conf, err, kinds


def section(name, conf, err, gate_line=''):
    n = len(conf)
    q10, q50, q90, q99, mx = (quant(conf, p) for p in (0.10, 0.50, 0.90, 0.99, 0.999999))
    band = q90 - q10
    lines = [f'{name}:n={n} err={sum(err)} acc={1 - sum(err) / n:.4f} mean_conf={sum(conf) / n:.4f}',
             f'  q10 {q10:.4f} / q50 {q50:.4f} / q90 {q90:.4f} / q99 {q99:.4f} / max {mx:.4f}'
             f'  band(q90-q10) = {band * 100:.2f}pp {gate_line}']
    cells = []
    for lo, hi in BINS:
        idx = [i for i, c in enumerate(conf) if lo <= c < hi]
        e = sum(err[i] for i in idx)
        cells.append(f'[{lo:.2f},{hi:.2f}) n={len(idx)}' + (f' err={e} ({e / len(idx):.3f})' if idx else ''))
    lines.append('  分箱: ' + ' | '.join(cells))
    return lines, band


def main():
    mconf, merr = load_main(args.main)
    sconf, serr, skinds = load_soft(args.soft, args.meta)
    m_lines, m_band = section('主 val', mconf, merr)
    s_lines, s_band = section('val_soft', sconf, serr)
    # annotate the val_soft header line with the round-over-round gate verdict
    s_lines[0] += f'  (v4/v5 对照 {args.ref_soft};门槛 ≥8pp ⇒ {"PASS" if s_band * 100 >= 8 else "FAIL"})'

    consid = [c for c, k in zip(sconf, skinds) if k == 'consid']
    consid_txt = f'consid(n={len(consid)})中位置信 = {quant(consid, 0.5):.4f}' if consid else 'consid: 无匹配行'
    lines = [f'conf_band_{args.round_tag} — 分离度全套(§5.2;复算口径 = 最接近序号法, conf=max(p,1-p))',
             '', f'== 主臂({args.round_tag}) =='] + m_lines + s_lines + ['',
             f'consid 中位置信:{consid_txt}(对照 {args.ref_consid};由 dump row_id ↔ val_soft.meta.json 边车回查)',
             '', f'主 val band 对照:{args.ref_main}', '']
    open(args.out, 'w', encoding='utf-8').write('\n'.join(lines))
    print('\n'.join(lines))
    print(f"\nval_soft band gate (>=8pp): {'PASS' if s_band * 100 >= 8 else 'FAIL'} -> {args.out}")


if __name__ == '__main__':
    main()
