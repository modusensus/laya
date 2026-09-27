# -*- coding: utf-8 -*-
"""Per-class vector-scaling pre-check (HANDOFF_NLI_V4.1.md, 改动2) — OFFLINE.

Runtime reality (V4.1 B1): temperature granularity = question-type x option-count
bucket (noul is always `noul:2`, no per-class axis); decode path is
`z = logits[r,:k] / t_scale` (agent.py:768) with no bias term; temperature is
clamped to [0.5, 5.0]. Per-class (tau_c, b_c) therefore requires a code change;
this round is pre-check + offline recalibration ONLY — no runtime change, no
served config. "argmax invariant" holds for a single tau only.

Pre-check criterion (upgraded per review B3): build the class-x-set matrix of
mean_conf - acc; a class that needs OPPOSITE fixes on the two sets => FAIL =>
negative-conclusion report (a negative conclusion is a valid deliverable).
Both sets must contain errors for a class to be judgeable.

If the pre-check unexpectedly PASSES, fitting requires raw per-row logits
([l_false, l_true]) which p_true alone cannot recover — the logits-dump +
(τ_t,b_t,τ_f,b_f) fit on val_soft-300 is implemented on demand (v5 if ever).

Usage: python vector_scaling.py <out_txt>
"""
import json
import sys

import numpy as np

MAIN = r'D:\laya\data_local\val_probs_v4.json'          # 1000 x {gold, p_true}, tau=1.2176
SOFT = r'D:\laya\data_local\val_soft_v4.json'           # 300 x {pred, p_true, label, meta}
OUT = sys.argv[1] if len(sys.argv) > 1 else r'D:\laya\data_local\vector_scaling_v4.txt'


def load_main():
    rows = json.load(open(MAIN, encoding='utf-8'))
    p = np.array([r['p_true'] for r in rows], float)
    g = np.array([int(r['gold']) for r in rows], int)
    return p, g, '主 val 1000 (MNLI 系, tau=1.2176)'


def load_soft():
    rows = json.load(open(SOFT, encoding='utf-8'))
    p = np.array([r['p_true'] for r in rows], float)
    g = np.array([1 if r['label'] in (True, 'true', 1) else 0 for r in rows], int)
    return p, g, 'val_soft 300 (合成软冲突)'


def reliability(p, g, cls, edges):
    """(n, acc, mean_conf) per max_conf bin within one gold class."""
    m = g == cls
    p, mc = p[m], np.maximum(p[m], 1 - p[m])
    correct = ((p >= 0.5).astype(int) == cls).astype(float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (mc >= lo) & (mc < hi) if hi < edges[-1] else (mc >= lo) & (mc <= hi)
        if b.sum():
            out.append((lo, hi, int(b.sum()), float(correct[b].mean()), float(mc[b].mean())))
    return out


def main():
    sets = [load_main(), load_soft()]
    lines = []
    log = lambda s='': (lines.append(s), print(s))  # noqa: E731

    log('per-class vector scaling — pre-check (HANDOFF_NLI_V4.1 改动2)')
    log('criterion: any class needing OPPOSITE fixes across sets => FAIL (negative conclusion)')
    log('offline only; runtime has no per-class axis / bias term / tau not clamped here; nothing served')
    log()
    matrix = {}
    for p, g, name in sets:
        err = (p >= 0.5).astype(int) != g
        mc = np.maximum(p, 1 - p)
        for cls, cname in [(1, 'true'), (0, 'false')]:
            m = g == cls
            acc = 1 - err[m].mean()
            gap = float(mc[m].mean() - acc)
            direction = 'soften(τ>1)' if gap > 0 else 'sharpen(τ<1)'
            matrix[(cname, name)] = gap
            log(f'{name} | {cname:5s} class: n={m.sum():4d} err={err[m].sum():3d} '
                f'acc={acc:.4f} mean_conf={mc[m].mean():.4f} conf-acc={gap:+.4f} -> {direction}')
        log()

    log('class x set direction matrix (judgeable only if both sets have errors in the class):')
    fail = False
    for cname in ['true', 'false']:
        g1 = matrix[(cname, sets[0][2])]
        g2 = matrix[(cname, sets[1][2])]
        judgeable = True  # errors per class printed above; val_soft true=1/false=2, main 57/42
        opposite = (g1 > 0) != (g2 > 0)
        if opposite:
            fail = True
        log(f'  {cname:5s}: main {g1:+.4f} vs soft {g2:+.4f} -> '
            f'{"OPPOSITE => FAIL" if opposite else "same direction"}{"" if judgeable else " (NOT judgeable)"}')
    log()
    if fail:
        log('PRE-CHECK: FAIL — negative conclusion. Per-class scaling cannot fix a class that')
        log('needs softening on one set and sharpening on the other (and the bias term (b_t-b_f)')
        log('is a prior shift, which cannot split a within-class cross-set conflict either).')
        log('Root cause is the confidence compression band (~0.92 for nearly everything), see')
        log('V4.1 预核算 §2 — fix belongs to training (soft CE / R-Drop / temperature-in-train), v5.')
    else:
        log('PRE-CHECK: PASS (unexpected) — fitting needs raw logits dump per row')
        log('([l_false, l_true]; not recoverable from p_true), then (τ_t,b_t,τ_f,b_f) fit on')
        log('val_soft-300, NLL/confidence objective. Implement on demand (v5 scope).')

    log()
    log('reliability tables (max_conf bins; classes split by gold):')
    edges = np.array([0.0, 0.5, 0.8, 0.88, 0.90, 0.905, 0.910, 0.915, 0.918, 0.920,
                      0.921, 0.922, 0.923, 0.924, 0.925, 0.926, 0.927, 1.0])
    for p, g, name in sets:
        for cls, cname in [(1, 'true'), (0, 'false')]:
            log(f'  {name} | {cname} class')
            log('    bin              n    acc   mean_conf')
            for lo, hi, n, acc, mconf in reliability(p, g, cls, edges):
                log(f'    [{lo:.3f},{hi:.3f}) {n:5d} {acc:5.3f} {mconf:9.4f}')
            log()

    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('saved:', OUT)
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
