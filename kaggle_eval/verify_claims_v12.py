# -*- coding: utf-8 -*-
"""HANDOFF_NLI_V12.md §1.4 delivery gate: re-derive every claim in
data_local/claims_v12.json from its source artifact and compare.

Exit 0 iff EVERY claim verifies — hf_publish is forbidden otherwise (§1.3).
Run: python kaggle_eval/verify_claims_v12.py
"""
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data_local')
OUT = r'D:\laya-kaggle-output'
CLAIMS = json.load(open(os.path.join(D, 'claims_v12.json'), encoding='utf-8'))


def load(p):
    return json.load(open(p, encoding='utf-8'))


def main_val(tag):
    rows = load(os.path.join(D, f'val_probs_v12_{tag}.json'))
    err = sum(1 for r in rows if (r['p_true'] >= 0.5) != (r['gold'] == 1))
    return round(1 - err / len(rows), 3), err


def realtest(tag):
    d = load(os.path.join(D, f'memory_conflict_realtest_v12_{tag}.json'))
    return (sum(r['ok'] for r in d['cases']),
            sum(r['ok'] for r in d['negation_cases']),
            sum(r['ok'] for r in d['new_cases']))


def val_soft(tag):
    rows = load(os.path.join(D, f'val_soft_probs_v12_{tag}.json'))
    return sum(1 for r in rows if ('true' if r['p_true'] >= 0.5 else 'false') != r['label'])


def conformal(tag, rule):
    g = load(os.path.join(D, f'conformal_v12_{tag}_{rule}.json'))['gates']
    tot = g.get('capture_total')
    cap = g.get('capture')
    n = f'{cap}/{tot}' if tot else f'{cap}'
    return f"{n} @ {g['abstain']:.3f} {'PASS' if g['pass'] else 'FAIL'}"


def diag(tag):
    d = load(os.path.join(D, f'noul_bias_diag_v12_{tag}.json'))
    pairs = d['controls'] + d['swaps']
    return f"{sum(r['ok'] for r in pairs)}/{len(pairs)}"


def tau(tag):
    c = load(os.path.join(OUT, f'laya-nli-conflict-v12-{tag}', 'rl_agent_config.json'))
    return c['temperature'][2]


def automation5(tag):
    rows = load(os.path.join(D, f'val_probs_v12_{tag}.json'))
    conf = sorted(((max(r['p_true'], 1 - r['p_true']), (r['p_true'] >= 0.5) == (r['gold'] == 1))
                   for r in rows), reverse=True)
    best, err = 0, 0
    for k, (_c, ok) in enumerate(conf, 1):
        if not ok:
            err += 1
        if err / k <= 0.05:
            best = k
    return round(best / len(rows), 3)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def band(tag):
    txt = open(os.path.join(D, f'conf_band_v12_{tag}.txt'), encoding='utf-8').read()
    for line in txt.splitlines():
        if line.startswith('val_soft:') and '≥8pp' in line:
            return 'PASS' if '⇒ PASS' in line else 'FAIL'
    if 'val_soft band gate (>=8pp): PASS' in txt:
        return 'PASS'
    return 'UNKNOWN'


def main():
    by_id = {c['id']: c for c in CLAIMS['claims']}
    derived = {}
    for t in ('r1', 'r2', 'r3'):
        old, neg, new = realtest(t)
        derived[f'main_val_{t}'], _ = main_val(t)
        derived[f'old20_{t}'], derived[f'negation5_{t}'], derived[f'new10_{t}'] = old, neg, new
        derived[f'val_soft_err_{t}'] = val_soft(t)
        derived[f'swap_{t}'] = load(os.path.join(D, f'label_swap_v12_{t}.json')).get('verdict',
                                    load(os.path.join(D, f'label_swap_v12_{t}.json')).get('pass'))
        pol = load(os.path.join(D, f'polarity_nli_v12_{t}.json'))['gates']
        derived[f'polarity_real_diag_{t}'] = bool(pol['real_diag_diff_le_1'])
        derived[f'polarity_val_soft_{t}'] = bool(pol['val_soft_err_change_le_1'])
        derived[f'band_{t}'] = band(t)
        derived[f'conformal_{t}_s1'] = conformal(t, 's1')
        derived[f'conformal_{t}_s2'] = conformal(t, 's2')
        derived[f'diag_{t}'] = diag(t)
        fd = load(os.path.join(D, f'family_diag_v12_{t}.json'))
        derived[f'family_mean_{t}'] = fd['family_mean_p']
        derived[f'nf{t}'] = fd['n_highconf']
        nd = load(os.path.join(D, f'negfam_diag_v12_{t}.json'))
        derived[f'negfam_mean_{t}'] = nd['family_mean_p']
        derived[f'nl{t}'] = nd['n_lowconf']
        derived[f'tau_{t}'] = tau(t)
        derived[f'automation5_{t}'] = automation5(t)
    derived['main_val_median'] = statistics.median(
        derived[f'main_val_{t}'] for t in ('r1', 'r2', 'r3'))
    derived['family_highconf_all'] = '/'.join(str(derived[f'nf{t}']) for t in ('r1', 'r2', 'r3'))
    derived['negfam_lowconf_all'] = '/'.join(str(derived[f'nl{t}']) for t in ('r1', 'r2', 'r3'))
    derived['hygiene'] = 'cases touched: 0/35' if 'cases touched: 0/35' in open(
        os.path.join(D, 'leak_audit_v12.txt'), encoding='utf-8').read() else 'MISSING'
    derived['corpus_rows'] = sum(1 for l in open(os.path.join(D, 'nli_conflict_train_v12.jsonl'),
                                                encoding='utf-8') if l.strip())
    derived['verdict'] = 'PROTOCOL VERDICT: PASS' if 'PROTOCOL VERDICT: PASS' in open(
        os.path.join(D, 'protocol_verdict_v12.log'), encoding='utf-8').read() else 'NOT PASS'
    for t in ('r1', 'r2', 'r3'):
        p = os.path.join(OUT, f'laya-nli-conflict-v12-{t}', 'model.safetensors')
        assert os.path.getsize(p) == 643835524, (t, os.path.getsize(p))
        if t == 'r2':
            derived['sha256_model_r2'] = sha256(p)

    failures = []
    for cid, claim in by_id.items():
        if cid == 'sha256_model':
            got = derived['sha256_model_r2']
        elif cid == 'weights_bytes':
            got = 643835524
        else:
            got = derived[cid]
        want = claim['value']
        if isinstance(want, float):
            ok = abs(float(got) - want) < 5e-4
        elif isinstance(want, bool):
            ok = bool(got) == want
        else:
            ok = str(got) == str(want)
        if not ok:
            failures.append((cid, want, got))
        print(f'  {"OK " if ok else "BAD"} {cid}: claimed {want!r} | derived {got!r}')

    print(f'\n{len(by_id)} claims checked, {len(failures)} failures')
    if failures:
        for cid, want, got in failures:
            print(f'  MISMATCH {cid}: claimed {want!r} vs derived {got!r}')
        sys.exit(1)
    print('VERIFY: exit 0 — claims pack matches artifacts; hf_publish permitted (§1.3)')


if __name__ == '__main__':
    main()
