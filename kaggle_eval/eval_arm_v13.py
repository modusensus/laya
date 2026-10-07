# -*- coding: utf-8 -*-
"""Round-13 per-run local eval battery — eval_arm_v11.py + the negfam section
(HANDOFF_NLI_V12.md §2.2/§5).  All artifact names carry the v13_ infix; the
run()/rc policy is the frozen v11 one (gate FAILs are informational, the
battery is broken only when an expected artifact is missing/empty).

Usage: D:/Miniconda/envs/laya-ft/python.exe eval_arm_v13.py <ckpt> <tag>
"""
import os
import shutil
import subprocess
import sys

CKPT, TAG = sys.argv[1], sys.argv[2]
D = r'D:\laya\data_local'
KAGGLE_EVAL = r'D:\laya\kaggle_eval'
P = r'D:\Miniconda\envs\laya-ft\python.exe'
RC = 0


def run(label, args, expect=None, tails=3):
    """Gate scripts exit nonzero when their GATE fails (v9/v10 convention) —
    that is informational: protocol_verdict.py owns gate verdicts. The battery
    is "broken" only when an expected artifact is missing/empty."""
    global RC
    print(f'== {label} [{TAG}] ==', flush=True)
    r = subprocess.run([P] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', cwd=KAGGLE_EVAL)
    lines = ((r.stdout or '') + '\n' + (r.stderr or '')).strip().splitlines()
    for line in lines[-tails:]:
        print('   ' + line, flush=True)
    if r.returncode != 0:
        print(f'   (section rc={r.returncode}; gate outcome -> verdict tool)', flush=True)
    if expect is not None and not (os.path.exists(expect) and os.path.getsize(expect) > 0):
        RC = 1
        print(f'   (MISSING artifact: {expect})', flush=True)


shutil.copy(os.path.join(CKPT, 'val_probs.json'),
            os.path.join(D, f'val_probs_v13_{TAG}.json'))

run('realtest (old20/neg5/new10)',
    ['realtest_v4.py', CKPT, f'{D}/memory_conflict_realtest_v13_{TAG}.json'],
    expect=f'{D}/memory_conflict_realtest_v13_{TAG}.json')
run('dump val_soft', ['dump_probs.py', CKPT, f'{D}/val_soft_probs_v13_{TAG}.json'],
    expect=f'{D}/val_soft_probs_v13_{TAG}.json')
run('polarity', ['option_polarity_check.py', CKPT, f'{D}/polarity_nli_v13_{TAG}.json'],
    expect=f'{D}/polarity_nli_v13_{TAG}.json')
run('label swap', ['label_swap_check.py', CKPT, f'{D}/label_swap_v13_{TAG}.json'],
    expect=f'{D}/label_swap_v13_{TAG}.json')
run('bias diag', ['noul_bias_diag.py', CKPT, f'{D}/noul_bias_diag_v13_{TAG}.json'],
    expect=f'{D}/noul_bias_diag_v13_{TAG}.json')
run('band', ['band_report.py',
             '--main', f'{D}/val_probs_v13_{TAG}.json',
             '--soft', f'{D}/val_soft_probs_v13_{TAG}.json',
             '--out', f'{D}/conf_band_v13_{TAG}.txt',
             '--round-tag', f'v13-{TAG}'], expect=f'{D}/conf_band_v13_{TAG}.txt')
run('in-dist probe', ['probe_indist_v9.py', CKPT, f'{D}/indist_probe_v13_{TAG}.json'],
    expect=f'{D}/indist_probe_v13_{TAG}.json')
run('conformal alt probe (sel budget 0.34)',
    ['conformal_alt_probe.py', CKPT, f'{D}/conformal_alt_v13_{TAG}.json',
     '--val-probs', f'{D}/val_probs_v13_{TAG}.json', '--sel-budget', '0.34'],
    expect=f'{D}/conformal_alt_v13_{TAG}.json')
run('conformal maxconf archive',
    ['conformal_abstain.py',
     '--main', f'{D}/val_probs_v13_{TAG}.json',
     '--soft', f'{D}/val_soft_probs_v13_{TAG}.json',
     '--real', f'{D}/memory_conflict_realtest_v13_{TAG}.json',
     '--diag', f'{D}/noul_bias_diag_v13_{TAG}.json',
     '--out', f'{D}/conformal_v13_{TAG}.json', '--split-half'],
    expect=f'{D}/conformal_v13_{TAG}.json')
run('conformal s1 (pinned tie-break)',
    ['conformal_abstain.py', '--rule', 's1_maxconf', '--split-half', '--sel-budget', '0.34',
     '--alt-scores', f'{D}/conformal_alt_v13_{TAG}.json',
     '--main', f'{D}/val_probs_v13_{TAG}.json',
     '--soft', f'{D}/val_soft_probs_v13_{TAG}.json',
     '--real', f'{D}/memory_conflict_realtest_v13_{TAG}.json',
     '--diag', f'{D}/noul_bias_diag_v13_{TAG}.json',
     '--out', f'{D}/conformal_v13_{TAG}_s1.json'],
    expect=f'{D}/conformal_v13_{TAG}_s1.json')
run('conformal s2',
    ['conformal_abstain.py', '--rule', 's2', '--split-half', '--sel-budget', '0.34',
     '--alt-scores', f'{D}/conformal_alt_v13_{TAG}.json',
     '--main', f'{D}/val_probs_v13_{TAG}.json',
     '--soft', f'{D}/val_soft_probs_v13_{TAG}.json',
     '--real', f'{D}/memory_conflict_realtest_v13_{TAG}.json',
     '--diag', f'{D}/noul_bias_diag_v13_{TAG}.json',
     '--out', f'{D}/conformal_v13_{TAG}_s2.json'],
    expect=f'{D}/conformal_v13_{TAG}_s2.json')
run('family diag (v11 gate #12, regression guard)',
    ['family_diag.py', CKPT, f'{D}/family_diag_v13_{TAG}.json'],
    expect=f'{D}/family_diag_v13_{TAG}.json')
run('negfam diag (NEW V12 §2.2)', ['negfam_diag.py', CKPT, f'{D}/negfam_diag_v13_{TAG}.json'],
    expect=f'{D}/negfam_diag_v13_{TAG}.json')

print(f'== done [{TAG}] rc={RC} ==', flush=True)
sys.exit(RC)
