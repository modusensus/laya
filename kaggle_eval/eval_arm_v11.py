# -*- coding: utf-8 -*-
"""Round-11 per-run local eval battery — python port of stage/eval_arm_v11.sh.

Why the port: the orchestrator's subprocess env resolves `bash` unreliably
(WSL/Git ambiguity, CRLF hazards; 2026-10-03 r1 rc=127 twice), while every
battery step is just a kaggle_eval python script — call them directly.

Usage: D:/Miniconda/envs/laya-ft/python.exe eval_arm_v11.py <ckpt> <tag>
Exit 0 iff every section returned rc 0.
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


def run(label, args, tails=3):
    global RC
    print(f'== {label} [{TAG}] ==', flush=True)
    r = subprocess.run([P] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', cwd=KAGGLE_EVAL)
    lines = ((r.stdout or '') + '\n' + (r.stderr or '')).strip().splitlines()
    for line in lines[-tails:]:
        print('   ' + line, flush=True)
    if r.returncode != 0:
        RC = 1
        print(f'   (section rc={r.returncode})', flush=True)


shutil.copy(os.path.join(CKPT, 'val_probs.json'),
            os.path.join(D, f'val_probs_v11_{TAG}.json'))

run('realtest (old20/neg5/new10)',
    ['realtest_v4.py', CKPT, f'{D}/memory_conflict_realtest_v11_{TAG}.json'],
    expect=f'{D}/memory_conflict_realtest_v11_{TAG}.json')
run('dump val_soft', ['dump_probs.py', CKPT, f'{D}/val_soft_probs_v11_{TAG}.json'],
    expect=f'{D}/val_soft_probs_v11_{TAG}.json')
run('polarity', ['option_polarity_check.py', CKPT, f'{D}/polarity_nli_v11_{TAG}.json'],
    expect=f'{D}/polarity_nli_v11_{TAG}.json')
run('label swap', ['label_swap_check.py', CKPT, f'{D}/label_swap_v11_{TAG}.json'],
    expect=f'{D}/label_swap_v11_{TAG}.json')
run('bias diag', ['noul_bias_diag.py', CKPT, f'{D}/noul_bias_diag_v11_{TAG}.json'],
    expect=f'{D}/noul_bias_diag_v11_{TAG}.json')
run('band', ['band_report.py',
             '--main', f'{D}/val_probs_v11_{TAG}.json',
             '--soft', f'{D}/val_soft_probs_v11_{TAG}.json',
             '--out', f'{D}/conf_band_v11_{TAG}.txt',
             '--round-tag', f'v11-{TAG}'], expect=f'{D}/conf_band_v11_{TAG}.txt')
run('in-dist probe', ['probe_indist_v9.py', CKPT, f'{D}/indist_probe_v11_{TAG}.json'],
    expect=f'{D}/indist_probe_v11_{TAG}.json')
run('conformal alt probe (sel budget 0.34)',
    ['conformal_alt_probe.py', CKPT, f'{D}/conformal_alt_v11_{TAG}.json',
     '--val-probs', f'{D}/val_probs_v11_{TAG}.json', '--sel-budget', '0.34'],
    expect=f'{D}/conformal_alt_v11_{TAG}.json')
run('conformal maxconf archive',
    ['conformal_abstain.py',
     '--main', f'{D}/val_probs_v11_{TAG}.json',
     '--soft', f'{D}/val_soft_probs_v11_{TAG}.json',
     '--real', f'{D}/memory_conflict_realtest_v11_{TAG}.json',
     '--diag', f'{D}/noul_bias_diag_v11_{TAG}.json',
     '--out', f'{D}/conformal_v11_{TAG}.json', '--split-half'],
    expect=f'{D}/conformal_v11_{TAG}.json')
run('conformal s1 (pinned tie-break)',
    ['conformal_abstain.py', '--rule', 's1_maxconf', '--split-half', '--sel-budget', '0.34',
     '--alt-scores', f'{D}/conformal_alt_v11_{TAG}.json',
     '--main', f'{D}/val_probs_v11_{TAG}.json',
     '--soft', f'{D}/val_soft_probs_v11_{TAG}.json',
     '--real', f'{D}/memory_conflict_realtest_v11_{TAG}.json',
     '--diag', f'{D}/noul_bias_diag_v11_{TAG}.json',
     '--out', f'{D}/conformal_v11_{TAG}_s1.json'],
    expect=f'{D}/conformal_v11_{TAG}_s1.json')
run('conformal s2',
    ['conformal_abstain.py', '--rule', 's2', '--split-half', '--sel-budget', '0.34',
     '--alt-scores', f'{D}/conformal_alt_v11_{TAG}.json',
     '--main', f'{D}/val_probs_v11_{TAG}.json',
     '--soft', f'{D}/val_soft_probs_v11_{TAG}.json',
     '--real', f'{D}/memory_conflict_realtest_v11_{TAG}.json',
     '--diag', f'{D}/noul_bias_diag_v11_{TAG}.json',
     '--out', f'{D}/conformal_v11_{TAG}_s2.json'],
    expect=f'{D}/conformal_v11_{TAG}_s2.json')
run('family diag (NEW V11 §2.2)', ['family_diag.py', CKPT, f'{D}/family_diag_v11_{TAG}.json'],
    expect=f'{D}/family_diag_v11_{TAG}.json')

print(f'== done [{TAG}] rc={RC} ==', flush=True)
sys.exit(RC)
