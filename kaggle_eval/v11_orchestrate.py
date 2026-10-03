# -*- coding: utf-8 -*-
"""v11 r1/r2/r3 orchestrator (HANDOFF_NLI_V11.md §1.1/§1.3).

Idempotent tick driven by a state file; safe to re-run / schedule.

Discipline baked in (do not weaken):
  - NO kernel push before the GPU-quota reset (RESET_AT, local time);
  - PULL-BEFORE-PUSH (v10 lesson): run N's output is pulled and its eval
    battery finished BEFORE run N+1 is pushed (Kaggle CLI can only pull the
    LATEST version's output);
  - one long operation per tick; a fresh lock file blocks overlapping ticks;
  - any command failure -> state "ERROR_<phase>" and no further pushes
    (human review); the tick then always no-ops;
  - after r3's battery the protocol verdict is written and the orchestrator
    STOPS -- HF delivery is a separate human-gated step (§1.3).

Usage: D:/Miniconda/envs/laya-ft/python.exe v11_orchestrate.py [--force-now]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

PY = r'D:\Miniconda\envs\laya-ft\python.exe'
STAGE = r'C:\kaggle_cfg\stage'
D = r'D:\laya\data_local'
OUT = r'D:\laya-kaggle-output'
KAGGLE_EVAL = r'D:\laya\kaggle_eval'
KAGGLE_ID = 'daphnelaurent/laya-nli-conflict-ce'
STATE_P = os.path.join(D, 'v11_orchestrate_state.json')
LOG_P = os.path.join(D, 'v11_orchestrate.log')
LOCK_P = os.path.join(D, 'v11_orchestrate.lock')
RESET_AT = '2026-10-03 09:05'
RUNS = ('r1', 'r2', 'r3')
KERNEL_META = {
    "id": KAGGLE_ID, "title": "Laya nli conflict ce", "code_file": "nli_kernel_ce.py",
    "language": "python", "kernel_type": "script", "is_private": "true",
    "enable_gpu": "true", "enable_internet": "true",
    "dataset_sources": ["daphnelaurent/nli-conflict-pairs"],
    "competition_sources": [], "kernel_sources": [], "model_sources": [],
}


def log(msg):
    line = f'[{datetime.now():%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    with open(LOG_P, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sh(cmd, check=True, cwd=None):
    log('$ ' + ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', cwd=cwd)
    out = '\n'.join(((r.stdout or '') + '\n' + (r.stderr or '')).strip().splitlines()[-8:])
    log(out if out.strip() else f'(rc={r.returncode})')
    if check and r.returncode != 0:
        raise RuntimeError(f'rc={r.returncode}: {" ".join(cmd)}')
    return r


def sh_env(cmd, check=True):
    """kaggle CLI with env auth; the key is read in-process, never logged."""
    key = json.load(open(r'C:/kaggle_cfg/kaggle.json', encoding='utf-8'))['key']
    env = dict(os.environ, KAGGLE_USERNAME='daphnelaurent', KAGGLE_KEY=key,
               TMPDIR=r'C:\kaggle_cfg\tmp')
    log('$ (kaggle) ' + ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env)
    out = '\n'.join(((r.stdout or '') + '\n' + (r.stderr or '')).strip().splitlines()[-8:])
    log(out if out.strip() else f'(rc={r.returncode})')
    if check and r.returncode != 0:
        raise RuntimeError(f'rc={r.returncode}: {" ".join(cmd)}')
    return r


def kernel_status():
    r = sh_env([PY, '-m', 'kaggle', 'kernels', 'status', KAGGLE_ID], check=False)
    txt = ((r.stdout or '') + (r.stderr or '')).lower()
    if 'complete' in txt:
        return 'complete'
    if 'error' in txt or 'cancel' in txt:
        return 'error'
    return 'running'


def main():
    if os.path.exists(LOCK_P) and time.time() - os.path.getmtime(LOCK_P) < 3 * 3600:
        print('tick skipped: another tick holds a fresh lock', flush=True)
        return 0
    open(LOCK_P, 'w').write(str(os.getpid()))
    try:
        return tick()
    except Exception as e:
        s = load_state()
        if not s['phase'].startswith('ERROR'):
            s['phase'] = f'ERROR_{s["phase"]}'
            save_state(s)
        log(f'FATAL: {e} -> state={s["phase"]} (no further pushes; human review)')
        return 2
    finally:
        os.remove(LOCK_P)


def load_state():
    if os.path.exists(STATE_P):
        return json.load(open(STATE_P, encoding='utf-8'))
    return {'phase': 'init', 'done': []}


def save_state(s):
    json.dump(s, open(STATE_P, 'w', encoding='utf-8'), indent=1)


def tick():
    s = load_state()
    if s['phase'].startswith('ERROR') or s['phase'] == 'done':
        log(f'state={s["phase"]} — no-op (human review / finished)')
        return 0

    if s['phase'] == 'init':
        corpus = os.path.join(D, 'nli_conflict_train_v11.jsonl')
        n = sum(1 for _ in open(corpus, encoding='utf-8'))
        assert n == 15914, f'v11 corpus rows {n} != 15914'
        assert os.path.exists(os.path.join(KAGGLE_EVAL, 'protocol_verdict.py'))
        s = {'phase': 'wait_reset', 'done': [], 'corpus_rows': n}
        save_state(s)

    if s['phase'] == 'wait_reset':
        now = datetime.now()
        reset = datetime.strptime(RESET_AT, '%Y-%m-%d %H:%M')
        if '--force-now' not in sys.argv and now < reset:
            log(f'waiting for GPU-quota reset at {RESET_AT} (now {now:%H:%M}); corpus rows={s["corpus_rows"]}')
            return 0
        log('reset window reached -> pushing r1')
        s['phase'] = 'push_r1'
        save_state(s)

    for i, tag in enumerate(RUNS):
        if s['phase'] == f'push_{tag}':
            d = os.path.join(STAGE, f'kernel_v11_{tag}')
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(KAGGLE_EVAL, 'nli_kernel_ce.py'),
                        os.path.join(d, 'nli_kernel_ce.py'))
            json.dump(KERNEL_META, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8'), indent=1)
            sh_env([PY, '-m', 'kaggle', 'kernels', 'push', '-p', d])
            s[f'pushed_{tag}_at'] = datetime.now().isoformat(timespec='seconds')
            s['phase'] = f'wait_{tag}'
            save_state(s)
            return 0
        if s['phase'] == f'wait_{tag}':
            st = kernel_status()
            if st == 'running':
                log(f'{tag} still running; will re-check next tick')
                return 0
            if st == 'error':
                s['phase'] = f'ERROR_wait_{tag}'
                save_state(s)
                log(f'{tag} kernel reported ERROR/cancelled')
                return 2
            s['phase'] = f'pull_{tag}'
            save_state(s)
        if s['phase'] == f'pull_{tag}':
            pull = os.path.join(STAGE, f'kernel_v11_{tag}_out')
            # failure taxonomy (2026-10-03 r1 lesson): NETWORK flakes (SSL EOF /
            # IncompleteRead / RemoteDisconnected) retry within the tick and
            # across ticks by LEAVING the phase unchanged; only INTEGRITY
            # assertion failures below go to ERROR state.
            for attempt in range(1, 4):
                shutil.rmtree(pull, ignore_errors=True)
                os.makedirs(pull)
                try:
                    sh_env([PY, '-m', 'kaggle', 'kernels', 'output', KAGGLE_ID, '-p', pull])
                    break
                except RuntimeError as e:
                    if attempt == 3:
                        log(f'pull network-flaked 3x this tick; phase stays {s["phase"]} '
                            f'— next tick retries (not an ERROR)')
                        return 0
                    log(f'pull attempt {attempt}/3 network-failed; retry in 120s')
                    time.sleep(120)
            # Kaggle preserves the kernel's working subdir (laya-nli-conflict/);
            # locate the artifact dir at either level (r1 fix, 2026-10-03)
            cands = [pull] + [os.path.join(pull, d) for d in sorted(os.listdir(pull))
                              if os.path.isdir(os.path.join(pull, d))]
            art = next((c for c in cands
                        if os.path.exists(os.path.join(c, 'model.safetensors'))), None)
            assert art, f'pull incomplete: model.safetensors not found under {pull}'
            assert os.path.getsize(os.path.join(art, 'model.safetensors')) == 643835524, \
                'model.safetensors size != architecture size (truncated pull?)'
            # version fingerprint: only the v11 corpus has 15914 rows
            logs = [os.path.join(pull, f) for f in sorted(os.listdir(pull)) if f.endswith('.log')]
            fp = any('15914 train items' in open(l, encoding='utf-8', errors='replace').read()
                     for l in logs)
            assert fp, f'version fingerprint missing (15914 train items) in {logs}'
            for f in ('val_probs.json', 'metrics.json', 'rl_agent_config.json'):
                assert os.path.exists(os.path.join(art, f)), f'pull incomplete: missing {f}'
            ckpt = os.path.join(OUT, f'laya-nli-conflict-v11-{tag}')
            shutil.copytree(art, ckpt, dirs_exist_ok=True)
            log(f'{tag} pulled from {art} -> {ckpt}')
            s['phase'] = f'eval_{tag}'
            save_state(s)
        if s['phase'] == f'eval_{tag}':
            # torch probe (SAC may block torch for a while; v10 lesson)
            for probe in range(30):
                r = subprocess.run([PY, '-c', 'import torch'], capture_output=True)
                if r.returncode == 0:
                    break
                log(f'torch blocked (probe {probe + 1}); retry in 120s')
                time.sleep(120)
            r = sh(['bash', 'C:/kaggle_cfg/stage/eval_arm_v11.sh',
                    os.path.join(OUT, f'laya-nli-conflict-v11-{tag}').replace('\\', '/'), tag],
                   check=False)
            if r.returncode != 0:
                s['phase'] = f'ERROR_eval_{tag}'
                save_state(s)
                log(f'{tag} eval battery failed rc={r.returncode}')
                return 2
            s['done'].append(tag)
            s['phase'] = f'push_{RUNS[i + 1]}' if i < len(RUNS) - 1 else 'verdict'
            save_state(s)
            return 0

    if s['phase'] == 'verdict':
        r = sh([PY, 'protocol_verdict.py', '--tags', 'r1,r2,r3', '--prefix', 'v11_'],
               check=False, cwd=KAGGLE_EVAL)
        out = '\n'.join(((r.stdout or '') + '\n' + (r.stderr or '')).strip().splitlines())
        with open(os.path.join(D, 'protocol_verdict_v11.log'), 'w', encoding='utf-8') as f:
            f.write(out + '\n')
        s['verdict_rc'] = r.returncode
        s['phase'] = 'done'
        save_state(s)
        log(f'VERDICT rc={r.returncode} ({"PASS -> eligible for delivery" if r.returncode == 0 else "FAIL -> no delivery"}); '
            f'orchestrator stopped — HF delivery is human-gated (§1.3)')
        return 0

    log(f'unhandled phase {s["phase"]} — no-op')
    return 0


if __name__ == '__main__':
    sys.exit(main())
