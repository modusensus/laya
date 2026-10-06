# -*- coding: utf-8 -*-
"""Drive v12_orchestrate.py to completion.

Replaces the 30-min scheduler that used to tick the orchestrator (it went
silent with the previous operator). The orchestrator itself is idempotent and
self-locking, so a plain loop is enough: each turn runs one tick, then waits.

Stops on phase 'done' (verdict written) or an ERROR_* phase (human review).

Usage: D:\\Miniconda\\envs\\laya-ft\\python.exe v12_drive.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

PY = r'D:\Miniconda\envs\laya-ft\python.exe'
ORCH_DIR = r'D:\laya\kaggle_eval'
ORCH = os.path.join(ORCH_DIR, 'v12_orchestrate.py')
STATE = r'D:\laya\data_local\v12_orchestrate_state.json'
INTERVAL = 300          # seconds between ticks; a busy tick blocks for its own duration
MAX_TURNS = 400


def phase():
    return json.load(open(STATE, encoding='utf-8'))['phase']


def main():
    for turn in range(1, MAX_TURNS + 1):
        p = phase()
        stamp = datetime.now().strftime('%m-%d %H:%M:%S')
        if p == 'done' or p.startswith('ERROR'):
            print(f'[{stamp}] drive stop: phase={p}', flush=True)
            return 0
        print(f'[{stamp}] drive turn {turn}: phase={p}', flush=True)
        r = subprocess.run([PY, ORCH], cwd=ORCH_DIR)
        if r.returncode not in (0,):
            print(f'[{stamp}] tick rc={r.returncode} (phase={phase()})', flush=True)
        time.sleep(INTERVAL)
    print('drive stop: MAX_TURNS reached', flush=True)
    return 2


if __name__ == '__main__':
    sys.exit(main())
