# -*- coding: utf-8 -*-
"""Exact-field leak audit: frozen realtest case texts vs training corpus.

Field-level equality on parsed state known/new (no substring matching --
substrings yield false positives, e.g. a superset sentence containing a
case sentence is a different sentence).

Usage:
    python kaggle_eval/leak_audit.py                 # default: v5 train vs v4 train
    python kaggle_eval/leak_audit.py <train.jsonl>   # audit any generated corpus

Expected post-fix (v6): 0 cases touched on the generated corpus.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from realtest_v2 import CASES, NEGATION_CASES          # noqa: E402
from realtest_v4 import NEW_CASES                       # noqa: E402

DATA = HERE.parent / 'data_local'


def rows(path):
    out = []
    for l in open(path, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            s = r['state']
            out.append(json.loads(s) if isinstance(s, str) else s)
    return out


def analyze(train, tag):
    knowns = Counter()
    news = Counter()
    pairs = []
    for s in train:
        k, n = s.get('known'), s.get('new')
        knowns[k] += 1
        news[n] += 1
        pairs.append((k, n))
    print(f"--- {tag}: {len(train)} rows")
    hits = 0
    for (state, _e), note in list(CASES) + list(NEGATION_CASES) + list(NEW_CASES):
        k, n = state['known'], state['new']
        kc, nc = knowns.get(k, 0), news.get(n, 0)
        pairc = sum(1 for (a, b) in pairs if a == k and b == n)
        kk = 'known HIT' if kc else ''
        nn = 'new HIT' if nc else ''
        pp = 'FULL-PAIR HIT' if pairc else ''
        if kc or nc or pairc:
            hits += 1
            print(f"  {note:28s} k={kc:3d} n={nc:3d} pair={pairc} {kk} {nn} {pp}")
    print(f"  -> cases touched: {hits}/35")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    else:
        targets = [DATA / 'nli_conflict_train_v5.jsonl', DATA / 'nli_conflict_train_v4.jsonl']
    for p in targets:
        analyze(rows(str(p)), p.name)
