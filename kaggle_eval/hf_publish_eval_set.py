# -*- coding: utf-8 -*-
"""Publish the frozen evaluation face of the multi-run protocol as the Hub
dataset slow-stack/laya-nli-conflict-eval.

One row per case, exported PROGRAMMATICALLY from the instrument modules (no
hand transcription): realtest_v2.CASES/NEGATION_CASES, realtest_v4.NEW_CASES,
noul_bias_diag.CONTROL_PAIRS/SWAP_PAIRS, family_diag.FAMILY_CASES,
negfam_diag.NEGFAM_CASES.  Hard asserts: 66 cases total, unique (known,new)
pairs, gold polarity per instrument, and FIELD-LEVEL zero overlap against the
current v12 training corpus (the same discipline leak_audit.py enforces).

Idempotent: re-run overwrites in place; readback-verified.
"""
import json
import os
import re

from huggingface_hub import HfApi, create_repo

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)

from realtest_v2 import CASES, NEGATION_CASES
from realtest_v4 import NEW_CASES
from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS, QUESTION
from family_diag import FAMILY_CASES
from negfam_diag import NEGFAM_CASES

REPO = 'slow-stack/laya-nli-conflict-eval'
D = os.path.join(HERE, '..', 'data_local')
CJK = re.compile(r'[\u4e00-\u9fff]')

GROUPS = [
    ('acceptance_old20', 'gate #2 old-20 (median = 20/20 AND no case missed in >=2 runs)', CASES, None),
    ('acceptance_negation5', 'gate #4 negation-5 (median = 5/5 AND no case missed in >=2 runs)', NEGATION_CASES, None),
    ('acceptance_new10', 'gate #3 new-10 (median = 10/10 AND no case missed in >=2 runs)', NEW_CASES, None),
    ('bias_diag_control', 'gate #11 bias diag (>=2/3 runs >=13/14; controls track the input)', CONTROL_PAIRS, None),
    ('bias_diag_swap', 'gate #11 bias diag (literal known/new swaps keep their label)', SWAP_PAIRS, None),
    ('family_compat', 'gate #12 family rate (median mean p_conflict <0.5 AND every run <=1 case p>=0.9)', FAMILY_CASES, None),
    ('family_negation', 'gate #13 negation family rate (median mean p_conflict >=0.5 AND every run <=1 case p<0.5)', NEGFAM_CASES, None),
]

CHRONIC = {'否定-养宠物反转', '新-订阅服务(否定句)'}


def lang_of(s):
    return 'zh' if CJK.search(s) else 'en'


def build_rows():
    rows = []
    for inst, gate, cases, _ in GROUPS:
        for entry in cases:
            (state, gold), note = entry[0], entry[1]
            flags = {}
            if inst == 'family_compat':
                flags['is_b2'] = bool(entry[2])
            if inst == 'family_negation':
                flags['chronic_domain'] = bool(entry[2])
            if note in CHRONIC:
                flags['chronic_acceptance'] = True
            rows.append({'instrument': inst, 'gate': gate,
                         'note': note, 'lang': lang_of(state['known'] + state['new']),
                         'known': state['known'], 'new': state['new'], 'gold': gold,
                         'flags': flags})
    return rows


def main():
    rows = build_rows()
    counts = {}
    for r in rows:
        counts[r['instrument']] = counts.get(r['instrument'], 0) + 1
    expect = {'acceptance_old20': 20, 'acceptance_negation5': 5, 'acceptance_new10': 10,
              'bias_diag_control': 10, 'bias_diag_swap': 4, 'family_compat': 9, 'family_negation': 8}
    assert counts == expect, (counts, expect)
    assert len(rows) == 66, len(rows)
    # Internal sharing is BY DESIGN: diag controls/swaps reuse acceptance
    # sentences, and B2 anchors family_compat F0 for v7-v10 reading continuity.
    # Pin the exact designed sharing; family_negation shares nothing with
    # anything else. (Hygiene that matters = overlap vs TRAINING corpus, below.)
    dups = {}
    for r in rows:
        dups.setdefault((r['known'], r['new']), []).append(r['note'])
    designed = {
        ('用户住在上海', '用户最近搬到北京工作了'): {'地点变更', '对照A1 地点变更', '交换A1'},
        ('用户对花生过敏', '用户说他可以吃花生酱了'): {'关键事实反转', '对照D1 过敏反转'},
        ('用户的项目部署在阿里云', '项目已经迁移到腾讯云了'): {'部署变更', '对照E1 部署变更'},
        ('用户的猫叫小白', '用户的猫叫小黑'): {'名字矛盾', '对照B1 名字矛盾'},
        ('用户的猫叫小白', '用户的猫是白色的'): {'对照B2 名字一致', '家族F0 (B2) 名字一致'},
    }
    shared = {k: set(v) for k, v in dups.items() if len(v) > 1}
    assert len(dups) == 60 and shared == designed, (len(dups), shared)
    nf_pairs = {(r['known'], r['new']) for r in rows if r['instrument'] == 'family_negation'}
    assert all(sum((r['known'], r['new']) == p for r in rows) == 1 for p in nf_pairs), \
        'family_negation must not share pairs with any other instrument'
    fam_gold = {r['gold'] for r in rows if r['instrument'] == 'family_compat'}
    nf_gold = {r['gold'] for r in rows if r['instrument'] == 'family_negation'}
    assert fam_gold == {'false'} and nf_gold == {'true'}, (fam_gold, nf_gold)
    chronic = sorted(r['note'] for r in rows if r['flags'].get('chronic_acceptance'))
    assert chronic == ['否定-养宠物反转', '新-订阅服务(否定句)'], chronic

    # hygiene re-assertion: field-level zero overlap with the current v12 corpus
    train_fields = set()
    for l in open(os.path.join(D, 'nli_conflict_train_v12.jsonl'), encoding='utf-8'):
        if l.strip():
            s = json.loads(json.loads(l)['state'])
            train_fields.add(s['known']); train_fields.add(s['new'])
    hits = [(r['known'], r['new']) for r in rows
            if r['known'] in train_fields or r['new'] in train_fields]
    assert not hits, f'case fields hit the v12 train corpus: {hits[:3]}'
    print(f'66 cases assembled; instruments {counts}; v12-corpus field overlap 0')

    out = os.path.join(D, 'laya_nli_conflict_eval.jsonl')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    qpath = os.path.join(D, 'question_noul.json')
    with open(qpath, 'w', encoding='utf-8') as f:
        json.dump(QUESTION, f, ensure_ascii=False, indent=2)

    card = os.path.join(HERE, 'dataset_card_nli_conflict_eval.md')
    api = HfApi()
    create_repo(REPO, repo_type='dataset', exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=card, path_in_repo='README.md', repo_id=REPO, repo_type='dataset')
    api.upload_file(path_or_fileobj=out, path_in_repo='laya_nli_conflict_eval.jsonl',
                    repo_id=REPO, repo_type='dataset')
    api.upload_file(path_or_fileobj=qpath, path_in_repo='question_noul.json',
                    repo_id=REPO, repo_type='dataset')
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
    remote = {f.rfilename: f.size for f in info.siblings}
    for name in ('README.md', 'laya_nli_conflict_eval.jsonl', 'question_noul.json'):
        assert name in remote and remote[name] > 0, (name, remote)
    # readback: row count + first row round-trip
    from huggingface_hub import hf_hub_download
    back = hf_hub_download(REPO, 'laya_nli_conflict_eval.jsonl', repo_type='dataset')
    back_rows = [json.loads(l) for l in open(back, encoding='utf-8') if l.strip()]
    assert len(back_rows) == 66 and back_rows[0]['known'] == rows[0]['known']
    print(f'OK {len(remote)} files, readback 66 rows -> https://huggingface.co/datasets/{REPO}')


if __name__ == '__main__':
    main()
