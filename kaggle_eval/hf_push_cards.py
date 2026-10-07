# -*- coding: utf-8 -*-
"""Push the two model cards (README.md) to the Hub — round-4 改动4 sync.

Usage: python hf_push_cards.py
"""
from huggingface_hub import HfApi

JOBS = [
    (r'D:\laya\kaggle_eval\hf_model_card.md',
     'slow-stack/laya-typed-decisions-multilingual'),
    (r'D:\laya\kaggle_eval\hf_model_card_nli_conflict.md',
     'slow-stack/laya-nli-memory-conflict'),
]

api = HfApi()
for path, repo in JOBS:
    res = api.upload_file(path_or_fileobj=path, path_in_repo='README.md',
                          repo_id=repo, repo_type='model',
                          commit_message='round-4 diagnostics: same-benchmark comparison, '
                                         'option-name polarity, conformal abstention layer')
    print('pushed', repo, '->', res)
