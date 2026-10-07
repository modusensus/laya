# -*- coding: utf-8 -*-
"""Mirror the Kaggle dataset daphnelaurent/nli-conflict-pairs to the Hub as
slow-stack/nli-conflict-pairs (dataset repo) with a proper dataset card.

Prereq: `python -m kaggle datasets download daphnelaurent/nli-conflict-pairs
--unzip -p <dir>` already run. Idempotent: re-upload overwrites in place.
"""
import os
import sys

from huggingface_hub import HfApi, create_repo

REPO = 'slow-stack/nli-conflict-pairs'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else r'D:\kaggle_pull\nli-conflict-pairs-v15'
CARD = os.path.join(HERE, 'dataset_card_nli_conflict_pairs.md')
FILES = ['nli_conflict_train.jsonl', 'nli_conflict_val.jsonl', 'nli_conflict_val_soft.jsonl']
EXPECTED_ROWS = {'nli_conflict_train.jsonl': 15950, 'nli_conflict_val.jsonl': 1000,
                 'nli_conflict_val_soft.jsonl': 300}


def main():
    api = HfApi()
    create_repo(REPO, repo_type='dataset', exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=CARD, path_in_repo='README.md', repo_id=REPO, repo_type='dataset')
    print('uploaded README.md (dataset card)', flush=True)
    for f in FILES:
        p = os.path.join(SRC, f)
        n = sum(1 for _ in open(p, encoding='utf-8'))
        assert n == EXPECTED_ROWS[f], f'{f}: {n} 行 != 预期 {EXPECTED_ROWS[f]}'
        api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO, repo_type='dataset')
        print(f'uploaded {f} ({n} rows)', flush=True)
    kaggle_readme = os.path.join(SRC, '_README.md')
    if os.path.exists(kaggle_readme):
        api.upload_file(path_or_fileobj=kaggle_readme, path_in_repo='kaggle_README.md', repo_id=REPO, repo_type='dataset')
        print('uploaded kaggle_README.md (version table provenance)', flush=True)

    # 读回校验:外部写入必须回读确认
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    missing = [f for f in FILES + ['README.md', 'kaggle_README.md'] if f not in remote]
    assert not missing, f'回读缺文件: {missing}'
    for f in FILES:
        local = os.path.getsize(os.path.join(SRC, f))
        assert remote[f] == local, f'{f} 字节数不一致 local={local} remote={remote[f]}'
    print(f'\nOK {len(remote)} files -> https://huggingface.co/datasets/{REPO}', flush=True)


if __name__ == '__main__':
    main()
