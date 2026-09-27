# -*- coding: utf-8 -*-
"""Publish the delivered conflict head to the Hub as a loadable model repo.

Auth: run `hf auth login` once (interactive). The token is never passed on the
command line and never written to a file by this script.

Usage: python hf_publish.py [user/repo] [ckpt_dir]
       (defaults: <your-hf-name>/laya-nli-memory-conflict, D:/laya-kaggle-output/laya-nli-conflict-v4)
"""
import os
import sys

from huggingface_hub import HfApi, create_repo

REPO = sys.argv[1] if len(sys.argv) > 1 else None
CKPT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
CARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_model_card_nli_conflict.md')
WEIGHT_FILES = ['model.safetensors', 'rl_agent_config.json', 'metrics.json']
WEIGHT_DIRS = ['encoder', 'tokenizer']


def main():
    api = HfApi()
    repo = REPO or f"{api.whoami()['name']}/laya-nli-memory-conflict"
    create_repo(repo, repo_type='model', exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=CARD, path_in_repo='README.md', repo_id=repo)
    for f in WEIGHT_FILES:
        api.upload_file(path_or_fileobj=os.path.join(CKPT, f), path_in_repo=f, repo_id=repo)
        print('uploaded', f)
    for d in WEIGHT_DIRS:
        api.upload_folder(folder_path=os.path.join(CKPT, d), path_in_repo=d, repo_id=repo)
        print('uploaded', d + '/')

    # 读回校验:外部写入必须回读确认
    files = set(api.list_repo_files(repo))
    missing = [f for f in WEIGHT_FILES if f not in files] + [d + '/' for d in WEIGHT_DIRS
                                                             if not any(x.startswith(d + '/') for x in files)]
    assert not missing, f'回读缺文件: {missing}'
    assert 'README.md' in files, 'README(model card) 未上传'
    print(f'\nOK {len(files)} files -> https://huggingface.co/{repo}')


if __name__ == '__main__':
    main()
