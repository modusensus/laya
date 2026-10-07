# -*- coding: utf-8 -*-
"""Upload the undelivered round-5..10 checkpoints to the Hub as research
archive repos: one repo per checkpoint, each carrying a "NOT delivered"
model card. The delivered head (v4) is NOT touched.

Auth: `hf auth login` already done; the token is never passed on the command
line and never written by this script. Set HF_HOME to a non-C: cache dir to
keep the C: drive out of the upload path.

Usage: python hf_publish_archive.py [v6]        # one repo, or all if omitted
"""
import os
import sys

from huggingface_hub import HfApi, create_repo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r'D:\laya-kaggle-output'
CARDS = os.path.join(HERE, 'archive_cards')

# repo_id -> (local checkpoint dir, card file name)
ARCHIVE = {
    'slow-stack/laya-nli-conflict-v5':    (rf'{OUT}\laya-nli-conflict-v5\laya-nli-conflict', 'v5.md'),
    'slow-stack/laya-nli-conflict-v5-ce': (rf'{OUT}\laya-nli-conflict-v5-ce\laya-nli-conflict', 'v5-ce.md'),
    'slow-stack/laya-nli-conflict-v6':    (rf'{OUT}\laya-nli-conflict-v6', 'v6.md'),
    'slow-stack/laya-nli-conflict-v7':    (rf'{OUT}\laya-nli-conflict-v7', 'v7.md'),
    'slow-stack/laya-nli-conflict-v8':    (rf'{OUT}\laya-nli-conflict-v8', 'v8.md'),
    'slow-stack/laya-nli-conflict-v9':    (rf'{OUT}\laya-nli-conflict-v9', 'v9.md'),
    'slow-stack/laya-nli-conflict-v10-l1': (rf'{OUT}\laya-nli-conflict-v10-l1', 'v10-l1.md'),
    'slow-stack/laya-nli-conflict-v10-l2': (rf'{OUT}\laya-nli-conflict-v10-l2', 'v10-l2.md'),
    'slow-stack/laya-nli-conflict-v10-s2': (rf'{OUT}\laya-nli-conflict-v10-s2', 'v10-s2.md'),
}
FILES = ['model.safetensors', 'rl_agent_config.json', 'metrics.json', 'val_probs.json']
DIRS = ['encoder', 'tokenizer']


def publish(repo, ckpt, card):
    api = HfApi()
    create_repo(repo, repo_type='model', exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=os.path.join(CARDS, card), path_in_repo='README.md', repo_id=repo)
    print('uploaded', repo, 'README.md', flush=True)
    for f in FILES:
        p = os.path.join(ckpt, f)
        assert os.path.exists(p), p
        api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=repo)
        print('uploaded', repo, f, flush=True)
    for d in DIRS:
        api.upload_folder(folder_path=os.path.join(ckpt, d), path_in_repo=d, repo_id=repo)
        print('uploaded', repo, d + '/', flush=True)

    # 读回校验:外部写入必须回读确认(文件在位 + 权重字节数一致)
    info = api.repo_info(repo, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    missing = [f for f in FILES if f not in remote]
    for d in DIRS:
        if not any(x.startswith(d + '/') for x in remote):
            missing.append(d + '/')
    assert not missing, f'{repo} 回读缺文件: {missing}'
    assert 'README.md' in remote, f'{repo} README(model card) 未上传'
    local = os.path.getsize(os.path.join(ckpt, 'model.safetensors'))
    assert remote.get('model.safetensors') == local, f'{repo} model.safetensors 字节数不一致'
    print(f'OK {repo}: {len(remote)} files, model size {local} verified', flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for repo, (ckpt, card) in ARCHIVE.items():
        if only and not repo.endswith(only):
            continue
        publish(repo, ckpt, card)
    print('\nALL DONE', flush=True)


if __name__ == '__main__':
    main()
