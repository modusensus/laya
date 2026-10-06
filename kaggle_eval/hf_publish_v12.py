# -*- coding: utf-8 -*-
"""HANDOFF_NLI_V12.md §1.3 delivery: publish the v12 head.

Three repos in one pass (user decision 2026-10-07):
  - Modusnsus/laya-nli-conflict-v12      = r2, DELIVERED head (supersedes v4)
  - Modusnsus/laya-nli-conflict-v12-r1   = sibling research archive (NOT delivered)
  - Modusnsus/laya-nli-conflict-v12-r3   = sibling research archive (NOT delivered)

Gate: verify_claims_v12.py must have exited 0 before this script runs.
Readback: file sizes for all three repos; full weight SHA256 re-download for the
delivered repo.  Also appends a "superseded by v12" line to the v4 card and
adds the delivered repo to the project collection (both idempotent).
"""
import hashlib
import os
import shutil
import sys

from huggingface_hub import HfApi, add_collection_item, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r'D:\laya-kaggle-output'
COLLECTION = 'Modusnsus/laya-nli-memory-conflict-head-v4-and-three-run-protocol-6ac03c397eb43e9e3ecf87f0'
R2_SHA = '6648d892449c13f81107b0b5891aca260293d176567ccbe432b8a6ef4af3f72b'
FILES = ['model.safetensors', 'rl_agent_config.json', 'metrics.json', 'val_probs.json',
         'encoder/config.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json']
REPOS = {
    'Modusnsus/laya-nli-conflict-v12': (rf'{OUT}\laya-nli-conflict-v12-r2', os.path.join(HERE, 'delivery_card_v12.md')),
    'Modusnsus/laya-nli-conflict-v12-r1': (rf'{OUT}\laya-nli-conflict-v12-r1', os.path.join(HERE, 'archive_cards', 'v12-r1.md')),
    'Modusnsus/laya-nli-conflict-v12-r3': (rf'{OUT}\laya-nli-conflict-v12-r3', os.path.join(HERE, 'archive_cards', 'v12-r3.md')),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi()
    for repo, (ckpt, card) in REPOS.items():
        for f in FILES:
            assert os.path.exists(os.path.join(ckpt, f)), (repo, f)
    if input('verify_claims_v12.py exit 0 confirmed and publish 3 repos? [yes/N] ').strip() != 'yes':
        sys.exit('aborted by operator')

    for repo, (ckpt, card) in REPOS.items():
        api.create_repo(repo, repo_type='model', exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=card, path_in_repo='README.md', repo_id=repo)
        for f in FILES:
            print(f'   {repo} <- {f}', flush=True)
            api.upload_file(path_or_fileobj=os.path.join(ckpt, f), path_in_repo=f, repo_id=repo)
        info = api.repo_info(repo, repo_type='model', files_metadata=True)
        remote = {f.rfilename: f.size for f in info.siblings}
        for f in FILES + ['README.md']:
            assert f in remote and remote[f] == os.path.getsize(os.path.join(ckpt, f)) \
                if f != 'README.md' else f in remote, (repo, f)
        print(f'OK {repo}: {len(remote)} files, sizes verified', flush=True)

    # delivered repo: full weight readback hash
    back = hf_hub_download('Modusnsus/laya-nli-conflict-v12', 'model.safetensors', repo_type='model')
    got = sha256(back)
    assert got == R2_SHA, (got, R2_SHA)
    print(f'delivered weight readback SHA256 OK: {got}', flush=True)

    # v4 card: append the superseded note (idempotent)
    v4 = 'Modusnsus/laya-nli-memory-conflict'
    p = hf_hub_download(v4, 'README.md', repo_type='model')
    txt = open(p, encoding='utf-8').read()
    note = ('\n> **Superseded (2026-10-07)**: [`Modusnsus/laya-nli-conflict-v12`](https://huggingface.co/Modusnsus/laya-nli-conflict-v12)'
            ' — the first head to pass the full preregistered 14-axis three-run protocol — is now the delivered head.'
            ' v4 remains published for reference.\n')
    if 'Superseded (2026-10-07)' not in txt:
        api.upload_file(path_or_fileobj=(txt.rstrip() + '\n' + note).encode('utf-8'),
                        path_in_repo='README.md', repo_id=v4)
        print('v4 card: superseded note appended', flush=True)
    else:
        print('v4 card: note already present', flush=True)

    add_collection_item(COLLECTION, item_id='Modusnsus/laya-nli-conflict-v12',
                        item_type='model', exists_ok=True)
    print('collection: v12 delivered head added', flush=True)
    print('DELIVERY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
