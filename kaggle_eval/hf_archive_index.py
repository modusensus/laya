# -*- coding: utf-8 -*-
"""One-off: create the HF collection linking v4 + the nine round-5..10
research-archive repos, then append the archive index section to the
delivered v4 model card. Idempotent: re-running skips done work.
"""
import io
import sys

from huggingface_hub import HfApi, hf_hub_download

V4 = 'Modusnsus/laya-nli-memory-conflict'
COLL = 'Modusnsus/laya-nli-conflict-rounds'
# 合集范围(用户拍板 2026-10-03):现役 v4 + 三跑协议核心样本(v9 / S2 / L2)
COLLECTION_ITEMS = [
    V4,
    'Modusnsus/laya-nli-conflict-v9',
    'Modusnsus/laya-nli-conflict-v10-s2',
    'Modusnsus/laya-nli-conflict-v10-l2',
]
ARCHIVE = [
    ('laya-nli-conflict-v5', 'Round 5 arm A — RL+CE (v4 recipe); RL term clashed with graded soft targets'),
    ('laya-nli-conflict-v5-ce', 'Round 5 arm B — pure CE; val_soft/polarity gates failed'),
    ('laya-nli-conflict-v6', 'Round 6 — data-hygiene reset; conformal/compression-band gates failed'),
    ('laya-nli-conflict-v7', 'Round 7 — 11/12 gates; B2 bias probe flipped (pet-family imbalance)'),
    ('laya-nli-conflict-v8', 'Round 8 — B2 dose-response −15pp; abstain 35.4% over the line'),
    ('laya-nli-conflict-v9', 'Round 9 — main val 0.9050, val_soft 0/300; B2 dose-response falsified'),
    ('laya-nli-conflict-v10-l1', 'Round 10 — "remove block B" lever (reconstructed ckpt); block B is load-bearing'),
    ('laya-nli-conflict-v10-l2', 'Round 10 — v9-config rerun; completes the three-run noise band'),
    ('laya-nli-conflict-v10-s2', 'Round 10 — v9-config noise baseline; three-run B2 evidence'),
]
ANCHOR = '## Rounds 5–10 research archive'


def main():
    api = HfApi()
    items = COLLECTION_ITEMS

    # 1) collection(先建,卡面要用它的真实 slug;token 无 collections 权限时降级为不阻塞)
    cid = None
    try:
        col = api.create_collection(
            'Laya NLI memory-conflict head — v4 & three-run protocol',
            namespace='Modusnsus',
            description='Delivered head (v4) plus the round-9/10 same-config runs that '
                        'established the three-run noise band (v9 / S2 / L2).',
            exists_ok=True)
        cid = getattr(col, 'slug', None) or getattr(col, 'id', None)
        print('collection:', cid, flush=True)
        for repo in items:
            try:
                api.add_collection_item(cid, item_id=repo, item_type='model', exists_ok=True)
                print('collection +', repo, flush=True)
            except Exception as e:
                print('collection item', repo, '->', e, flush=True)
    except Exception as e:
        print('collection unavailable(token 权限不足?),跳过合集,卡面改链 profile:', e, flush=True)

    # 2) v4 卡面追加归档索引
    p = hf_hub_download(V4, 'README.md')
    with io.open(p, encoding='utf-8') as f:
        card = f.read()
    if ANCHOR in card:
        print('v4 card already has the archive section; skip', flush=True)
        return
    tail = (f'\n\nAll ten repos are grouped in [the "Laya NLI memory-conflict head — rounds" collection](https://huggingface.co/collections/{cid}).'
            if cid else
            '\n\nAll repos are visible on [the Modusnsus profile](https://huggingface.co/Modusnsus).')
    section = f'\n\n{ANCHOR}\n\n' \
        'Rounds 5–10 did not pass their acceptance gates, so **v4 remains the delivered head**. ' \
        'Each round checkpoint is preserved as a separate research-archive repo (NOT delivered; ' \
        'every card lists the failed gates, weight SHA256 and provenance):\n\n' + '\n'.join(
            f'- [`Modusnsus/{n}`](https://huggingface.co/Modusnsus/{n}) — {d}' for n, d in ARCHIVE
        ) + tail
    api.upload_file(path_or_fileobj=(card + section).encode('utf-8'), path_in_repo='README.md',
                    repo_id=V4, commit_message='Add rounds 5-10 research archive index to model card')
    print('v4 card updated', flush=True)


if __name__ == '__main__':
    sys.exit(main())
