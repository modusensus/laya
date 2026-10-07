# -*- coding: utf-8 -*-
"""Publish the per-round training-corpus lineage as the Hub dataset
slow-stack/nli-conflict-train-lineage.

One jsonl per round (v3..v12) from data_local, each with row count + SHA256
computed here and rendered into the card (single source of truth).  This
completes reproducibility for the published checkpoints: the HF model
archives (v4 live, v5..v10 research archives) each trained on one of these
corpora, but Kaggle datasets only serve their LATEST version.

Hard asserts: known row counts for v4/v5/v7..v12 (kernel docstrings +
handoff records); byte-freeze lineage notes rendered from the gen-script
chain.  Idempotent; readback-verified (hash of two files re-computed after
download).
"""
import hashlib
import json
import os

from huggingface_hub import HfApi, create_repo, hf_hub_download

REPO = 'slow-stack/nli-conflict-train-lineage'
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data_local')

# (file, round label, checkpoints it trains, lineage note)
ROUNDS = [
    ('nli_conflict_train_v3.jsonl', 'v3 (round 3)',
     '— (v3 ckpt not published; MNLI-extraction source for later gens)', ''),
    ('nli_conflict_train_v4.jsonl', 'v4 (round 4)',
     '[slow-stack/laya-nli-memory-conflict](https://huggingface.co/slow-stack/laya-nli-memory-conflict) (delivered)',
     'MNLI re-rotation + v4 additions'),
    ('nli_conflict_train_v5.jsonl', 'v5 (round 5)',
     '[laya-nli-conflict-v5](https://huggingface.co/slow-stack/laya-nli-conflict-v5), '
     '[v5-ce](https://huggingface.co/slow-stack/laya-nli-conflict-v5-ce)',
     'v5 lever blocks; both arms trained on this corpus'),
    ('nli_conflict_train_v6.jsonl', 'v6 (round 6)',
     '[laya-nli-conflict-v6](https://huggingface.co/slow-stack/laya-nli-conflict-v6)',
     'data-hygiene reset: 35 acceptance sentences cleared (leak 0/35 from here)'),
    ('nli_conflict_train_v7.jsonl', 'v7 (round 7)',
     '[laya-nli-conflict-v7](https://huggingface.co/slow-stack/laya-nli-conflict-v7)',
     '+ bare-intent HC, alt-praise, metric blocks'),
    ('nli_conflict_train_v8.jsonl', 'v8 (round 8)',
     '[laya-nli-conflict-v8](https://huggingface.co/slow-stack/laya-nli-conflict-v8)',
     '+120 same-subject attribute-consistent rows (B2 dose-response)'),
    ('nli_conflict_train_v9.jsonl', 'v9 (round 9)',
     '[laya-nli-conflict-v9](https://huggingface.co/slow-stack/laya-nli-conflict-v9), '
     '[v10-s2](https://huggingface.co/slow-stack/laya-nli-conflict-v10-s2) (S-arm reuses v9 verbatim)',
     '+400 attr-add/attr-contra/waver/change rows'),
    ('nli_conflict_train_v10.jsonl', 'v10 (round 10)',
     '[v10-l1](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l1), '
     '[v10-l2](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l2)',
     '= v9 minus exactly the 40 attr_contradiction rows (carried rows byte-identical)'),
    ('nli_conflict_train_v11.jsonl', 'v11 (round 11, r1-r3)',
     '— (verdict FAIL, no delivery; Kaggle dataset v16)',
     '= v9 carried verbatim (block B stays) + 94 lever rows'),
    ('nli_conflict_train_v12.jsonl', 'v12 (round 12, r1-r3)',
     '— (runs in flight; Kaggle dataset v17)',
     '= v11 carried verbatim + 36 negation-known-conflict lever rows'),
]

KNOWN_ROWS = {'nli_conflict_train_v4.jsonl': 12000, 'nli_conflict_train_v5.jsonl': 13400,
              'nli_conflict_train_v7.jsonl': 15300, 'nli_conflict_train_v8.jsonl': 15420,
              'nli_conflict_train_v9.jsonl': 15820, 'nli_conflict_train_v10.jsonl': 15780,
              'nli_conflict_train_v11.jsonl': 15914, 'nli_conflict_train_v12.jsonl': 15950}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def upload_retry(api, path_or_fileobj, path_in_repo):
    """HF endpoint drops long uploads intermittently (10054/RemoteProtocol) —
    per-file retry with backoff; uploads are idempotent (overwrite in place)."""
    import time
    for attempt in range(1, 5):
        try:
            print(f'   uploading {path_in_repo} (attempt {attempt})', flush=True)
            api.upload_file(path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo,
                            repo_id=REPO, repo_type='dataset')
            return
        except Exception as e:  # noqa: BLE001 - network flakes have many shapes
            if attempt == 4:
                raise
            print(f'   upload failed ({type(e).__name__}); retry in 60s', flush=True)
            time.sleep(60)


def main():
    rows_meta = []
    for fname, rnd, ckpts, note in ROUNDS:
        p = os.path.join(D, fname)
        n = sum(1 for l in open(p, encoding='utf-8') if l.strip())
        if fname in KNOWN_ROWS:
            assert n == KNOWN_ROWS[fname], (fname, n, KNOWN_ROWS[fname])
        rows_meta.append({'file': fname, 'round': rnd, 'rows': n, 'sha256': sha256(p),
                          'checkpoints': ckpts, 'lineage': note})
        print(f'{fname}: {n} rows')

    table = ['| file | round | rows | sha256 | trains | lineage |', '|---|---|---|---|---|---|']
    for m in rows_meta:
        table.append(f"| `{m['file']}` | {m['round']} | {m['rows']:,} | `{m['sha256']}` "
                     f"| {m['checkpoints']} | {m['lineage']} |")
    card = CARD_TEMPLATE.replace('%%TABLE%%', '\n'.join(table))
    card_path = os.path.join(HERE, 'dataset_card_nli_conflict_lineage.md')
    with open(card_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(card)

    api = HfApi()
    create_repo(REPO, repo_type='dataset', exist_ok=True, private=False)
    upload_retry(api, card_path, 'README.md')
    for m in rows_meta:
        upload_retry(api, os.path.join(D, m['file']), m['file'])
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
    remote = {f.rfilename: f.size for f in info.siblings}
    for m in rows_meta:
        assert remote.get(m['file']) == os.path.getsize(os.path.join(D, m['file'])), m['file']
    # readback hash check on the smallest + the current corpus
    for fname in ('nli_conflict_train_v3.jsonl', 'nli_conflict_train_v12.jsonl'):
        back = hf_hub_download(REPO, fname, repo_type='dataset')
        assert sha256(back) == next(m['sha256'] for m in rows_meta if m['file'] == fname)
    print(f'OK {len(remote)} files, sizes + 2 hash readbacks verified '
          f'-> https://huggingface.co/datasets/{REPO}')


CARD_TEMPLATE = '''# Laya NLI conflict training-corpus lineage (v3-v12)

One training corpus per round of the Laya 0.3B noul memory-conflict head
([project handoffs](https://github.com/modusensus/laya/blob/main/kaggle_eval/HANDOFF_NLI_V12.md)),
with row counts and SHA256 computed at publish time. This completes the
reproducibility chain for the published checkpoints: Kaggle dataset versions
are rolling (only the latest is pullable), so the per-round corpora live here.

Corpora are deterministic outputs of the `gen_soft_conflicts_v*.py` scripts
(fixed seeds, hard leak asserts: zero field-level overlap with the 35
acceptance cases from v6 onward, val/val_soft byte-frozen). The frozen
evaluation sets live in
[slow-stack/laya-nli-conflict-eval](https://huggingface.co/datasets/slow-stack/laya-nli-conflict-eval);
the current round's corpus is also mirrored (rolling) at
[slow-stack/nli-conflict-pairs](https://huggingface.co/datasets/slow-stack/nli-conflict-pairs).

%%TABLE%%

## Notes

- `nli_conflict_train.jsonl` without a version suffix is NOT included: the
  per-round files above supersede it; v1/v2-era corpora and checkpoints are
  pre-hygiene (sentence-reuse era) and deliberately not published.
- Carried-row discipline: from v10 on, each round's corpus carries the
  previous round's rows VERBATIM (byte-identical lines) with only preregistered
  lever rows added/dropped, asserted at generation time.
- All content is synthetic (MultiNLI-derived + template-generated); no
  personal or real-user data. License: CC0-1.0.
'''


if __name__ == '__main__':
    main()
