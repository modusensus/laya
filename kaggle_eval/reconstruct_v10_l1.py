# -*- coding: utf-8 -*-
"""Round-10 L1 checkpoint reconstruction (documented deviation, V10 §10.5
precedent: report first, then apply).

The L1 (ce v7) kernel output zip could not be pulled whole (CLI pulls only
the latest version; the versioned download_zip endpoint works but the flaky
link never completed a full pass).  Salvage recovered, CRC-verified via the
zip data descriptors:
  - checkpoint_latest/model.safetensors (643835524 bytes == the L2 run's
    model size; the epoch-4 checkpoint -- identical weights to the final
    save, which only re-writes the same state dict after the no-op
    calibration step)
  - metrics.json (97 bytes)
Everything else needed by laya.load is run-invariant or refittable:
  - encoder/ and tokenizer/ are written from the frozen base model and are
    byte-identical across runs (taken from the L2 pull)
  - rl_agent_config.json differs only in temperature[2] (fitted on the 400
    calib items, seed 20260927 -- refit here locally with the exact kernel
    procedure by importing the kernel's own train script functions)
  - val_probs.json is recomputed here with the same loader/collate/eval path
This script assembles the L1 checkpoint dir and writes val_probs_v10_l1.json.
"""
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

L1 = r'D:\laya-kaggle-output\laya-nli-conflict-v10-l1'
L2 = r'D:\laya-kaggle-output\laya-nli-conflict-v10-l2'
SALVAGE_MODEL = r'C:\kaggle_cfg\stage\v10_l1.zip.best'

DATA = r'D:\laya\data_local'


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from train_nli_conflict import load_items, collate, fit_one_temp
    from laya.common import build_model

    # ---- 1. assemble the checkpoint dir ----
    os.makedirs(L1, exist_ok=True)
    model_path = os.path.join(L1, 'model.safetensors')
    if not os.path.exists(model_path):
        # extract the CRC-verified checkpoint_latest/model.safetensors from
        # the retained best prefix via the salvage scanner
        sys.path.insert(0, r'C:\kaggle_cfg\stage')
        from pull_zip_salvage import salvage_scan
        tmp_out = r'C:\kaggle_cfg\stage\v10_l1_best_extract'
        if not os.path.exists(os.path.join(tmp_out, 'checkpoint_latest', 'model.safetensors')):
            assert salvage_scan(SALVAGE_MODEL, tmp_out, strict=False), 'salvage failed to recover model.safetensors'
        shutil.copyfile(os.path.join(tmp_out, 'checkpoint_latest', 'model.safetensors'),
                        model_path)
    import hashlib
    h = hashlib.sha256(open(model_path, 'rb').read()).hexdigest()
    print('L1 model.safetensors SHA256 (salvaged, CRC-verified):', h)
    for d in ('encoder', 'tokenizer'):
        dst = os.path.join(L1, d)
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(L2, d), dst)
    print('encoder/ + tokenizer/ copied from the L2 pull (run-invariant)')

    # ---- 2. refit tau exactly as the kernel does ----
    cfg = json.load(open(os.path.join(L2, 'rl_agent_config.json'), encoding='utf-8'))
    cfg['gradient_checkpointing'] = True
    cfg['max_tokens_per_batch'] = 4096
    cfg['max_len'] = 384
    cfg['head_max_len'] = 192
    tok = AutoTokenizer.from_pretrained(os.path.join(L1, 'tokenizer'))
    model = build_model(cfg, encoder_dir=os.path.join(L1, 'encoder'))
    model.load_state_dict(load_file(model_path), strict=True)
    model.eval()

    all_train = load_items(os.path.join(DATA, 'nli_conflict_train_v10.jsonl'), tok, cfg, max_items=16200)
    assert len(all_train) == 15780, len(all_train)
    order = list(range(len(all_train)))
    random.Random(20260927).shuffle(order)
    n_calib = min(400, len(all_train) // 10)
    calib_items = [all_train[i] for i in sorted(order[:n_calib])]
    print(f'calib items: {len(calib_items)}')

    preds = []
    with torch.no_grad():
        for i in range(0, len(calib_items), 32):
            cb = collate(calib_items[i:i + 32], tok.pad_token_id)
            l_sub, _ = model(cb['input_ids'], cb['attention_mask'], cb['marker_pos'],
                             cb['marker_mask'], cb['qtype'])
            l_np = l_sub.float().cpu().numpy()
            for rr, it in enumerate(calib_items[i:i + 32]):
                kk = len(it['markers'])
                preds.append((it['qtype'], l_np[rr, :kk], it['target']))
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, tt) for q_type, z, tt in preds if q_type == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    print('refitted temps:', [round(t, 6) for t in temps])
    tau_noul = temps[2]

    # ---- 3. recompute val_probs (same eval path as the kernel) ----
    import numpy as np
    all_val = load_items(os.path.join(DATA, 'nli_conflict_val.jsonl'), tok, cfg, max_items=2000)
    correct, val_rows = [], []
    with torch.no_grad():
        for i in range(0, len(all_val), 32):
            cb = collate(all_val[i:i + 32], tok.pad_token_id)
            l_sub, _ = model(cb['input_ids'], cb['attention_mask'], cb['marker_pos'],
                             cb['marker_mask'], cb['qtype'])
            l_np = l_sub.float().cpu().numpy()
            for rr, it in enumerate(all_val[i:i + 32]):
                kk = len(it['markers'])
                zz = l_np[rr, :kk] / temps[it['qtype']]
                p = np.exp(zz - zz.max())
                p = p / p.sum()
                correct.append(float(int(np.argmax(p)) == it['label']))
                val_rows.append({'gold': int(it['label']), 'p_true': float(p[1])})
    acc = float(sum(correct) / len(correct))
    err = int(sum(1 for r in val_rows if (r['p_true'] >= 0.5) != (r['gold'] == 1)))
    print(f'L1 reconstructed main val: acc {acc:.4f} err {err} n {len(val_rows)}')
    json.dump(val_rows, open(os.path.join(DATA, 'val_probs_v10_l1.json'), 'w'))

    # ---- 4. write rl_agent_config.json LAST (the orchestrator gate file) ----
    cfg['fine_tuned'] = True
    cfg['temperature'] = temps
    if 'temperature_by_options' in cfg:
        cfg.pop('temperature_by_options', None)
    with open(os.path.join(L1, 'rl_agent_config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    metrics = {'val_accuracy': round(acc, 4), 'n_val': len(val_rows), 'no_rl': True,
               'reconstructed': 'local CPU recompute from CRC-verified salvaged weights'}
    with open(os.path.join(L1, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(L1, 'val_probs.json'), 'w', encoding='utf-8') as f:
        json.dump(val_rows, f)
    print('L1 checkpoint dir assembled:', L1)


if __name__ == '__main__':
    main()
