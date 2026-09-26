# -*- coding: utf-8 -*-
"""Per-class val accuracy of a conflict-head checkpoint, computed locally on CPU.

Reuses the training data pipeline (build_sequence + model forward) so the numbers
match the kernel-side eval; adds the class breakdown the kernel doesn't print.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, r'C:\Users\石晴\Desktop\laya\kaggle_eval')
from train_nli_conflict import load_items, collate  # noqa: E402

CKPT = sys.argv[1]
VAL = r'C:\Users\石晴\Desktop\laya\data_local\nli_conflict_val.jsonl'

with open(os.path.join(CKPT, 'rl_agent_config.json')) as f:
    cfg = json.load(f)
cfg['gradient_checkpointing'] = True
cfg['max_tokens_per_batch'] = 4096
cfg['max_len'] = 384
cfg['head_max_len'] = 192

from laya.common import build_model  # noqa: E402
tok = AutoTokenizer.from_pretrained(os.path.join(CKPT, 'tokenizer'))
model = build_model(cfg, encoder_dir=os.path.join(CKPT, 'encoder'))
model.load_state_dict(load_file(os.path.join(CKPT, 'model.safetensors')), strict=True)
model.eval()

items = load_items(VAL, tok, cfg, max_items=2000)
temps = cfg.get('temperature', [1.2, 1.2, 1.2])
QTYPES = {'noul': 2}

# recover each item's source label class from the val file order
rows = [json.loads(l) for l in open(VAL, encoding='utf-8') if l.strip()]
row_classes = []
for r in rows:
    lab = r['gold']['conflict']['label']
    row_classes.append(lab)
# load_items flattens one item per question; our rows have exactly one question each
assert len(items) == len(rows), (len(items), len(rows))

correct_by = {}
conf_by = {}
with torch.no_grad():
    for i in range(0, len(items), 32):
        cb = collate(items[i:i+32], tok.pad_token_id)
        l_sub, _ = model(cb['input_ids'], cb['attention_mask'],
                         cb['marker_pos'], cb['marker_mask'], cb['qtype'])
        l_np = l_sub.float().numpy()
        for rr, it in enumerate(items[i:i+32]):
            kk = len(it['markers'])
            zz = l_np[rr, :kk] / temps[it['qtype']]
            p = np.exp(zz - zz.max()); p = p / p.sum()
            pred = int(p.argmax())
            cls = row_classes[i + rr]
            ok = int(pred == it['label'])
            correct_by.setdefault(cls, []).append(ok)
            conf_by.setdefault(cls, []).append(float(p.max()))

total_ok = sum(sum(v) for v in correct_by.values())
total_n = sum(len(v) for v in correct_by.values())
print(f'overall: {total_ok}/{total_n} = {total_ok/total_n:.4f}')
for cls, v in correct_by.items():
    print(f'  gold {cls:13s}: {sum(v)}/{len(v)} = {sum(v)/len(v):.4f} | mean conf {np.mean(conf_by[cls]):.3f}')
