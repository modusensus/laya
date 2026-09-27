# -*- coding: utf-8 -*-
"""温度锐化扫描:在冻结集上扫 τ,看 ECE 能否同时满足「主 val ≤0.02 且 realtest ≤0.04」。

不重训、不动权重:温度在 laya 里是推理期 cfg 值(rl_agent_config.json),
概率 = softmax(logits / τ)。已存的评测 JSON 记录的是 τ_cur 下的概率,换 τ 只需在
logit 空间重标定:  p_new = sigmoid( logit(p_old) * τ_cur / τ_new )。
(二分类 noul 下这是精确变换;锐化不改变 argmax,故 accuracy 保持不变。)

用法:
  python temperature_sweep.py --dump-val    # 跑一次 CPU 推理,落 val 1000 的逐条概率
  python temperature_sweep.py               # 扫 τ 出表 + 推荐区间
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, r'D:\laya\kaggle_eval')
from train_nli_conflict import load_items, collate  # noqa: E402

CKPT = sys.argv[2] if len(sys.argv) > 2 else r'D:\laya-kaggle-output\laya-nli-conflict-v4'
D = pathlib.Path(r'D:\laya\data_local')
VAL = D / 'nli_conflict_val.jsonl'
VAL_PROBS = D / 'val_probs_v4.json'
TAU_GRID = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4]
GATES = {'val': 0.02, 'val_soft': 0.07, 'realtest': 0.04}


def tau_cur(cfg):
    """本头 noul 用的温度下标与值(见 val_breakdown.py: temps[qtype],noul→2)。"""
    t = cfg.get('temperature', [1.0, 1.0, 1.0])
    return float(t[2] if len(t) > 2 else t[-1])


def dump_val():
    cfg = json.load(open(os.path.join(CKPT, 'rl_agent_config.json'), encoding='utf-8'))
    cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096, max_len=384, head_max_len=192)
    from laya.common import build_model  # noqa: E402
    tok = AutoTokenizer.from_pretrained(os.path.join(CKPT, 'tokenizer'))
    model = build_model(cfg, encoder_dir=os.path.join(CKPT, 'encoder'))
    model.load_state_dict(load_file(os.path.join(CKPT, 'model.safetensors')), strict=True)
    model.eval()
    items = load_items(str(VAL), tok, cfg, max_items=20000)
    temps = cfg.get('temperature', [1.2, 1.2, 1.2])
    rows_out = []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            cb = collate(items[i:i + 32], tok.pad_token_id)
            l_sub, _ = model(cb['input_ids'], cb['attention_mask'],
                             cb['marker_pos'], cb['marker_mask'], cb['qtype'])
            l_np = l_sub.float().numpy()
            for rr, it in enumerate(items[i:i + 32]):
                kk = len(it['markers'])
                zz = l_np[rr, :kk] / temps[it['qtype']]
                p = np.exp(zz - zz.max())
                p = p / p.sum()
                rows_out.append({'gold': int(it['label']), 'p_true': float(p[1]) if kk > 1 else float(p[0])})
    json.dump(rows_out, open(VAL_PROBS, 'w', encoding='utf-8'))
    n_ok = sum(1 for r in rows_out if (r['p_true'] >= 0.5) == bool(r['gold']))
    print(f'dumped {len(rows_out)} val rows -> {VAL_PROBS} | acc {n_ok}/{len(rows_out)} = {n_ok/len(rows_out):.4f}')


def rescale(p, t_old, t_new):
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-z * t_old / t_new))


def ece(triples, bins=10):
    n, hit, conf = [0] * bins, [0.0] * bins, [0.0] * bins
    for pred, gold, p in triples:
        c = p if pred else 1 - p
        b = min(int(c * bins), bins - 1)
        n[b] += 1
        hit[b] += pred == gold
        conf[b] += c
    tot = sum(n)
    return sum(n[b] / tot * abs(hit[b] / n[b] - conf[b] / n[b]) for b in range(bins) if n[b])


def load_sets():
    s = {}
    vp = json.load(open(VAL_PROBS, encoding='utf-8'))
    s['val'] = [(r['gold'], r['p_true']) for r in vp]
    vs = json.load(open(D / 'val_soft_v4.json', encoding='utf-8'))
    s['val_soft'] = [(int(r['label']), r['p_true']) for r in vs]
    rt = json.load(open(D / 'memory_conflict_realtest_v4.json', encoding='utf-8'))
    s['realtest'] = [((r['expect'] != 'false'), r['p_conflict']) for r in rt['cases'] + rt['new_cases']]
    dg = json.load(open(D / 'noul_bias_diag_v4.json', encoding='utf-8'))
    s['diag'] = [((c['expect'] != 'false'), c['p_conflict']) for c in dg['controls'] + dg['swaps']]
    return s


def main():
    sets = load_sets()
    t0 = tau_cur(json.load(open(os.path.join(CKPT, 'rl_agent_config.json'), encoding='utf-8')))
    met = json.load(open(os.path.join(CKPT, 'metrics.json'), encoding='utf-8'))
    grid = sorted(set(TAU_GRID + [round(t0, 4)]))

    def table(tau):
        out = {}
        for name, rows in sets.items():
            tri = [(p >= 0.5, bool(g), rescale(p, t0, tau)) for g, p in rows]
            acc = sum(1 for pr, g, _ in tri if pr == g) / len(tri)
            out[name] = (acc, ece(tri))
        return out

    # 自检:τ=τ_cur 时必须复现 metrics.json（accuracy 精确;ECE 差在分桶约定量级内）
    base = table(t0)
    assert abs(base['val'][0] - met['val_accuracy']) < 1e-9, (base['val'][0], met['val_accuracy'])
    assert abs(base['val'][1] - met['val_ece']) < 1.5e-3, (base['val'][1], met['val_ece'])
    print(f'自检通过:τ_cur={t0:.4f} 复现 metrics.json (val {base["val"][0]:.3f} / '
          f'ECE {base["val"][1]:.4f};kernel 报 {met["val_ece"]:.4f},差在分桶约定,'
          'τ 选取用同一函数做相对比较)')
    print()
    hdr = f'{"τ":>7} | ' + ' | '.join(f'{k:>16}' for k in GATES) + ' | 门槛'
    print(hdr)
    print('-' * len(hdr))
    ok_taus = []
    for tau in grid:
        r = table(tau)
        passed = all(r[k][1] <= GATES[k] for k in GATES)
        if passed:
            ok_taus.append(tau)
        cols = ' | '.join(f'{r[k][0]:.3f} ece {r[k][1]:.4f}' for k in GATES)
        tag = ('  ← τ_cur' if abs(tau - t0) < 1e-9 else '') + ('  ✓全过' if passed else '')
        print(f'{tau:>7.4f} | {cols} |{tag}')
    print()
    if ok_taus:
        print(f'可接受的 τ 区间: {min(ok_taus):.3f} ~ {max(ok_taus):.3f}'
              f'(含网格点 {len(ok_taus)} 个)')
    else:
        print('整个网格没有同时满足三项门槛的 τ:需要重标定门槛或重训(见交接文档)')
    print('注意:τ 必须在一个与报数集不相交的标定切片上选,否则 ECE 是自证。')


if __name__ == '__main__':
    if '--dump-val' in sys.argv:
        dump_val()
    else:
        main()
