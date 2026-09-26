import json, os, sys, time, random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, QTYPES

# ---------------------------------------------------------------- data layer
def load_items(path, tok, cfg, max_items=0):
    """JSONL rows {state, questions, gold} -> training items (same shape as the notebook's)."""
    from laya.common import build_sequence
    items = []
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for row in rows:
        state = row['state']
        questions = json.loads(row['questions']) if isinstance(row['questions'], str) else row['questions']
        gold = json.loads(row['gold']) if isinstance(row['gold'], str) else row['gold']
        for qid, q in questions.items():
            if qid not in gold:
                continue
            g = gold[qid]
            t = q['type']
            crit = q.get('criteria', {})
            if t == 'noul':
                target = [g['probabilities'].get('false', 0.5), g['probabilities'].get('true', 0.5)]
            elif t == 'choice':
                keys = list(crit.keys())
                target = [g['probabilities'].get(k, 0.0) for k in keys]
            else:
                continue  # this run trains noul only
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [0.5, 0.5]
            label = target.index(max(target))
            # pass `labels` through: v2 questions use semantic label words, and the option
            # text rendered here must match what laya renders at inference time
            seq, markers = build_sequence(tok, state, {'t': t, 'ins': q['instructions'], 'crit': crit,
                                                       'labels': q.get('labels')},
                                          cfg['max_len'], cfg['head_max_len'])
            if len(markers) != len(target):
                continue
            items.append({'ids': seq, 'markers': markers, 'qtype': QTYPES[t], 'target': target, 'label': label})
        if max_items and len(items) >= max_items:
            break
    return items


def collate(items, pad_id):
    n, L = len(items), max(len(it['ids']) for it in items)
    kmax = max(len(it['markers']) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it['ids'])] = torch.tensor(it['ids'])
        att[i, :len(it['ids'])] = 1
        k = len(it['markers'])
        mpos[i, :k] = torch.tensor(it['markers'])
        mmask[i, :k] = True
        target[i, :len(it['target'])] = torch.tensor(it['target'], dtype=torch.float32)
    return {'input_ids': ids, 'attention_mask': att, 'marker_pos': mpos, 'marker_mask': mmask,
            'target': target, 'qtype': torch.tensor([it['qtype'] for it in items]),
            'label': torch.tensor([it['label'] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--train-jsonl', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--output-dir', required=True)
    args, _ = ap.parse_known_args()

    with open(os.path.join(args.model_dir, 'rl_agent_config.json')) as f:
        cfg = json.load(f)
    cfg['gradient_checkpointing'] = True
    cfg['max_tokens_per_batch'] = 4096
    # NLI pairs are short; shrink the windows for speed
    cfg['max_len'] = 384
    cfg['head_max_len'] = 192

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, 'tokenizer'))
    model = build_model(cfg, encoder_dir=os.path.join(args.model_dir, 'encoder'))
    model.load_state_dict(load_file(os.path.join(args.model_dir, 'model.safetensors')), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank == 0:
        print('Building items...', flush=True)
    all_train = load_items(args.train_jsonl, tok, cfg, max_items=8000)
    all_val = load_items(args.val_jsonl, tok, cfg, max_items=2000)
    if rank == 0:
        print(f'{len(all_train)} train items | {len(all_val)} val items', flush=True)

    CALIB_MAX = 400
    order = list(range(len(all_train)))
    random.Random(20260927).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_train) // 10)
    calib_items = [all_train[i] for i in sorted(order[:n_calib])]
    train_items = [all_train[i] for i in sorted(order[n_calib:])]
    my_items = train_items[rank::world_size]

    EPOCHS = 3
    MICRO_BATCH = 16      # short sequences -> bigger batches fit
    GRAD_ACCUM = 2
    LR_ENCODER = 2.5e-5
    LR_HEAD = 1e-4
    SIGMA_START, SIGMA_END = 0.4, 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if 'encoder.' in n]
    head_params = [p for n, p in ddp_model.named_parameters() if 'encoder.' not in n]
    optimizer = torch.optim.AdamW([
        {'params': enc_params, 'lr': LR_ENCODER},
        {'params': head_params, 'lr': LR_HEAD}], weight_decay=0.01)
    total_updates = max(1, (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    if rank == 0:
        print(f'Starting: {len(train_items)} train ({len(calib_items)} held out) | {len(my_items)}/rank | {EPOCHS} epochs', flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        random.Random(42 + epoch + rank).shuffle(my_items)
        optimizer.zero_grad(set_to_none=True)
        accum_step, n_batches, epoch_loss = 0, 0, 0.0
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate(chunk, tok.pad_token_id)
            with torch.autocast('cuda', dtype=torch.float16):
                logits, act = ddp_model(
                    batch['input_ids'].to(device), batch['attention_mask'].to(device),
                    batch['marker_pos'].to(device), batch['marker_mask'].to(device),
                    batch['qtype'].to(device))
            logits = logits.float()
            mask = batch['marker_mask'].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch['target'].to(device)

            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch['qtype'].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if rank == 0 and n_batches % 50 == 0:
                print(f'  ep{epoch+1} step {n_batches} | loss {loss.item()*GRAD_ACCUM:.4f} | reward {r.mean().item():.3f}', flush=True)
        if rank == 0:
            print(f'=== Epoch {epoch+1}/{EPOCHS} done in {time.time()-t0:.0f}s | avg loss {epoch_loss/max(1,n_batches):.4f} ===', flush=True)
            ckpt_dir = os.path.join(args.output_dir, 'checkpoint_latest')
            os.makedirs(ckpt_dir, exist_ok=True)
            save_file({kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()},
                      os.path.join(ckpt_dir, 'model.safetensors'))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, 'encoder'))
            tok.save_pretrained(os.path.join(ckpt_dir, 'tokenizer'))
        dist.barrier()

    if rank == 0:
        print('Fitting calibration temperatures...', flush=True)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(calib_items), 32):
                cb = collate(calib_items[i:i+32], tok.pad_token_id)
                with torch.autocast('cuda', dtype=torch.float16):
                    l_sub, _ = model(cb['input_ids'].to(device), cb['attention_mask'].to(device),
                                     cb['marker_pos'].to(device), cb['marker_mask'].to(device),
                                     cb['qtype'].to(device))
                l_np = l_sub.float().cpu().numpy()
                for rr, it in enumerate(calib_items[i:i+32]):
                    kk = len(it['markers'])
                    preds.append((it['qtype'], l_np[rr, :kk], it['target']))
        temps = [1.2, 1.2, 1.2]
        for qt in range(3):
            sel = [(z, tt) for q_type, z, tt in preds if q_type == qt]
            if sel:
                temps[qt] = fit_one_temp(sel)
        print('temps:', [round(t, 3) for t in temps], flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        save_file({kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()},
                  os.path.join(args.output_dir, 'model.safetensors'))
        model.encoder.config.save_pretrained(os.path.join(args.output_dir, 'encoder'))
        tok.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))
        cfg['fine_tuned'] = True
        cfg['temperature'] = temps
        cfg.pop('temperature_by_options', None)
        with open(os.path.join(args.output_dir, 'rl_agent_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        # eval metrics on val items
        import numpy as np
        correct, confs = [], []
        with torch.no_grad():
            for i in range(0, len(all_val), 32):
                cb = collate(all_val[i:i+32], tok.pad_token_id)
                with torch.autocast('cuda', dtype=torch.float16):
                    l_sub, _ = model(cb['input_ids'].to(device), cb['attention_mask'].to(device),
                                     cb['marker_pos'].to(device), cb['marker_mask'].to(device),
                                     cb['qtype'].to(device))
                l_np = l_sub.float().cpu().numpy()
                for rr, it in enumerate(all_val[i:i+32]):
                    kk = len(it['markers'])
                    zz = l_np[rr, :kk] / temps[it['qtype']]
                    p = np.exp(zz - zz.max()); p = p / p.sum()
                    correct.append(float(int(np.argmax(p)) == it['label']))
                    confs.append(float(p.max()))
        acc = float(sum(correct) / len(correct))
        # ECE 10 bins
        ece = 0.0
        confs_a = np.array(confs); corr_a = np.array(correct)
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            m = (confs_a > lo) & (confs_a <= hi)
            if m.sum():
                ece += m.mean() * abs(confs_a[m].mean() - corr_a[m].mean())
        metrics = {'val_accuracy': round(acc, 4), 'val_ece': round(float(ece), 4), 'n_val': len(correct)}
        print('METRICS:', json.dumps(metrics), flush=True)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
