"""Fine-tune Laya on one device: CPU, or any single CUDA/MPS/XPU GPU.

Same RLCD recipe as notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
(proper-scoring-rule reward over noisy-logit samples + soft cross-entropy, then
per-type temperature calibration), minus DDP and Kaggle plumbing.

Two ways to run:

  python train_local.py --smoke
      Tiny random encoder + synthetic items: trains, fits temperatures, saves a
      checkpoint, reloads it with laya.load and predicts. Verifies the whole
      loop on any machine in under a minute; no network, no GPU.

  python train_local.py --model convaiinnovations/laya-multilingual \
                        --data my_decisions.jsonl \
                        --freeze-encoder --epochs 1 --out out/laya-ft
      Real run. `--data` lines are {"state": ..., "questions": {...}, "gold": {...}}
      in the LocalLLaMA/typed-decisions format (or pass --dataset to pull that
      dataset from the Hub). --freeze-encoder detaches the encoder output so
      only the decision head trains: the only mode that finishes on CPU. On a
      CUDA GPU drop the flag for the full recipe.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laya.common import QTYPES, build_model, build_sequence, proper_reward  # noqa: E402

CALIB_SEED = 20260922  # same seed as the notebook: stable held-out calibration slice
SIGMA_START, SIGMA_END = 0.4, 0.1
GROUP_SIZE = 4
WEIGHT_DECAY = 0.01


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def fit_temperatures(model, calib_items, tok, device, log=print):
    """Post-training per-type temperature on the held-out calibration slice."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(calib_items), 16):
            cb = collate(calib_items[i : i + 16], tok.pad_token_id)
            with (torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()):
                l_sub, _ = model(
                    cb["input_ids"].to(device), cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                    cb["qtype"].to(device),
                )
            l_np = l_sub.float().cpu().numpy()
            for r, it in enumerate(calib_items[i : i + 16]):
                k = len(it["markers"])
                preds.append((it["qtype"], l_np[r, :k], it["target"]))
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    log("Fitted calibration temperatures (choice, score, noul): %s" % [round(t, 3) for t in temps])
    return temps


def train(model, items, tok, device, *, epochs, micro_batch, grad_accum,
          lr_encoder, lr_head, freeze_encoder, log=print):
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    groups = ([{"params": head_params, "lr": lr_head}] if freeze_encoder else
              [{"params": enc_params, "lr": lr_encoder}, {"params": head_params, "lr": lr_head}])
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    updates = max(1, (len(items) // (micro_batch * grad_accum)) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else nullcontext

    t0 = time.time()
    for epoch in range(epochs):
        random.Random(42 + epoch).shuffle(items)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
        optimizer.zero_grad(set_to_none=True)
        accum_step, epoch_loss, n_batches = 0, 0.0, 0
        for b_idx in range(0, len(items), micro_batch):
            chunk = items[b_idx : b_idx + micro_batch]
            if not chunk:
                continue
            batch = collate(chunk, tok.pad_token_id)
            with autocast():
                logits, act = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device), detach_encoder=freeze_encoder,
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # Noisy-logit policy gradient over GROUP_SIZE zero-mean noise samples.
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device),
                                  mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            # No DDP here, so the notebook's 0.0*act.sum() keep-alive term is dead weight.
            loss = (loss_rl + 1.0 * loss_ce) / grad_accum

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            if n_batches % 50 == 0:
                log("  epoch %d | step %d | loss %.4f | reward %.3f"
                    % (epoch + 1, n_batches, loss.item() * grad_accum, r.mean().item()))
        log("epoch %d/%d done in %.1fs | avg loss %.4f"
            % (epoch + 1, epochs, time.time() - t0, epoch_loss / max(1, n_batches)))
    return model


def save_checkpoint(model, tok, cfg, out_dir, temps, half):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: (v.half() if half else v).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg = dict(cfg)
    cfg["fine_tuned"] = True
    cfg["temperature"] = temps
    # This fit is per type; inherited bucket overrides would hide the new values.
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def build_training_item(state, q, gold_q, tok, cfg):
    """One (sequence, soft target) pair; mirrors the notebook's data builder."""
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    else:  # score
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    # len(markers) == option count; a target/markers mismatch means truncation ate an option.
    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                  cfg["max_len"], cfg["head_max_len"])
    if len(markers) != len(target):
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": label}


def rows_from_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                yield json.loads(row["state"]) if isinstance(row["state"], str) and row["state"].startswith(("{", "[")) else row["state"], \
                    json.loads(row["questions"]), json.loads(row["gold"])


def rows_from_dataset(dataset_id):
    from datasets import load_dataset

    for row in load_dataset(dataset_id, "all", split="train"):
        yield json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])


def items_from_rows(rows, tok, cfg, max_items, log=print):
    items = []
    for state, questions, gold in rows:
        for qid, q in questions.items():
            if qid in gold:
                it = build_training_item(state, q, gold[qid], tok, cfg)
                if it:
                    items.append(it)
        if max_items and len(items) >= max_items:
            break
    log("Built %d training items" % len(items))
    return items[:max_items] if max_items else items


# ----------------------------------------------------------------- smoke mode

def make_smoke_specs():
    """300 synthetic decisions across the three question types; 'hello' in the
    state means greet/true/positive with a soft (not one-hot) target."""
    specs = []
    for i in range(100):
        hello = i % 2 == 0
        state = "hello there" if hello else "goodbye now"
        margin = 0.85 if i % 4 == 0 else 0.7  # vary the soft targets a little
        specs.append((state, {"t": "noul", "ins": "Does the state greet?", "crit": {}},
                      [1 - margin, margin] if hello else [margin, 1 - margin]))
        specs.append((state, {"t": "choice", "ins": "Which action fits?",
                              "crit": {"greet": "say a greeting", "leave": "say a goodbye"}},
                      [margin, 1 - margin] if hello else [1 - margin, margin]))
        specs.append((state, {"t": "score", "ins": "How positive is the state?",
                              "crit": ["negative", "neutral", "positive"]},
                      [1 - margin, 0.15, margin] if hello else [margin, 0.15, 1 - margin]))
    return specs


def smoke(args):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    from laya.common import DecisionModel, render_options

    specs = make_smoke_specs()
    words = set()
    for state, q, _ in specs:
        words |= set(state.split())
        words |= set((q["t"] + " question: " + q["ins"]).split())
        for o in render_options(q):
            words |= set(o.split())
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {w: i for i, w in enumerate(specials + sorted(words - set(specials)))}
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    tok.backend_tokenizer.pre_tokenizer = Whitespace()

    cfg = {"encoder": "unused/offline", "head_layers": 2, "act_costs": {"act": 0},
           "max_len": 64, "head_max_len": 32}
    items = []
    for state, q, target in specs:
        seq, markers = build_sequence(tok, state, {"t": q["t"], "ins": q["ins"], "crit": q["crit"]},
                                      cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                      "target": target, "label": target.index(max(target))})

    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64)
    model = DecisionModel(BertModel(config), head_layers=2)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True

    device = resolve_device(args.device)
    print("smoke: %d items on %s" % (len(items), device))
    train(model, items, tok, device, epochs=args.epochs, micro_batch=8, grad_accum=1,
          lr_encoder=2.5e-5, lr_head=1e-4, freeze_encoder=False)

    # Hold out the same fixed slice as the real run so fit_temperatures sees unseen items.
    order = list(range(len(items)))
    random.Random(CALIB_SEED).shuffle(order)
    calib_items = [items[i] for i in sorted(order[: len(items) // 10])]
    temps = fit_temperatures(model, calib_items, tok, device)

    out = args.out or os.path.join(tempfile.mkdtemp(prefix="laya_smoke_"), "ckpt")
    save_checkpoint(model, tok, cfg, out, temps, half=device.type == "cuda")
    print("smoke: checkpoint saved to %s" % out)

    # Reload through the public loader and predict: the end-to-end contract.
    import laya

    agent = laya.load(out, device="cpu")
    res = agent.predict("hello there", {"greeting": {"type": "noul",
                                                     "instructions": "Does the state greet?"}})
    p = res["answers"]["greeting"]["noul"]
    print("smoke: laya.load predict noul(true) = %.4f (want > 0.5)" % p)
    assert p > 0.5, "fine-tuned checkpoint did not learn the smoke task"
    with open(os.path.join(out, "rl_agent_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert "temperature_by_options" not in saved and len(saved["temperature"]) == 3
    print("SMOKE OK")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="tiny model + synthetic data, end-to-end self-check")
    ap.add_argument("--model", default="convaiinnovations/laya-multilingual", help="base checkpoint (Hub id or local dir)")
    ap.add_argument("--dataset", default=None, help="HF dataset id in the typed-decisions format")
    ap.add_argument("--data", default=None, help="local .jsonl with state/questions/gold lines")
    ap.add_argument("--out", default=None, help="output checkpoint directory")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=0, help="cap on training items (0 = all)")
    ap.add_argument("--freeze-encoder", action="store_true", help="train the decision head only (CPU-viable)")
    ap.add_argument("--micro-batch", type=int, default=8, help="sequences per forward pass")
    ap.add_argument("--max-len", type=int, default=None, help="override cfg max_len (typed-decisions notebook: 1024)")
    ap.add_argument("--head-max-len", type=int, default=None, help="override cfg head_max_len (typed-decisions notebook: 256)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps | xpu")
    args = ap.parse_args()

    if args.smoke:
        smoke(args)
        return
    if not args.dataset and not args.data:
        ap.error("provide --data <file.jsonl> or --dataset <hf-id> (or run --smoke)")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from laya.agent import _fix_tokenizer_config

    torch.manual_seed(0)
    random.seed(0)

    model_dir = args.model
    if not os.path.exists(model_dir):
        model_dir = snapshot_download(model_dir, allow_patterns=[
            "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
        _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if args.max_len:
        cfg["max_len"] = args.max_len
    if args.head_max_len:
        cfg["head_max_len"] = args.head_max_len

    device = resolve_device(args.device)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    print("Loaded %s on %s" % (args.model, device))

    rows = rows_from_jsonl(args.data) if args.data else rows_from_dataset(args.dataset)
    items = items_from_rows(rows, tok, cfg, args.max_items)

    order = list(range(len(items)))
    random.Random(CALIB_SEED).shuffle(order)
    calib_items = [items[i] for i in sorted(order[: min(400, len(items) // 10)])]
    train_items = [items[i] for i in sorted(order[len(calib_items):])]
    print("%d train items (%d held out for calibration)" % (len(train_items), len(calib_items)))

    train(model, train_items, tok, device, epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=4,
          lr_encoder=2.5e-5, lr_head=1e-4, freeze_encoder=args.freeze_encoder)
    temps = fit_temperatures(model, calib_items, tok, device)

    out = args.out or "laya_finetuned"
    save_checkpoint(model, tok, cfg, out, temps, half=device.type == "cuda")
    print("Saved fine-tuned checkpoint to %s" % out)
    print("Evaluate on held-out data before relying on it: "
          "laya.load(%r) then agent.predict(state, questions)." % out)


if __name__ == "__main__":
    main()
