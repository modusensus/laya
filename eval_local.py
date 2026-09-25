"""Evaluate a Laya checkpoint against a typed-decisions-format JSONL or the HF test split.

  python eval_local.py --ckpt out/laya-ft --data my_decisions.jsonl
  python eval_local.py --ckpt out/laya-ft --dataset LocalLLaMA/typed-decisions --split test

Prints accuracy (argmax vs gold label), soft accuracy, Brier, ECE and score MAE,
plus a confusion-style breakdown per question type. Mirrors the metrics the
Kaggle notebook reports; pandas-free.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laya.common import ece_score  # noqa: E402


def iter_rows(args):
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    state = r["state"]
                    yield (json.loads(state) if isinstance(state, str) and state.startswith(("{", "[")) else state,
                           json.loads(r["questions"]), json.loads(r["gold"]))
    else:
        from datasets import load_dataset

        for row in load_dataset(args.dataset, "all", split=args.split):
            yield json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint directory (fine-tuned or hub id)")
    ap.add_argument("--dataset", default=None, help="HF dataset id in the typed-decisions format")
    ap.add_argument("--data", default=None, help="local .jsonl with state/questions/gold lines")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import laya

    agent = laya.load(args.ckpt, device=args.device)

    accs, confs, corrects = [], [], []
    soft, brier, tv = [], [], []
    score_mae, within_one = [], []
    by_type = {}
    n_cases = 0

    for state, questions, gold in iter_rows(args):
        n_cases += 1
        res = agent.predict(state, questions)["answers"]
        for qid, qdef in questions.items():
            if qid not in gold:
                continue
            p_ans, g_ans, t = res[qid], gold[qid], qdef["type"]
            if t == "choice":
                keys = list(qdef["criteria"].keys())
                is_corr = float(p_ans["choice"] == str(g_ans["label"]))
                p = np.array([p_ans["probabilities"].get(k, 1e-6) for k in keys])
                g = np.array([g_ans["probabilities"].get(k, 1e-6) for k in keys])
                p, g = p / p.sum(), g / g.sum()
            elif t == "noul":
                gold_label = str(g_ans["label"]).lower()
                is_corr = float(("true" if p_ans["noul"] >= 0.5 else "false") == gold_label)
                p = np.array([1.0 - p_ans["noul"], p_ans["noul"]])
                g = np.array([1.0 - g_ans.get("probabilities", {}).get("true", 0.5),
                              g_ans.get("probabilities", {}).get("true", 0.5)])
            else:  # score
                n_levels = len(qdef.get("criteria", []))
                p = np.array([p_ans["probabilities"].get(str(i), 0.0) for i in range(n_levels)])
                g = np.array([g_ans["probabilities"].get(str(i), 0.0) for i in range(n_levels)])
                if p.sum() > 0:
                    p = p / p.sum()
                p_lvl = int(np.argmax(p)) if p.sum() > 0 else int(round(p_ans["score"]))
                g_lvl = int(g_ans.get("label", int(round(g_ans.get("score", 0.0)))))
                is_corr = float(p_lvl == g_lvl)
                score_mae.append(abs(p_ans["score"] - g_ans.get("score", 0.0)))
                within_one.append(float(abs(p_ans["score"] - g_ans.get("score", 0.0)) <= 1.0))
            accs.append(is_corr)
            corrects.append(is_corr)
            confs.append(float(max(p.max(), 1.0 - p.max())) if t == "noul" else float(p.max()))
            soft.append(float((p * g).sum()))
            brier.append(float(((p - g) ** 2).sum()))
            tv.append(float(0.5 * np.abs(p - g).sum()))
            by_type.setdefault(t, []).append(is_corr)

    def fmt(v):
        return "%.4f" % v if isinstance(v, float) else str(v)

    print("cases: %d | decisions: %d" % (n_cases, len(accs)))
    for t, vals in sorted(by_type.items()):
        print("  %-6s acc %.4f (n=%d)" % (t, float(np.mean(vals)), len(vals)))
    summary = {
        "accuracy": round(float(np.mean(accs)), 4) if accs else None,
        "soft_acc": round(float(np.mean(soft)), 4) if soft else None,
        "brier": round(float(np.mean(brier)), 4) if brier else None,
        "ece": round(float(ece_score(np.array(confs), np.array(corrects))), 4) if confs else None,
        "score_mae": round(float(np.mean(score_mae)), 4) if score_mae else None,
        "within_1_level": round(float(np.mean(within_one)), 4) if within_one else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
