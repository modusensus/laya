"""Laya fine-tune evaluation on Kaggle CPU: 400-case test split, temperature refit,
multilingual spot check. pandas-free; runs unattended, writes /kaggle/working/results.json.

Inputs: this kernel attaches the output of
daphnelaurent/laya-finetune-typed-decisions-2xt4-kaggle (the fine-tuned checkpoint).
"""
import glob
import json
import math
import os
import subprocess
import sys
import urllib.request

RESULTS = {"checkpoint_found": None, "n_cases": None, "errors": []}


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True)


print("== install deps ==", flush=True)
sh(sys.executable, "-m", "pip", "install", "-q", "laya", "pyarrow")

# ---------------------------------------------------------------- checkpoint
cands = [c for c in glob.glob("/kaggle/input/**/laya_finetuned_typed_decisions", recursive=True)
         if os.path.exists(os.path.join(c, "rl_agent_config.json"))
         and os.path.getsize(os.path.join(c, "model.safetensors")) > 1e8]
ft_dir = cands[0] if cands else None
RESULTS["checkpoint_found"] = bool(ft_dir)
print("ft checkpoint:", ft_dir, flush=True)

# --------------------------------------------------------------- test split
print("== fetch full test split (400 cases) ==", flush=True)
url = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"
pq_path = "/kaggle/working/test.parquet"
urllib.request.urlretrieve(url, pq_path)

import pyarrow.parquet as pq

tbl = pq.read_table(pq_path, columns=["state", "questions", "gold"])
rows = [{"state": tbl.column("state")[i].as_py(),
         "questions": json.loads(tbl.column("questions")[i].as_py()),
         "gold": json.loads(tbl.column("gold")[i].as_py())}
        for i in range(tbl.num_rows)]
RESULTS["n_cases"] = len(rows)
print("test cases:", len(rows), flush=True)

# -------------------------------------------------------------------- eval
import numpy as np
import laya


def evaluate(agent, split):
    """One pass over `split`; returns records + summary. Record keeps probs+gold for temp refit."""
    recs = []
    for ci, row in enumerate(split):
        try:
            ans = agent.predict(row["state"], row["questions"])["answers"]
        except Exception as e:  # a bad row must not kill the run
            RESULTS["errors"].append("case %d: %s" % (ci, e))
            continue
        for qid, qdef in row["questions"].items():
            if qid not in row["gold"]:
                continue
            t = qdef["type"]
            g = row["gold"][qid]
            if t == "choice":
                keys = list(qdef["criteria"].keys())
                p = [ans[qid]["probabilities"].get(k, 1e-6) for k in keys]
                gold = keys.index(str(g["label"]))
                acc = float(np.argmax(p) == gold)
            elif t == "noul":
                p = [1.0 - ans[qid]["noul"], ans[qid]["noul"]]
                gold = 1 if str(g["label"]).lower() == "true" else 0
                acc = float(np.argmax(p) == gold)
            else:  # score
                n = len(qdef.get("criteria", []))
                p = [ans[qid]["probabilities"].get(str(i), 0.0) for i in range(n)]
                s = sum(p)
                p = [x / s for x in p] if s > 0 else [1.0 / n] * n
                gold = int(g.get("label", round(g.get("score", 0.0))))
                acc = float(int(np.argmax(p)) == gold)
            recs.append({"type": t, "acc": acc, "probs": p, "gold": gold})
    by = {}
    for r in recs:
        by.setdefault(r["type"], []).append(r["acc"])
    summary = {"n": len(recs),
               "accuracy": round(float(np.mean([r["acc"] for r in recs])), 4) if recs else None}
    for t, v in sorted(by.items()):
        summary["acc_" + t] = round(float(np.mean(v)), 4)
    return recs, summary


def ece(conf, corr, bins=10):
    conf, corr = np.asarray(conf), np.asarray(corr)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            out += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return round(float(out), 4)


def fit_temp(recs, iters=80):
    """Fit one temperature per type from logged probabilities (softmax(log p / T) is
    shift-invariant, so probabilities carry the same information as logits)."""
    temps = {}
    for t in ("choice", "score", "noul"):
        sel = [r for r in recs if r["type"] == t]
        if len(sel) < 10:
            temps[t] = 1.0
            continue
        K = max(len(r["probs"]) for r in sel)
        logp = torch.full((len(sel), K), -1e4)
        gold = torch.zeros(len(sel), dtype=torch.long)
        for i, r in enumerate(sel):
            lp = torch.log(torch.tensor(r["probs"]).clamp_min(1e-9))
            logp[i, :len(lp)] = lp
            gold[i] = r["gold"]
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logp / logT.exp(), gold)
            loss.backward()
            return loss

        opt.step(closure)
        temps[t] = round(float(torch.clamp(logT.exp(), 0.2, 5.0).item()), 3)
    return temps


def apply_and_ece(recs, temps):
    conf, corr = [], []
    for r in recs:
        T = temps.get(r["type"], 1.0)
        lp = torch.log(torch.tensor(r["probs"]).clamp_min(1e-9)) / T
        p = torch.softmax(lp, -1)
        conf.append(float(p.max()))
        corr.append(r["acc"])
    return ece(conf, corr)


import torch  # noqa: E402  (after pip install above)

print("== load agents ==", flush=True)
agent_base = laya.load("convaiinnovations/laya-multilingual", device="cpu")
print("base loaded", flush=True)
recs_base, base_summary = evaluate(agent_base, rows)
print("base:", base_summary, flush=True)
RESULTS["base_400"] = base_summary

if ft_dir:
    agent_ft = laya.load(ft_dir, device="cpu")
    print("ft loaded", flush=True)
    recs_ft, ft_summary = evaluate(agent_ft, rows)
    print("ft:", ft_summary, flush=True)
    RESULTS["ft_400"] = ft_summary

    # held-out calibration: fit temps on the first half, report ECE on the second half
    half = len(recs_ft) // 2
    temps = fit_temp(recs_ft[:half])
    RESULTS["refit_temperatures"] = temps
    RESULTS["ft_ece_as_shipped"] = ece([max(r["probs"]) for r in recs_ft[half:]],
                                       [r["acc"] for r in recs_ft[half:]])
    RESULTS["ft_ece_refit_on_disjoint_half"] = apply_and_ece(recs_ft[half:], temps)
    RESULTS["base_ece_same_half"] = ece([max(r["probs"]) for r in recs_base[half:]],
                                        [r["acc"] for r in recs_base[half:]])
    del agent_ft

# --------------------------------------------------- multilingual spot check
SPOTS = [
    ("zh", "客户要求删除其个人数据。队列里有 3 个待处理账户。约束：不得触碰名单之外的客户数据。",
     {"action": {"type": "choice", "instructions": "下一步应该执行哪个操作？",
                 "criteria": {"delete_data": "删除队列内账户的个人数据",
                              "ask_human": "转人工确认后再操作",
                              "ignore": "忽略该请求"}}}),
    ("zh", "系统检测到疑似凭证泄露，多个账户出现异地登录。当前没有可逆操作的记录。",
     {"risk": {"type": "score", "instructions": "事件的风险等级？",
               "criteria": ["低：无影响", "中：需要关注", "高：需要立即响应"]}}),
    ("es", "El cliente pide eliminar sus datos personales. Hay 3 cuentas en la cola de borrado.",
     {"action": {"type": "choice", "instructions": "¿Cuál es la siguiente acción?",
                 "criteria": {"delete_data": "Eliminar los datos de las cuentas en la cola",
                              "ask_human": "Escalar a un humano",
                              "ignore": "Ignorar la solicitud"}}}),
    ("es", "Se detectó un posible filtración de credenciales con inicios de sesión desde ubicaciones desconocidas.",
     {"risk": {"type": "score", "instructions": "¿Nivel de riesgo del incidente?",
               "criteria": ["bajo: sin impacto", "medio: vigilar", "alto: responder de inmediato"]}}),
    ("ja", "顧客が個人データの削除を要求しています。キューには3件のアカウントがあります。",
     {"action": {"type": "choice", "instructions": "次に実行すべき操作は？",
                 "criteria": {"delete_data": "キュー内のアカウントのデータを削除する",
                              "ask_human": "人間に確認する",
                              "ignore": "要求を無視する"}}}),
    ("ja", "認証情報の漏洩の疑いがあり、複数のアカウントに不明な場所からのログインが発生しています。",
     {"risk": {"type": "score", "instructions": "インシデントのリスクレベルは？",
               "criteria": ["低：影響なし", "中：要監視", "高：即時対応が必要"]}}),
]
print("== multilingual spot check ==", flush=True)
spots = []
for lang, state, questions in SPOTS:
    entry = {"lang": lang}
    for name, agent in (("base", agent_base), ("ft", agent_ft if ft_dir else None)):
        if agent is None:
            continue
        a = agent.predict(state, questions)["answers"]
        qid = list(questions)[0]
        ans = a[qid]
        if ans["type"] == "choice":
            entry[name] = ans["choice"]
        else:
            entry[name] = round(ans["score"], 2)
    spots.append(entry)
    print(entry, flush=True)
RESULTS["multilingual_spot_check"] = spots

# ------------------------------------------------------------------ results
with open("/kaggle/working/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print("\n=== FINAL RESULTS ===")
print(json.dumps(RESULTS, indent=2, ensure_ascii=False))
