"""校准/切片报告:主 val 之外的分语言、分类型、holdout ECE(10-bin,conf 取预测标签概率)。

用法: python calib_report.py [data_local 目录]   # 默认 D:/laya/data_local
只读现成 JSON,不跑模型。
"""
import json
import pathlib
import sys
from collections import defaultdict

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "D:/laya/data_local")
HOLDOUT = {"email", "desk_floor", "degree", "barber"}  # 只在 val_soft 出现,训练集未见


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


def slices_soft(path):
    t = defaultdict(list)
    for r in json.loads(pathlib.Path(path).read_text(encoding="utf-8")):
        row = (int(r["pred"]), int(r["label"]), r["p_true"])
        t["all"].append(row)
        t["lang:" + r["meta"]["lang"]].append(row)
        t["kind:" + r["meta"]["kind"]].append(row)
        if r["meta"]["cat"] in HOLDOUT:
            t["holdout"].append(row)
    return t


def slices_real(path):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = d if isinstance(d, list) else d.get("cases", d.get("results", []))
    return {"all": [(r["got"] != "false", r["expect"] != "false", r["p_conflict"]) for r in rows]}


def line(tag, t):
    parts = []
    for k, v in t.items():
        acc = sum(1 for p, g, _ in v if p == g)
        parts.append(f"{k} {acc}/{len(v)} acc={acc / len(v):.3f} ece={ece(v):.4f}")
    print(f"{tag}: " + " | ".join(parts))


if __name__ == "__main__":
    for tag, f in (("val_soft v2", "val_soft_v2_baseline.json"), ("val_soft v3", "val_soft_v3.json"),
                   ("val_soft v4", "val_soft_v4.json")):
        p = D / f
        line(tag, slices_soft(p)) if p.exists() else print(f"{tag}: 缺 {p}")
    for tag, f in (("realtest v3", "memory_conflict_realtest_v3.json"),
                   ("realtest v4", "memory_conflict_realtest_v4.json")):
        p = D / f
        line(tag, slices_real(p)) if p.exists() else print(f"{tag}: 缺 {p}")
