# 第 11 轮收尾缺口清单

- 提出人：dsh（2026-10-04，首次按 `kaggle_eval/ONBOARDING.md` §5 启动清单接手核查）
- 收件人：ZCode（执行）/ Hermes（验收）
- 性质：**流程与留痕缺口**，不是判定分歧。v11 的判定结论本身经本机独立复读确认无误，`不交付、HF 不动` 的结论不变。
- 复读源：`data_local/protocol_verdict_v11.log`、`data_local/v11_orchestrate_state.json`、`data_local/leak_audit_v11.txt`

## 缺口 1｜HANDOFF 的「第 11 轮执行结果」是空壳，且缺 round-11 收尾提交

- `kaggle_eval/HANDOFF_NLI_V11.md:123-124` 仍只有占位行：
  `## 第 11 轮执行结果` / `(三跑完成后由执行者按 §1.2 逐轴填写;先回测 §4-4,再跑 r1/r2/r3)`
- `git log --all --grep='round-11'` 只命中 DRAFT / FINAL 两份任务书，**没有收尾记录提交**。
  对照第 10 轮有 `d409638 docs/record(kaggle-eval): round-10 wrap-up - verifier addendum + last kernel line + orchestration log tail`。
- 后果：按 §7 文档同步纪律，第 11 轮全部读数目前只存在于 `data_local/` 日志与 `ONBOARDING.md` §4；只读 HANDOFF 的接手者会看到空壳。
- 建议补做：①按 §1.2 逐轴填表写入 `HANDOFF_NLI_V11.md` 该小节；②补一条 `docs/record(kaggle-eval): round-11 wrap-up - ...` 提交。

### dsh 独立复读的读数（供填表核对）

| 轴 | r1 / r2 / r3 | 聚合 | 判定 |
| --- | --- | --- | --- |
| 1 主 val | 0.897 / 0.895 / 0.893（错 103 / 105 / 107，n=1000） | 中位 0.895 | **FAIL**（门槛 ≥0.896，差 1 例） |
| 2 老 20 | 20 / 20 / 20，`repeated(>=2): none` | 中位 20 | PASS |
| 3 新 10 | 10 / 9 / 10 | 三跑全 10 | **FAIL** |
| 4 否定 5 | 5 / 4 / 5 | 三跑全 5 | **FAIL** |
| 5 val_soft 错数 | 1 / 0 / 0 | 三跑全 ≤3 | PASS |
| 6 swap | — | 三跑全 = 0 | PASS |
| 7 / 8 极性 | — | 三跑全 (True, True) | PASS |
| 9 band | — | 三跑全 ≥8pp | PASS |
| 10 conformal 采用档 s1 | 36/52@0.334 / 37/55@0.322 / 42/56@0.342 | 全 ≥50% @ ≤35% | PASS |
| 11 偏置诊断 | 12/14 / 13/14 / 13/14 | ≥2/3 跑 ≥13/14 | PASS |
| 12 同构家族 mean p | 0.2169 / 0.0939 / 0.0794 | 中位 0.0939；p≥0.9 案数 0/0/0 | PASS |
| 13 卫生 | `leak_audit` 0/35 | — | PASS |
| τ(noul)（非门控，照报） | 1.0768 / 1.0870 / 1.1042 | — | — |

合计 13 轴：**10 PASS / 3 FAIL**，`PROTOCOL VERDICT: FAIL -> no delivery, HF untouched`。

## 缺口 2｜v11 评测产物未入库（历轮惯例是入库）

`git status` 有 **44 个未跟踪文件**，全在 `data_local/`：

```
conf_band_v11_r{1,2,3}.txt
conformal_v11_r{1,2,3}.json / _s1.json / _s2.json   (9 个)
conformal_alt_v11_r{1,2,3}.json
family_diag_v11_r{1,2,3}.json
indist_probe_v11_r{1,2,3}.json
label_swap_v11_r{1,2,3}.json
leak_audit_v11.txt
memory_conflict_realtest_v11_r{1,2,3}.json
noul_bias_diag_v11_r{1,2,3}.json
polarity_nli_v11_r{1,2,3}.json
protocol_backtest_v11.log
protocol_verdict_v11.log
v11_orchestrate.log
v11_orchestrate_state.json
val_probs_v11_r{1,2,3}.json
val_soft_probs_v11_r{1,2,3}.json
```

对照：`git ls-files data_local` 有 **132 个已跟踪文件**，其中 `conf_band_v9.txt`、`conf_band_v10_l1.txt`、`conformal_v10_l1.json`、`conformal_alt_v10_*.json` 等历轮同类产物**均已入库** ⇒ v11 这批是落差。

不在本项范围：语料 `data_local/**/*.jsonl` 不提交是设计如此（`.gitignore:86`），`nli_conflict_train_v11.jsonl`（15914 行 / 9794054 字节）保持不入库。

## 缺口 3｜HANDOFF §1.4 要求的 claims 交付包缺失

- `kaggle_eval/claims_v11.json` 不存在
- `kaggle_eval/verify_claims_v11.py` 不存在

§1.3 规定「推前门 = `verify_claims_v11.py` 必须 exit 0」。本轮 verdict FAIL、未推 HF，**可能是执行者有意省略而非漏做**，dsh 不下结论。请明确其一：

1. 有意省略 ⇒ 把该决定写进第 11 轮执行结果小节（或 §1.4），使接手者知道省略是决定而非遗漏；
2. 漏做 ⇒ 补齐 `claims_v11.json` + `verify_claims_v11.py`。

## 已核对无误的部分（无需处理）

- v11 逐轴判定与 `ONBOARDING.md` §4 记载一致；
- 老 20 三跑 20/20 且无任何案在 ≥2 跑 miss ⇒ 杠杆 1b 确实治好了回测暴露的两案真实弱点；
- 家族率仪器三跑全绿（mean p 中位 0.0939、p≥0.9 案数 0/0/0）⇒ v11 新增的家族轴设计有效；
- `data_local/leak_audit_v11.txt`：`nli_conflict_train_v11.jsonl: 15914 rows` / `cases touched: 0/35`；
- `D:\laya-kaggle-output\` 下 v11-r1 / v11-r2 / v11-r3 三份 ckpt 均在位（2026-10-03 13:13 / 14:38 / 16:17）。

## 附：本机历史改写提示（与上述三项无关，仅避免误解）

`1b91114` 提交信息尾部的 AI 署名（`(dsh/Hermes)`）已改写为 `764834b`，**仅改提交信息，文件内容零改动**（作者与作者日期不变）。若你本地持有旧 clone 或旧 bundle，需重新拉取 `origin/main`。
