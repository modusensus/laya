# Laya NLI 记忆冲突检测头 — v4 训练任务交接（第 3 份交接文档）

> 命名说明:文档名按「交接轮次」编号。`HANDOFF_NLI_V2.md` 是第 2 轮（产出 v2/v3 两个头）,
> 本文件是第 3 轮,**要产出的是 v4 头**。别被文件名和产物版本号错位绕进去。

## 任务一句话

v3 头已交付且全门槛通过（真实 case 19/20、val 0.902、偏置诊断 13/14 PASS）,
但仍有「意图违反硬约束时漏报」这一已知代价,且从未验证过模型是否在读标签字面。
v4 = **放大稀有类 + 新增硬约束类 + 标签面改写增广 + 补两轴新验收**,目标:真实 case 20/20、新 case ≥8/10、标签面交换不掉分。

## 必读前置（不重复,自己去看）

| 文件/节 | 为什么必读 |
|---|---|
| `kaggle_eval/HANDOFF_NLI_V2.md` 全文 | 环境、凭据、Kaggle 三坑、实测 case 原样 20 条、资产路径全在里面 |
| ├ 「v3 数据工位」节 | 合成数据的四类构成、holdout 定义、接手训练前的 5 条提醒 |
| ├ 「v3 训练结果」节 | v3 基线数字、已知代价、Kaggle 资产版本对照 |
| └ 「v4 待办」节 | 本任务的来源;本文件是它的可执行化版本,冲突时以本文件为准 |
| `kaggle_eval/gen_soft_conflicts_v3.py` | v4 数据生成器照它改（确定性 seed、四类句式、中英双语） |
| `kaggle_eval/calib_report.py` | 分切片 ECE 报告,本任务必跑 |
| `data_local/nli_conflict_val_soft.jsonl` + `.meta.json` | 软冲突验证集（300 条,20 类,4 类 holdout） |

## 起点状态（v3 实测,别重算）

- 主 val 1000（冻结）:acc 0.902 / ECE 0.0146
- 偏置诊断 14 例:13/14 PASS（唯一错=对照D2 其他坚果无关,误报 0.884）;A1 地点变更两方向 0.9217
- 真实 case 老 20 条:19/20（唯一 miss=饮食禁忌松动,期望 true 得 0.077）
- val_soft 300:295/300;分 kind = true 0.990 / compat 1.000 / consid 0.938 / unrelated 1.000;holdout 4 类 95/100
- 分切片 ECE（`calib_report.py` 实测）:val_soft 全体 0.0625、holdout 0.0310、realtest 0.0309;分语言 zh 0.0569 / en 0.0782
- 分语言 acc:zh 216/221 = 97.7%、en 79/79 = 100%
- 置信天花板:所有真冲突正例的 `p_true` 落在 0.92 附近（max 0.925,无一条 > 0.95）

## v4 改动清单

### 改动 1（数据,必做）:合成块 ×1.8 + 新增第 5 类「硬约束违背」

照着 `gen_soft_conflicts_v3.py` 写 `gen_soft_conflicts_v4.py`,`SEED` 换新值（建议 20260930）。

四类整体 ×1.8（总量 3000 → 5400）,块内维持 冲突:非冲突 = 1:2:

| 类 | v3 | v4 | kind |
|---|---|---|---|
| 软冲突（状态更新） | 1000 | 1800 | true |
| 兼容细节 | 800 | 1440 | false / compat |
| 变更意图（在考虑） | 600 | 1080 | false / consid |
| 无关提及 | 600 | 1080 | false / unrelated |
| **硬约束违背（新增）** | — | **300 正 + 300 对照** | true / hardconstraint |

**硬约束类是本轮必须修的那条 miss**,例子:

- `known: 用户对花生过敏` / `new: 用户想试试花生酱饼干` → **true**（意图违反硬约束）
- `known: 用户不吃辣` / `new: 用户今天想尝试一下微辣` → **true**（= v3 漏的那条原形）
- `known: 用户对花生过敏` / `new: 用户买了不含花生的小饼干` → **false**（对照组!否则模型会学成
  「凡过敏+食物=冲突」,把漏报换成误报,等于没修）

每类必须中英双语都有。**对照行数量至少与正例 1:1**,这是本类能不能真修好的关键。

训练总量 = 6000 MNLI（原样,不动） + 5400 四类合成 + 600 硬约束类（300 正 + 300 对照）
= **12000 行**,全局 矛盾:非矛盾 ≈ 1:1.93（≈1:2）。

### 改动 2（数据,必做）:标签面改写增广

同一个决策,训练行里轮换 label 的**取值措辞与键名**,至少 4 套,按行随机:

| 套 | labels |
|---|---|
| 1 | `{"false": "兼容", "true": "冲突"}`（现状） |
| 2 | `{"false": "一致或无关", "true": "矛盾"}` |
| 3 | `{"false": "不冲突", "true": "需更新旧记忆"}` |
| 4 | `{"false": "keep", "true": "supersede"}` |

`instructions` 里的判据文字同时轮换（4 套同义说法,**与 labels 成套配对,不许错配**）——
目的就是让模型无法靠某个词面猜答案。依据见 `HANDOFF_NLI_V2.md`「v4 待办」第 1 节
（Hammer 的 function masking 同构做法）,治的是 #156 同族的表面依赖。

**顺带（一行改动,强烈建议）**:train JSONL 每行带 `meta: {cat, kind, lang}`。
现在 train 行只有 `state/questions/gold`,导致事后切不出 holdout/in-dist 子集,复盘只能靠生成器回溯。

### 改动 3（评测,必做）:标签面交换验收（新轴）

新脚本 `kaggle_eval/label_swap_check.py`:把 labels 两个取值互换（`冲突` ↔ `兼容`）后,
重跑**真实 case 老 20 条 + 偏置诊断 14 例**,输出标准版 vs 交换版逐条对比。

判据:**两版 accuracy 差 ≤1 条**（各自方向都算）。差得多 = 模型还在读标签字面,增广没吃进去,
不许报「修好了」。现有诊断的「输入交换」和正/负对照覆盖不到这一轴。

### 改动 4（评测,必做）:分切片校准报告

跑 `python calib_report.py`,`val_soft_v4.json` / `memory_conflict_realtest_v4.json` 出来后
在同一个表里出 v4 列。门槛见下节。

### 改动 5（真实 case,必做）:追加 10 条与合成模板无关的新 case

- 老 20 条**一行都不许改**（跨版本可比),新 case 写进 `data_local/memory_conflict_realtest_v4.json`
- 题材避开合成块的 20 个属性类:会议时间、订阅服务、语言学习、通勤方式、宠物粮食、快递地址、
  音乐口味、室友、笔记本型号、运动伤病史
- 其中含 2-3 条硬约束类、1-2 条否定句

### 改动 6（可选,加分）:#238 CE-only 对照

把 loss 改成 `loss = loss_ce / GRAD_ACCUM` 一行,再跑一个候选（命名 `v4-ce`）,
同表对比。若时间紧可跳过,跳过要在汇报里写明。

## 硬门槛（不许降）

| 项 | v3 基线 | v4 门槛 |
|---|---|---|
| 主 val 1000（冻结文件,一行不许改） | 0.902 / ECE 0.0146 | acc ≥ 0.90,ECE ≤ 0.02 |
| 偏置诊断 14 例 | 13/14 PASS | ≥13/14 PASS,**且 A1 两方向 ≥0.85** |
| 真实 case 老 20 条 | 19/20 | **≥19/20**（目标 20/20,饮食禁忌松动必须翻正） |
| 真实 case 新 10 条（无模板） | — | ≥8/10 |
| **标签面交换（新）** | — | 与标准版差 ≤1 条（双向） |
| val_soft 300 | 295/300 | ≥295/300,且 holdout 4 类 ≥95% |
| 分切片 ECE | val_soft 0.0625 / realtest 0.0309 | val_soft ≤0.07,realtest ≤0.04 |
| 分语言 acc | zh 0.977 / en 1.000 | 两者均 ≥0.97 |
| 否定句 5 条 | 4/5 | 报告即可,不作门槛（上游 #377 模型层限制） |

任何一项没过:如实报告 + 逐条列出错例,不许调阈值凑数、不许改冻结集。

## 汇报格式（5 行数字 + 交付物,照抄这个结构）

```
v4 验收汇报(checkpoint:D:\laya-kaggle-output\laya-nli-conflict-v4)
主 val <acc> / ECE <x>,偏置诊断 <n>/14 <PASS|FAIL>,A1 两方向 <p1>/<p2>
真实 case 老 20 条 <n>/20,新 10 条 <n>/10,否定句 <n>/5
val_soft <n>/300（holdout <n>/100,consid <n>/65）,ECE <x>
标签面交换:标准版 <n>/20 vs 交换版 <n>/20,诊断 <n>/14 vs <n>/14
```

后面附:已知代价与残留、交付物路径、Kaggle 资产版本号（kernel v? × 数据集 v?）、
kernel 页/数据集页描述是否已同步。

### 交付物清单

- checkpoint:`D:\laya-kaggle-output\laya-nli-conflict-v4\`（**同时把 SHA256 写进交接**）
- `data_local/`:新 train/val_soft JSONL、`memory_conflict_realtest_v4.json`、`label_swap_v4.json`、
  `noul_bias_diag_v4.json`、`val_soft_v4.json`、calib_report 输出
- 生成器与评测脚本提交进 `kaggle_eval/`（`gen_soft_conflicts_v4.py`、`label_swap_check.py`）
- 本文件末尾补「v4 执行结果」一节（照 `HANDOFF_NLI_V2.md` 的写法:对照表 + 未过项如实写 + 上游资产版本）
- 盘上最多留一个最强备选（命名 `-v4-alt`）,**被否决的候选一律删掉**,别留一堆

## 执行注意（只列会烧时间的）

1. Python 一律 `D:\Miniconda\envs\laya-ft\python.exe`。
2. Kaggle CLI 在中文用户名路径下挂:tmp 指 ASCII 路径（`C:/kaggle_cfg/tmp`）,
   token 以 `KAGGLE_API_TOKEN` env 传入,认证走 Python API 而非 CLI 入口。
3. `kernel 内嵌脚本是 b64`:改了 `train_nli_conflict.py` 必须重新 b64 替换 `nli_kernel.py` 里的内嵌块,
   否则跑的还是旧代码。
4. **`max_items` 必须 ≥ 12000 + 余量（建议设 13000）**:v3 已因 8000 上限静默截断过一次;
   顺手在脚本里 `assert len(rows) == 预期行数` 并打印 token 长度 p99（防 train/serve 长度错配）。
5. 数据集挂载永远是最新版:上传后确认数据集版本号,并在汇报里写清 **kernel v? × 数据集 v?**。
6. torchrun 用命名参数;409 = 旧版本还在 running/saving,等或删。
7. `kernels output` 只拉最新版 → 交付 checkpoint 一律以本地副本为准,并记 SHA256。
8. token 一律不入文件、不入日志,用完提醒吊销。

## 明确不要做

- 不改主 val 1000、不改老 20 条真实 case（跨版本可比性靠它们）
- 不用调阈值/加后处理规则去凑真实 case——要改数据,不是改代码
- 不在还没跑「偏置诊断 + 标签面交换」两项的情况下报「修好了」
- 不动 `laya/` 核心库;不改成多分类;不换基座/架构（本轮只动数据与评测）
- 硬约束类不要只造正例:对照行缺失 = 把漏报换成误报,白干

## v4 执行结果(2026-09-27)

kernel v8 × 数据集 v8(12k train = 6000 MNLI 原样 + 5400 软冲突块 ×1.8 + 600 硬约束类;
合成行 4 套标签面轮换;每行 meta;max_items 13000 + expected-train 断言 + token p99=145/max 309)
→ **v4 头,交付**:

| 门槛项 | v4 门槛 | v4 实测 | 判定 |
|---|---|---|---|
| 主 val 1000(冻结) | acc ≥0.90, ECE ≤0.02 | 0.901 / 0.0192 | ✓ |
| 偏置诊断 14 例 | ≥13/14 且 A1 两方向 ≥0.85 | 13/14 PASS,A1 0.9225/0.9225 | ✓(唯一 miss=对照D2,与 v3 同一条固有歧义例) |
| 真实 case 老 20 条 | ≥19/20,饮食禁忌松动必须翻正 | **20/20**,该条 0.9245 翻正 | ✓ |
| 真实 case 新 10 条 | ≥8/10 | **10/10**(含 2 条硬约束均 0.92+) | ✓ |
| 标签面交换(新轴) | 差 ≤1 条(双向) | **real 20 vs 20、diag 13 vs 13,差 0** | ✓ |
| val_soft 300 | ≥295/300,holdout ≥95% | 297/300,holdout 97/100 | ✓ |
| 分切片 ECE | val_soft ≤0.07,realtest ≤0.04 | 0.0703 / 0.0765 | **✗(见下)** |
| 分语言 acc | 两者 ≥0.97 | zh 0.986 / en 1.000 | ✓ |
| 否定句 5 条 | 报告 | 4/5(本轮 miss=养宠物反转,v3 是无车与通勤) | 报告 |

**未过项如实说明(ECE 两项,含事后实测)**:realtest ECE 0.0765(v3 0.0309)与 val_soft 0.0703
(v3 0.0625)不是校准退化,而是**准确率满分撞上 0.92 置信天花板**的结构性结果——老 20 条全部
落进同一个 bin(0.9-1.0),acc=1.000 / mean conf=0.923,ECE 恒等于 1−conf=0.077;v3 的 19/20
时是 |0.95−0.92|=0.03。即 acc=1.0 时该门槛等于 |1−conf|,不含模型质量信息。

屋顶的来源已定位:**温度**。本头 noul 用 `rl_agent_config.json` 的 `temperature[2]=1.2176`
(>1 = 软化),`laya.load` 推理期除以它,原始 logits 的高置信因此被压到 0.92。温度是推理期
cfg 值、由 `train_local.py:fit_temperatures()` 在 held-out 切片上后训练拟合——**改它不需要
重训**(上段原写「调温度=改配方+重训」不准确,此处更正)。

据此扫了 τ(`kaggle_eval/temperature_sweep.py` 在 logit 空间精确重标定,accuracy 不变):

| τ | 主 val ECE | val_soft ECE | realtest ECE |
|---|---|---|---|
| 0.60 | 0.0901 | 0.0042 | **0.0064** |
| 0.95 | 0.0556 | 0.0363 | **0.0396** |
| 1.2176(现状) | **0.0187** | 0.0703 | 0.0767 |
| 1.40 | **0.0134** | 0.0987 | 0.1030 |

**结论:没有任何单一标量 τ 能同时满足三项门槛**——主 val 想要 τ≳1.2,realtest/val_soft 想要
τ≲0.8,三个集对置信尺度的最优解差约 2×。所以「重标定温度」不是一行修复,而是**按部署分布选 τ**
的决策:线上流量若像真实 case(短、中文、记忆对话)就用 τ≈0.7-0.8(置信上限同时升到 ~0.97),
若像 MNLI val 则维持 1.2。锐化不改 argmax,故本轮 20/20、297/300 等准确性结论与 τ 无关。

门槛改法(两条都建议):① ECE 门槛换成校准偏差 `|acc − mean_conf|` 或按 acc 分档——直接放水
到 0.08/0.09 会放过「校准坏但准确率 0.95」的模型;② 分集各报本集最优 τ 下的 ECE,并注明选 τ
的切片必须与报数集不相交,否则 ECE 是自证。

**改动 6(CE-only 对照)跳过**:主轮全部分类门槛通过,CE-only 是方法论对照而非门槛项,
跳过以省一轮训练+全套验收;一行 diff(`loss = loss_ce / GRAD_ACCUM`)随时可补跑。

**交付物**:
- checkpoint `D:\laya-kaggle-output\laya-nli-conflict-v4\`
  **model.safetensors SHA256 = e9e1c4a54560b8f1126510b0eb289d325757e42f6c765f098ed1a972d4b13758**
- 数据:`data_local/nli_conflict_train_v4.jsonl`(12000,含 meta)、
  `memory_conflict_realtest_v4.json`(老20+否定5+新10)、`label_swap_v4.json`、
  `noul_bias_diag_v4.json`、`val_soft_v4.json`、`calib_report_v4.txt`
- 脚本:`kaggle_eval/gen_soft_conflicts_v4.py`(seed 20260930)、`label_swap_check.py`、
  `realtest_v4.py`、`temperature_sweep.py`(ECE 未过项的收尾实测);calib_report.py 加了 v4 列
- 附:`data_local/val_probs_v4.json`(复杂 val 1000 的逐条概率,供温度扫描/复算用)
- HF 发布:`kaggle_eval/hf_model_card_nli_conflict.md`(模型卡)+ `hf_publish.py`(建仓+上传+回读校验);
  需先 `hf auth login`,仓库名 `laya-nli-memory-conflict`
- GitHub:全部脚本与文档已推 fork `main`(commit df2238a)
- 无被否决的 v4 候选(v4-ce 未跑),无 `-v4-alt` 需要清理;v3 头留档于
  `laya-nli-conflict-v3`(上一交付版),v2 头留档于 `laya-nli-conflict-v2`

**Kaggle 资产终态**:数据集 v9(README 含完整对照表+SHA256+ECE 注记)为最新;
kernel v8 = v4 训练(交付输出来源),kernel v9 = 文档版 quick save(最新版,无输出,
status ERROR 属占位)。取交付模型:本地副本,或网页 Versions → v8 → Output。
