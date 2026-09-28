# HANDOFF_NLI_V5.md — 第 5 轮任务书(NLI conflict head v5:训练轮)

> 读者 = 执行者(无聊天历史)。**开工前必读**:`HANDOFF_NLI_V2.md`(环境/凭据/Kaggle 三坑/插件红线)、
> `HANDOFF_NLI_V3.md`(温度扫描结论 + 本轮沿用的 v5 门槛重写意见)、`HANDOFF_NLI_V4.1.md`(第 4 轮判据与
> 交付;其文末「v5 训练轮预案」表是本轮范围来源)。冲突时以本文为准。凡引用外部结论必带链接。
> 全部数字可由仓内落盘数据复算;复算口径见 §2 末。

---

## 0. 本轮定位

v4 已交付(`D:\laya-kaggle-output\laya-nli-conflict-v4`,SHA256 见 V3)。第 4 轮复验结论:conformal PASS,
另外两轴 FAIL 成立,并确诊一个总根因:

| # | 失败/确诊 | v4 实测 | 来源 |
|---|---|---|---|
| 1 | **极性(名字体系)** NLI 侧 val_soft 决策错误 3 → 5(neutral) → 5(random),门槛「变化 ≤1」 | FAIL | `data_local/polarity_nli_v4.json` |
| 2 | **否定句 4/5**,连续两轮(v3、v4 都是 4/5) | 4/5 | `realtest_v2.py` 的 `NEGATION_CASES`(upstream laya#377) |
| 3 | **置信压缩带(总根因确诊)**:训练目标按类别均匀近硬(所有行 0.95/0.05),τ 拟合后输出恒定 ≈0.92 | 主 val band q90−q10 **0.62pp**;≥0.92 占 **87.1%**,该档错误率 6.66% | §2 |

**本轮目标(优先级从高到低)**

1. **名字体系增广(必做)** —— 修 #1:让模型不再从标签词本身的语义取捷径。
2. **counterfactual / 否定负例(必做)** —— 修 #2:否定算子与「表面像冲突、实为不冲突」的确定负例。
3. **让置信变可分(主攻)** —— 修 #3:让高置信档真的只装"确定"的样本,弃权层才有意义。

本轮是**训练轮**:runtime、τ、0.92 红线、`val`(1000)/`val_soft`(300) 全部零改动(见 §8)。

---

## 1. 范围表

| # | 项 | 本轮 | 依据 / 备注 |
|---|---|---|---|
| 1 | 选项名体系增广(中性名 + 随机串,行级混入) | **必做** | V4.1 §收尾:「两个头都判 FAIL ⇒ 预案 #1 从『建议』升级为『必做』」 |
| 5 | counterfactual 硬约束 / 否定负例 | **必做** | 预案 #5 触发条件连续两轮成立 |
| — | 分级软目标(置信可分的实现) | **必做** | V4.1:「v5 的主攻方向 = 让置信变可分」 |
| 2 | 纯 soft CE 消融臂(顺带 upstream laya#238 的 `loss = loss_ce / GRAD_ACCUM` 一行) | **必做(1 条对照)** | 预案 #2;laya#238 |
| 3 | R-Drop 消融臂(λ=0.1 起) | 可选(预算允许则跑) | 预案 #3 |
| 4 | NOTA / 显式弃权类训练 | 不做 | 触发条件原文是「**本轮弃权率 >15% 优先上调**」;29.4% 的弃权是本轮 conformal 层的行为,不是训练缺失类 ⇒ 留 v6 评估 |
| 6 | typed-decisions-v2(tasksource-jev 等) | 独立轨道,不阻塞 | 预案 #6 |
| 7 | 集成 miss 互补分析 | 轮空档再做 | 预案 #7 |
| — | multilingual 侧(另一个头) | **不在本轮** | 其训练管线不在本仓;极性失败如实记录在 V4.1 §收尾,需单独立项 |

---

## 2. 预核算(v5 门槛的分母;由 v4 落盘数据复算)

### 2.1 主 val(1000 MNLI,`data_local/val_probs_v4.json`)

- 错例 **99/1000**;置信 `conf = max(p_true, 1−p_true)` 分位:q01 **0.7408** / q10 **0.9186** / q50 **0.9230** /
  q90 **0.9248** / q99 **0.9263** / max **0.9279**
- **band = q90−q10 = 0.0062(0.62pp)**;q99−q10 = 0.0077
- ≥0.92 占比 **87.1%(871 行)**,该子集错 58 条、错误率 **6.66%**
- 置信分箱(错误率单调,但质量全挤在顶档):

| 箱 | n | 错 | 错误率 |
|---|---|---|---|
| [0.5,0.7) | 8 | 5 | 0.625 |
| [0.7,0.8) | 3 | 0 | 0.000 |
| [0.8,0.9) | 29 | 13 | 0.448 |
| [0.9,0.92) | 89 | 23 | 0.258 |
| [0.92,0.95) | 871 | 58 | **0.067** |

### 2.2 val_soft(300,`data_local/val_soft_v4.json`)

- 错例 **3/300**;conf q10 **0.9222** / q50 0.9240 / q90 **0.9250** / max 0.9265;**band 0.28pp**
- 按 kind:compat 85 行 0 错、consid 65 行 **2 错**、true 100 行 1 错、unrelated 50 行 0 错;
  holdout 类(4 个留出类)100 行 3 错
- consid 的 `p_true` 最高 0.9247 —— 边界样本已贴到 0.5 判定线附近,这正是极性翻转的温床

### 2.3 极性(`data_local/polarity_nli_v4.json`,seed 20260928,`option_polarity_check.py` 口径)

| 集合 | acc std/neutral/random | err std/neu/rnd | diff neu/rnd | mean\|Δp\| neu/rnd |
|---|---|---|---|---|
| real(20) | 20/20/19 | 0/0/1 | 0 / 1 | 0.0102 / 0.0805 |
| diag(14) | 13/13/13 | 1/1/1 | 0 / 0 | 0.0122 / 0.0640 |
| val_soft(300) | 297/295/295 | **3/5/5** | **2 / 2** | 0.0108 / 0.0458 |

gates:`real_diag_diff_le_1 = True`、`val_soft_err_change_le_1 = **False**` ⇒ **FAIL**

### 2.4 根因三条(代码实读,不靠推测)

1. **压缩带来自目标本身**:`gen_soft_conflicts_v4.py` L32-33 的 `GOLD_TRUE/GOLD_FALSE` 是常量 0.95/0.05,
   每一行都一样;`train_nli_conflict.py` L194 的 `loss_ce` 就对着这个 target 学 ⇒ 目标不含难度信息 ⇒
   模型把所有样本压在同一置信档,τ 拟合(L221-240)只是把这个常数搬了个位置。
2. **极性失败来自训练块的标签词永远是语义词**:4 套 labelset 全是「冲突/兼容/需更新旧记忆/supersede」这类
   自带极性的词,模型学到「答案词本身有语义」的捷径;探针的 random 档(val_soft 两档、real 的 random)
   最能暴露它。
3. **否定句失败是数据分布问题**:v3/v4 模板库里否定算子只出现在硬约束的安全侧(「不含花生」类),
   没有成对的「否定式冲突 / 否定式不冲突」样本 ⇒ #377 不是能力问题。

### 2.5 复算口径(必须与本节一致,防止计数歧义)

- 分位数用**最接近序号法**:`sorted(xs)[round(p·(n−1))]`,与本节数字同源;别混用线性插值。
- 错误率 = 该子集内 `argmax` 与 gold 不一致的比例;`p_true >= 0.5` 记 decision `true`。
- 复算输入就三个文件:`val_probs_v4.json`(1000 行 `{gold, p_true}`)、`val_soft_v4.json`(300 行
  `{pred, p_true, label, meta}`)、`polarity_nli_v4.json`。

---

## 3. 数据改造(`kaggle_eval/gen_soft_conflicts_v5.py`;「照着 v4 改」)

输出 `data_local/nli_conflict_train_v5.jsonl`。**val 与 val_soft 不重生成**(逐字节不变,直接可比)。

### 3.1 名字体系增广(必做)

- 保留 v4 的 4 套语义 labelset(LABEL_SETS 1-4,instructions 与 labels 一起轮换),**新增 2 套**:
  - **LS5 中性**:`labels = {'false': 'A', 'true': 'B'}`,`instructions` 不变(仍用语义定义句)。
  - **LS6 随机**:每行生成两个 **8 位、无元音** 的随机串(字母表 `BCDFGHJKLMNPQRSTVWXZ2456789`),
    与 `option_polarity_check.py` 的 random 规则**逐字一致**(探针/训练两侧必须同一构造,否则等于测了个新东西)。
- **覆盖面扩大**:v4 只对合成行轮换、MNLI 6000 行固定挂 LS1;v5 对**全部训练行**轮换(含 MNLI 与新块)。
  这不是笔误:v4 的一半数据没经过名字增广,模型仍能从 MNLI 行学到语义捷径。
- 配比(行级随机,硬性):语义 4 套合计 **≥50%**,LS5 **≥20%**,LS6 **≥20%**;计数写进交付报告。
- 每行 `meta.labelset ∈ {1,2,3,4,5,6}`(LS5=5,LS6=6),`meta.name_a/name_b` 记录实际标签串(便于审计)。
- 注意:该改动**破坏 v4 的「MNLI 6000 行字节不变」不变量**——本轮显式接受(训练块可变),理由写进报告。

### 3.2 counterfactual / 否定负例(必做)

两个新块,合计 **1400 行**(配额见下)。共同要求:**正负 1:1**,负例不得少于正例(v4 硬约束块的血泪教训:
只有正例时模型会把 miss 翻成 false alarm,原文写在 `gen_soft_conflicts_v4.py` 头注)。

**块 A「否定算子族」(800 行 = 400 true + 400 false)**

- 对现有 conflict / compat / consid 模板施加否定算子:
  ZH `{不, 没, 未, 从不, 不再, 没有}`、EN `{not, no longer, never, didn't, without}`。
- 每条模板必须成对产出:
  - 否定式**冲突**(true):`known=用户不吃辣` + `new=用户昨天点了麻辣香锅`
  - 否定式**不冲突**(false):`known=用户不吃辣` + `new=用户昨天没吃辣` / `用户点的是不辣的做法`
  - EN 对:`known=The user cannot eat spicy food` ↔ `The user ordered the spicy version`(true)/
    `The user ordered the non-spicy version`(false)
- 其中 ≥1/4 的负例必须是**双否定**(`known=用户已经戒烟了` + `new=用户没有复吸`,false)——
  `NEGATION_CASES` 里那条「双重否定-一致」若要 5/5,训练块必须见过这种形状。

**块 B「known 侧一步扰动」(600 行 = 300 true + 300 false)**

- 从 realtest 场景**形状**出发(不逐字复制 `realtest_v2.py` 的 20 条原文——那是对外验收集,
  原文入训会污染验收;用同一批属性类别另写模板即可),对 `known` 做**一步确定性扰动**
  (加限定词 / 换同义说法 / 加否定),使 `new` 成为**确定标签**的负例或正例:
  - 例:`known=用户对花生过敏` → `known=用户对花生过敏(已确诊)` + `new=用户点了不含花生的饼干` ⇒ false
  - 例:`known=用户每周三去健身房` → `known=用户的固定锻炼日是周三` + `new=用户把锻炼日改到周五` ⇒ true
- 该块的用途是把「表面像冲突」的负例密度提上来,专治边界类误报;负例的 `meta.kind = cf_false`。

**去重与泄漏**:沿用全局 `seen` 集合(含 val_soft 的 state,`gen_soft_conflicts_v4.py` L364-368 的写法),
assert 新块与 val / val_soft 零交集。

### 3.3 分级软目标(必做;压缩带的直接解药)

目标值 = **给 gold 标签的概率质量** `p(gold)`,另一侧为 `1−p(gold)`;`gold.label` 必须等于 argmax(assert)。
按 `meta.kind` 分档(**行级带 ± 抖动**,抖动只用于软目标,不改 argmax):

| meta.kind | v4 | v5 `p(gold)` | 理由 |
|---|---|---|---|
| mnli / contradiction | 0.95 | **0.95(不动)** | 主 val 就是它,保持逐轮可比 |
| mnli / entailment-neutral | 0.95 | **0.95(不动)** | 同上 |
| true | 0.95 | 0.93 ± 0.02 | 更新型冲突表述多样,略降 |
| compat | 0.95 | 0.93 ± 0.02 | |
| unrelated | 0.95 | 0.96 ± 0.01 | 最容易档 |
| **consid** | 0.95 | **0.80 ± 0.05** | **边缘类,必须下探**——这是 band 变宽的主来源 |
| hardconstraint | 0.95 | 0.90 ± 0.03 | 意图类天然有不确定性 |
| hc_safe | 0.95 | 0.90 ± 0.03 | |
| hc_unrelated | 0.95 | 0.96 ± 0.01 | |
| neg_true(新) | — | 0.90 ± 0.03 | |
| neg_false(新) | — | 0.88 ± 0.05 | 含否定算子的负例 |
| neg_false_boundary(双否定/近义否定) | — | 0.80 ± 0.05 | |
| cf_true(新) | — | 0.90 ± 0.03 | |
| cf_false(新) | — | 0.88 ± 0.05 | |

- 硬断言:每行 `abs(p(gold) − 0.5) >= 0.2`(防止抖动把某行推到判定线附近)。
- **已知代价(显式接受并写进报告)**:`val` 与 `val_soft` 的 gold 仍是 0.95,与 v5 输出的置信不再同尺度
  ⇒ 这两个集合的「标签-置信一致性/ECE」本轮**不作为判据**(门槛重写口径见 §5);它们只承担
  **准确率/错误计数**与**极性翻转**两类判据。

### 3.4 规模与硬不变量

| 块 | v4 | v5 |
|---|---|---|
| MNLI(取自 `nli_conflict_train.jsonl` 的 6000,状态逐字节不变) | 6000 | 6000 |
| 合成软冲突(true 1800 / compat 1440 / consid 1080 / unrelated 1080) | 5400 | 5400 |
| 硬约束(hardconstraint 300 / hc_safe 180 / hc_unrelated 120) | 600 | 600 |
| **否定算子族(新)** | — | **800** |
| **known 一步扰动(新)** | — | **600** |
| 合计 | 12000 | **13400** |

自检(全部 assert,写在生成脚本尾部):
1. 行数 = 13400;`state` 全局唯一;与 val / val_soft 零交集。
2. gold 计数按 kind 与上表一致;`true : false` 比例记录(不设硬门槛,但要报)。
3. labelset 分布满足 §3.1 配比(LS5 ≥20%、LS6 ≥20%);每个 labelset 至少出现在 3 个以上 kind。
4. 每行 `meta` 完整(`cat/kind/lang/labelset`;新块含 `name_a/name_b` 时一并保留)。
5. 每行 `gold.label == argmax(gold.probabilities)`;`abs(p(gold)−0.5) ≥ 0.2`。
6. `p(gold)` 的取值只在 §3.3 表列出的档位内(防止手滑写出 0.5)。

---

## 4. 训练改造(`kaggle_eval/train_nli_conflict.py` → v5)

- **主臂 A(必做)**:v4 配方原样(EPOCHS=4、MICRO_BATCH=16、GRAD_ACCUM=2、LR_ENCODER 2.5e-5、
  LR_HEAD 1e-4、SIGMA 0.4→0.1、`loss = (loss_rl + 1.0·loss_ce)/GRAD_ACCUM`)+ v5 数据。
  不换底座(`convaiinnovations/laya-multilingual`),不换 kernel 结构。
- **消融臂 B(必做,1 条)**:纯 soft CE —— 令 `loss = loss_ce / GRAD_ACCUM`(即 upstream laya#238 的
  一行 diff;v5 顺手把这行整理成可回帖的补丁文本,放交付物里)。数据与 A 完全相同,同表对比。
- **消融臂 C(可选)**:R-Drop(两次 dropout 前向的 KL,λ=0.1 起),仅在 A/B 都跑完且预算允许时跑。
- 新增 CLI:`--expected-train 13400`(沿用 v4 的防静默截断断言,数值改为 13400)、`--no-rl`(臂 B)、
  `--r-drop <λ>`(臂 C)。`max_items` 上限同步抬到 14000。
- 尾部保持:400 条 calib 切分(seed 20260927 不变)→ τ 逐题型拟合 → 落盘
  `model.safetensors` / `encoder` / `tokenizer` / `rl_agent_config.json`(`temperature[]` 为拟合值)
  / `metrics.json`。**新增**:把 val 1000 的逐行 `{gold, p_true}` 落盘为 `val_probs.json`(用 τ 校正后的 p)。
- τ 会随目标分布变化,这是预期行为;报告 τ 三个值并与 v4(`noul`=1.2176)对照,不做校准。

**止损线(先写死,执行时照判)**:
- 主 val acc 掉到 **< 0.885**(v4 0.901 −1.6pp)⇒ 判 FAIL,不再往上叠改动,原样报数。
- 极性 val_soft err 变化仍 **>1** ⇒ 本轮该轴判 FAIL,如实报(不许用「swap 轴过了」搪塞,那是另一根轴)。
- 否定句仍 **<5/5** ⇒ 判 FAIL 并列出逐条(5 条)判定与 p 值,写进报告。

---

## 5. 门槛(v5 口径;方向已由方向决策者确认,数值执行前复核)

### 5.1 硬门槛(全过才算交付)

| 集合 / 指标 | v4 实测 | v5 门槛 |
|---|---|---|
| 主 val(1000)acc | 0.901 | **≥ 0.896**(不显著退步) |
| realtest old-20 | 20/20 | **20/20** |
| realtest new-10 | 10/10 | **10/10** |
| realtest 否定 5 条 | 4/5 | **5/5**(本轮首修,upstream #377) |
| val_soft 错误数 | 3 | **≤ 3** |
| 偏置诊断 `noul_bias_diag` | 13/14 | **≥ 13/14** |
| label-surface swap diff | 0 | **0**(`label_swap_check.py` 口径不变) |
| 极性 real / diag 决策 |diff| | ≤1(已过) | **≤1 保持**(不放宽) |
| **极性 val_soft 错误变化** | **2(FAIL)** | **≤1(本轮第一验收)** |
| **分离度:val_soft band q90−q10** | **0.28pp** | **≥ 8pp**(新指标,§2.5 口径) |

### 5.2 报告项(不设硬门槛,但必须给数)

- 主 val:E**C 按 v5 重写口径报**——只在 `acc<1` 的切片上评 ECE / `|acc−mean_conf|`;`acc=1` 的切片
  改报 `(acc, mean_conf)` 二元组;0.92 置信天花板(≥0.92 占比)作为独立跟踪量单列。
  依据 = V3「温度扫描结论」末段(不放宽绝对 ECE;hard-label ECE 在 acc=1 时 ≡ 1−置信上限,会给假信号)。
- 分离度全套:主 val 与 val_soft 的 conf q10/q50/q90/q99/max、band、置信分箱(箱内 n / 错 / 错误率),
  与 §2 两张表逐格对照;`consid` 类的中位置信单列(v4 = 0.9239)。
- conformal(`conformal_abstain.py`)沿用:v4 = 捕捉 58/99 @ 弃权 29.4%,门槛 ≥50/99 @ ≤35%;AURC v4 = 0.05564
  (oracle 0.00512)对照报。
- 极性 mean|Δp|:real/diag/val_soft × neutral/random,与 §2.3 对照(名字增广是否真的压小了对名字的敏感度)。
- τ 三个拟合值(与 v4 `noul`=1.2176 对照)。
- 训练曲线:每 epoch 平均 loss、reward 均值;臂 A/B 同表。

### 5.3 交付判定

任一硬门槛 FAIL ⇒ 交付物照出、结论写 FAIL(负结论也算完成);**不许**通过调门槛、换判据集、
在报数集上挑样本等方式把 FAIL 抹平。预核算显示不可达的门槛(如「50/99 @ ≤20% 弃权」)一律不允许硬凑。

---

## 6. 交付物与归档

**脚本(进仓,`kaggle_eval/`)**

- `gen_soft_conflicts_v5.py`(数据生成,含 §3.4 全部断言)
- `train_nli_conflict.py`(v5:三个新 CLI 开关 + val probs 落盘)
- `nli_kernel.py`(kernel 版,版本表加 v10 行;嵌入的 train 脚本同步为 v5 版 base64)
- `dump_probs.py`(新,约 40 行:对 300 条 val_soft 落盘 `{pred, p_true, label, meta}`,形状同
  `data_local/val_soft_v4.json`)——若沿用上一轮已有脚本则不必新建,但**输出形状必须一致**
- 复验沿用:`option_polarity_check.py`、`realtest_v2.py`、`realtest_v4.py`、`noul_bias_diag.py`、
  `val_breakdown.py`、`conformal_abstain.py`(均**不改判据**)

**数据产物(`data_local/`)**

- `nli_conflict_train_v5.jsonl`(13400 行)
- `val_probs_v5.json`、`val_soft_probs_v5.json`、`polarity_nli_v5.json`、
  `memory_conflict_realtest_v5.json`、`noul_bias_diag_v5.json`、`conformal_v5.json`
- `conf_band_v5.txt`(§5.2 分离度全套,含与 §2 两张表的逐格对照)

**Kaggle 资产**

- 数据集 `daphnelaurent/nli-conflict-pairs` **v10**:新 train + 不变的 val(1000)/val_soft(300) +
  `_README.md`(版本表加 v10 行;顺带把 v4 遗留的「上_ITER」笔误与缺行修掉——上一轮修过但没推,这次随版本同步)
- kernel `daphnelaurent/laya-nli-memory-conflict-fine-tune` **v10** = 臂 A 训练;臂 B/C 各自单独 kernel
  (或同 kernel 加参数重跑,二选一,在报告里写清每次运行对应的 commit/参数)

**模型产物**:臂 A/B(/C)各自 `model.safetensors` + `metrics.json` + `rl_agent_config.json`,
本机留档 `D:\laya-kaggle-output\laya-nli-conflict-v5[-ce][-rdrop]\`;总大小与 SHA256 写进报告。

**HF**:v5 卡更新(同基准对比节 + 极性段 + 新的分离度/否定句数字);若只有臂 A 达标则只推 A,
其余在卡上作对照记录。

---

## 7. 执行环境与命令(踩过的坑,照抄)

- Kaggle CLI 的 **tmp 必须指到 ASCII 路径**(如 `C:/kaggle_cfg/tmp`);token 走 `KAGGLE_API_TOKEN`
  环境变量,不落盘、不进日志(细则见 `HANDOFF_NLI_V2.md`「环境与凭据注意事项」)。
- **数据集总挂最新版**:上传 v10 后 kernel 会自动拿到 v10;若中途只改 README 也要出新版号,否则
  kernel 侧看到的还是旧版。
- **每次 push kernel 都会从头重训**(CUDA 非确定性 ⇒ 权重不可位复现):本机留档的副本才是参照物;
  不要靠重跑复现数字。
- kernel 的 `QUICK_SAVE` 纯文档版无输出(status ERROR 属占位),别在那里找模型。
- 本机 CPU 复现路径:`D:/Miniconda/envs/laya-ft/python.exe` 直接跑 §6 的复验脚本(逐条 CPU 前向,
  300 条 val_soft 的极性检查约几分钟;`pandas` 在本机被系统策略挡,一律走 `pyarrow`)。

---

## 8. 不做清单(硬)

- 不动 τ 与 0.92 红线(红线属 runtime 配置,改动属 v6+ 单独立项)
- 不改 `val`(1000)/`val_soft`(300) 的任何字节;不在报告集上做任何拟合
- 不引入外部数据(tasksource-jev、Sev 等只作为**方法**参考,不下载不用)
- 不把 realtest 20/10/5 条原文写进训练数据(形状可借,原文不可;验收集必须保持未见过)
- 不重修 multilingual 头(另立项)
- 不放宽任何判据、不删既有负结论记录

---

## 9. 开放决策(执行前请方向决策者拍板;不拍板则按默认走)

1. **消融臂数量**:默认 = A(主) + B(CE-only)两条;B 是必做里的「同轮消融」。
   是否加 C(R-Drop)?默认**不加**(预算优先留给数据/门槛验证)。
2. **主 val 压缩带是否本轮追**:默认**不追**。追它需要 MNLI 逐条难度目标(逐标注者票型),
   而本仓 MNLI 取自 `nyu-mll/glue`(只有单一硬标签,无标注者分布)⇒ 需要换数据源,属范围外。
   默认口径:分离度主战场 = val_soft/合成块(§5.1),主 val 只报 band 不设门槛。
3. **typed-decisions(multilingual)头的名字增广**:默认**不同轮**。其训练管线不在本仓,
   若要求同轮,需先定「用哪个脚本重训」——那本身是一个独立小立项。

---

## 附录 A. 版本与文件对照(本轮)

| 对象 | v4 | v5 |
|---|---|---|
| 训练数据 | `nli_conflict_train_v4.jsonl`(12000) | `nli_conflict_train_v5.jsonl`(**13400**) |
| 数据生成 | `gen_soft_conflicts_v4.py` | `gen_soft_conflicts_v5.py` |
| 训练脚本 | `train_nli_conflict.py`(v4) | 同文件 v5(三开关) |
| Kaggle 数据集 | v8 | **v10**(README 同步) |
| Kaggle kernel | v8(交付)/v9(文档版) | **v10** |
| 交付头 | `laya-nli-conflict-v4` | `laya-nli-conflict-v5` |
| val / val_soft | 1000 / 300 | 逐字节不变 |

---

## 第 5 轮执行结果(2026-09-28 执行;两臂均判 FAIL,v4 仍是已交付头)

```
第 5 轮验收汇报(kernel v10 = 臂 A v4 配方 / kernel laya-nli-conflict-ce v1 = 臂 B 纯 CE;数据集 v10)
门槛项                | v4      | 臂 A        | 臂 B        | v5 门槛
主 val acc           | 0.901   | 0.893 ✗     | 0.902 ✓     | ≥0.896
realtest 老 20        | 20/20   | 19/20 ✗     | 19/20 ✗     | 20/20
realtest 新 10        | 10/10   | 8/10 ✗      | 10/10 ✓     | 10/10
realtest 否定         | 4/5     | 4/5 ✗       | 4/5 ✗       | 5/5
val_soft 错误         | 3       | 1 ✓         | 7 ✗         | ≤3
偏置诊断             | 13/14   | PASS ✓      | PASS ✓      | ≥13/14
标签面交换 diff       | 0       | 0 ✓         | 0 ✓         | 0
极性 real/diag        | ≤1      | ✓           | ✓           | ≤1
极性 val_soft 变化    | +2 FAIL | **≤1 PASS** | +2 ✗        | ≤1
val_soft band        | 0.28pp  | **16.29pp ✓** | **16.18pp ✓** | ≥8pp
conformal            | 58/99@29.4% | 12/107@2.3% ✗ | 16/98@3.0% ✗ | ≥50 @ ≤35%
```

**判定:两臂各 4 项硬门槛 FAIL ⇒ 本轮无交付头,v4 保持已交付;HF 不推**(§5.3:负结论照交,不调门槛)。
τ 拟合值:臂 A noul=1.138 / 臂 B noul=1.065(v4=1.2176;τ 随目标分布变化属预期,不做校准)。

### 达成的(写进下轮资产)

- **压缩带打穿(本轮主攻)**:分级软目标把 val_soft band 从 0.28pp 拉到 16.2-16.3pp(两臂一致,
  门槛 2 倍余量);臂 B 主 val band 3.96pp,置信分箱恢复单调错误梯度(0.64/0.47/0.35/0.19/0.06)。
  证据:`conf_band_v5.txt`。
- **极性轴修好(臂 A)**:val_soft 错例变化 ≤1(v4 +2)——名字体系增广(6 labelset 全行轮换)起效。
- **置信排序质量**:AURC 0.0464(A)/0.0440(B) vs v4 0.0556。
- val_soft 错误:臂 A 1/300(v4 3/300)。

### 三条根因(逐条对得上 miss 明细)

1. **新块缺 unrelated 控制行(v4 硬约束块教训原样重演)**:neg/cf 两块全部是域内正负对照,
   无 unrelated 侧 ⇒ 两臂都在老 20 的「无关补充」上新增误报(p 0.775/0.782)⇒ 老 20 双双 19/20。
   v6:两块各补 unrelated 控制行(约 1:0.5)。
2. **否定家族形状不命中**:否定句仍 4/5——臂 A 的 miss 换成了「否定-无车与通勤」(p=0.5022,贴线),
   臂 B 仍是「否定-养宠物反转」(0.1274)。我写的家族(饮食/烟/理发)没覆盖这两个形状
   (财产/通勤归属反转)。v6:按 NEGATION_CASES 的五条形状逐族补模板(形状可借,原文不入训)。
3. **RL 项与分级软目标相克**:臂 A 主 val 0.893(−0.8pp)且 realtest 三项崩;臂 B(纯 CE)主 val
   0.902 反超 v4。结论与 Sev 研究一致——目标含难度信息后,RL 项从「噪声平滑 CE」变成「直接伤害」。
   v6 主臂 = 纯 CE,RL 项留作消融对照。

### 残留与工具缺口

- conformal 门槛 FAIL(12/107、16/98 @ ~3%):band 内可分 ≠ 跨集同尺,val_soft 拟合阈值迁移主 val
  仍失效;弃权层要可用需同分布标定(v6:主 val 对半切参照组)。
- consid 中位置信未报:val_soft 冻结文件无逐行 meta,dump_probs 落占位 meta——v6 让 dump 带行号回查。
- 过程修复(数据生成期,commit 134312c 已含):gold_for 语义反转(p(gold) 错判给 true 侧)、
  多项抽样漂移→精确配额池、拒绝重抽压低 zh 份额→新块按语言精确配额、ZH neg 模板空间不足→补第 4 act。
- 本机→kaggleusercontent 大文件链路间歇断流:kernels_output 重试循环 + 串行下载解决(教训已入记忆)。
