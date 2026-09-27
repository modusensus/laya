# Laya 决策头 — 第 4 轮任务书(重写版 v4.1):极性诊断 + 校准预检 + 弃权审计(权重不动)

> 本文件**替代** `kaggle_eval/HANDOFF_NLI_V4.md`(其修补稿保留在盘上供对照;冲突时以本文件为准)。
> 命名:文档编号 = 交接轮次;v4.1 = 第 4 轮的重写修订,与数据/权重版本号无关。
> 重写依据:①复核 HANDOFF_NLI_V4.md 修补稿;②**落盘数据全量复算**(新增「预核算」节,数十项数字已核死);
> ③GitHub 侧生态调研(**新增存档 B 节**)。与原稿差异摘要如下——读完这一节等于读完全部改动:

**与 HANDOFF_NLI_V4.md 的差异摘要**

1. **数字修正**:主 val 错例 = **99**(非 98;真类 57 + 假类 42,复算见预核算节)。原稿「98」沿袭自
   HANDOFF_NLI_V3.md 的笔误,凡以「/98」为分母的判据全部按 99 重算。
2. **conformal 门槛重设**:原「捕捉 ≥50%(≥49/98)且弃权 ≤20%」经复算是**不可达判据**——
   主 val 上捕捉 50/99 最少需 20.6% 弃权,而 ≤20% 预算的包络上限恰好 49/99。改为:
   **主判据 = 名义 90% 规则下捕捉 ≥50/99 且弃权 ≤35%**(实测 58/99 @ 29.4%,富余 8 条);
   20% 档数字降为报告项(见改动 3)。
3. **改动 2 预检判据升级**:从「两类是否同向」改为「**矩阵:每类 × 每个集的方向**;任一类跨集方向
   相反 = FAIL」。预核算已给初判:true 类在主 val 需软化(+0.086)、在 val_soft 需锐化(−0.070),
   **方向相反 ⇒ 预期负结论收尾**(仍实跑曲线复核,负结论照样是交付)。
4. **改动 1 加料**:multilingual 数据拉取给出精确命令/文件/断言行;labels 改名行为**单例实测**(答案键
   恒为 `noul`,不随 labels 变;p 值随渲染变化 0.9225→0.8997→0.8696);随机串生成规则;基线复跑步骤;
   可选附加观察(NLI 假设句渲染,Verdict 先例)。
5. **改动 3 加料**:新增「组合规则审计表」(写入率 × 写入错误率 × 弃权率档位表,预核算已预填),
   直接把弃权层翻译成插件可用的产品档位选型;并新增 τ 变更的副作用警告。
6. **新增存档 B**:GitHub 侧生态地图(母体/移植/他人后训练/工具/可抄点映射)。
7. **恢复记录精确化**:现场恢复的 git 事实写法修正(被跟踪文件 vs gitignore 文件)。
8. **v5 预案**:保留 7 项;触发条件更新——#5(counterfactual 负例)**已触发**(否定句连续两轮 4/5);
   证据补 GitHub 侧三项(覆盖率实验 rlcd、可跑重建、kev/nimble 方法论)。

## 任务一句话

v4 头(SHA256 `e9e1c4a5…b13758`)分类门槛全过、两项 ECE 因「acc=1.0 撞 0.92 置信带」未过(结构性)。
本轮**零训练、零 runtime 改动**,产出四件事:**①选项名极性诊断(两个头)——检验头是否在读选项名
的语义极性;②per-class vector scaling 预检+离线重标定——检验「按类拆温度」能否解跨集死局;
③split-conformal 弃权层+组合规则审计——给插件写入阈值一张可审计的档位表;④两张模型卡补
「同基准对比」节**。v5 训练轮预案见文末(本轮只存档)。

## 必读前置(不重复,自己去看)

| 文件/节 | 为什么必读 |
|---|---|
| `kaggle_eval/HANDOFF_NLI_V3.md` 全文 | v4 执行结果、温度扫描全表、「v5 门槛重写」验收意见,本轮沿用 |
| `kaggle_eval/HANDOFF_NLI_V2.md` | 环境、凭据规矩、Kaggle 三坑、**插件接入红线**(「v4 待办」第 4 节:自动写入阈值不得 >0.92) |
| `kaggle_eval/HANDOFF_NLI_V4.md`(修补稿) | 对照本文件的差异摘要;生态存档 A 的出处也在本文件下方 |
| **本文件「预核算」节** | 全部判据的分母/包络/转移数字,执行前先读,执行后用脚本复核 |
| `kaggle_eval/temperature_sweep.py` + `data_local/temperature_sweep_v4.txt` | 单 τ 死局实测曲线,改动 2 是它的替代方案 |
| `kaggle_eval/label_swap_check.py` + `data_local/label_swap_v4.json` | 标签**取值措辞**轴已做过(差 0 通过);本轮的极性轴是另一件事,别混 |
| `kaggle_eval/realtest_v4.py`、`noul_bias_diag.py`、`calib_report.py` | 三套现成评测骨架,新脚本照改;校准数字全从 calib_report 出 |
| `laya/agent.py`(`_decode_answers`,:768 附近)+ `laya/common.py`(温度分桶) | 改动 2 定性依据:温度粒度 = 选项数桶、decode 无 bias 项、clamp [0.5,5.0]——别再当它是配置层 |

## 起点状态(v4 实测,别重算;除注明外数字来自 HANDOFF_NLI_V3.md)

- v4 头:主 val 0.901 / ECE 0.0192;偏置诊断 13/14(A1 两方向 0.9225/0.9225);真实 case 老 20 **20/20**、
  新 10 条 10/10、否定句 4/5;val_soft 297/300(holdout 97/100);zh 0.986 / en 1.000
- ECE 未过项(val_soft 0.0703 / realtest 0.0765)= acc 满分撞置信带的结构性结果;
  温度扫描:**全网格无单一 τ 同时满足三集**(主 val 要 τ≳1.2,realtest/val_soft 要 τ≲0.8,最优解差约 2×);
  配置保持 τ=1.2176(noul)
- v5 门槛重写方向(已确认):只在 acc<1 的切片评 ECE/|acc−mean_conf|;acc=1 切片报 (acc, mean_conf);
  置信带作为独立跟踪量;不放宽绝对数
- typed-decisions multilingual(另一头,官方 400 cases/2000 decisions 口径):acc 0.7875
  (choice 0.752 / noul 0.862 / score 0.759),ECE 0.159,温度 choice 1.11 / score 1.05 / noul 1.20;
  per-language held-out 仍欠账(模型卡自认)
  ⚠️ **数据口径**:本机 `data_local/typed_decisions_test.jsonl` 只有 100 cases(spot 子集),
  0.7875 是官方 400-case 口径;本轮 multilingual 侧一律从 HF 拉全量(命令见改动 1),
  100-case 文件只许冒烟不许出验收数字
- 生态定位:0.7875 处于 specialist frontier(0.7885–0.796)与官方 laya-td(0.766)之间;
  ECE 0.159 落后第一梯队(0.092–0.096)——校准是短板

---

# 预核算(2026-09-28,用落盘数据算死;执行者按此复核而非替算)

> 复核注记(2026-09-28,`D:/Miniconda/envs/laya-ft/python.exe` 全量复算):§1 分解、§3
> 阶梯五行、§4 conformal(k=271/q̂=0.0780/覆盖 0.9067/四档迁移 13.1-41、29.4-58、37.7-64、
> 45.9-68)、§5 val_soft 288/2 与 realtest 30/0、34/0、预检方向(true 类主 val +0.0861 vs
> val_soft −0.0696)**全部复现一致**;唯一出入 = §3 包络 20.0%/20.6% 两格复算为 **48/49**
> (文档 49/50,并列值在第 k 位序统计量处的计数歧义)——「≥50 @ ≤20% 不可达」结论不受影响
> (复算口径下 ≤20% 最多 48-49,凑 50 需 ~21%+ 预算)。

**数据来源与口径**:`data_local/val_probs_v4.json`(主 val 1000 条 {gold, p_true},τ=1.2176)、
`val_soft_v4.json`(300 条 pred/label/p_true/meta)、`memory_conflict_realtest_v4.json`(20+5+10)、
`noul_bias_diag_v4.json`(14)、`temperature_sweep_v4.txt`。
复算口径:pred = (p_true ≥ 0.5);max_conf = max(p, 1−p);「决定/写入」= max_conf ≥ 阈值;
错例 = pred ≠ gold(与 metrics.json 0.901 自洽)。

### 1. 主 val 错例分解(n=1000,errors=99)

| 切片 | n | acc | 错例 | mean_conf | mean_conf − acc |
|---|---:|---:|---:|---:|---:|
| 真类(gold=true) | 334 | 0.8293 | **57** | 0.9154 | **+0.0861**(过信) |
| 假类(gold=false) | 666 | 0.9369 | **42** | 0.9187 | −0.0182(略欠信) |
| 全体 | 1000 | 0.9010 | 99 | 0.9176 | +0.0166 |

### 2. 置信分布是「压缩带」(本轮最重要的新事实)

- 主 val max_conf 分位:q10 **0.9185** / q50 **0.9230** / q90 0.9248 / q99 0.9263 / **max 0.9279**;
  **无一条 ≥0.93**
- val_soft:q10 **0.9221** / q50 0.9240 / max 0.9265
- 读数:**这个头对几乎一切输入都输出 ~0.92**。准确率 90.1% 与置信 91.8% 看起来「校准良好」,
  其实是「恒定输出 0.92」的副产品——**置信几乎没有区分力**。一切死局(单 τ 无解、ECE 恒等于
  1−置信、per-class 难有戏)的总根因在这里,不止是「0.92 天花板」。
- 附:两集分位数犬牙交错(val_soft 略高)——任何按分位数定阈值的规则跨集迁移都会踩到悬崖(见 §4)。

### 3. 阈值阶梯(主 val,按 max_conf 决定)

| 阈值 t | 决定率 | 写入错误率 | 弃权率 | 捕捉错例 |
|---|---:|---:|---:|---:|
| 0.900 | 96.0% | 8.44% | 4.0% | 18/99 |
| 0.910 | 93.8% | 7.78% | 6.2% | 26/99 |
| 0.920(红线档) | 87.1% | 6.66% | **12.9%** | **41/99** |
| 0.9220(90% 名义档) | 70.6% | 5.81% | 29.4% | **58/99** |
| 0.9225(85% 名义档) | 62.3% | 5.62% | 37.7% | 64/99 |

包络(任何仅用 max_conf 的规则的极限):10%→37、12.9%→41、**20.0%→49**、**20.6%→50**、25%→54、30%→58、40%→64。
⇒ **「≥50/99 @ ≤20%」结构上不可达**(差 0.6pp 预算);0.9200→0.9220 之间 0.2pp 阈值浮动 ⇒ 弃权率
12.9%→29.4%——置信值离散堆叠,阈值全踩在台阶上,写报告时必须报实际档。

### 4. split-conformal(名义 90%,LAC)在 val_soft 300 上拟合 → 迁移数字

- 非一致性得分 s = 1 − p(真类);q̂ = 0.0780(271 位序统计量)→ 决定条件 max_conf ≥ 0.9220
- **val_soft 自身**:决定 274/300(91.3%),覆盖率(决定∧正确)/n = **0.907**(名义 90% 成立)
- **迁移到主 val**:决定 706/1000(70.6%),弃权 29.4%,捕捉 58/99(58.6%),写入错误率 5.81%
- 迁移成本:同样的规则,val_soft 上弃权 8.7%、主 val 上弃权 29.4%——**分布迁移把弃权率翻了三倍多**
- 名义 95% / 85% / 80% 变体迁移到主 val 分别为:弃权 13.1%/37.7%/45.9%,捕捉 41/64/68/99

### 5. 组合规则审计(给插件的档位表;红线语义未动)

写入规则 = max_conf ≥ t;下表即「产品选哪档」的依据(主 val):

| 档 | 写入率 | 写入错误率 | 备注 |
|---|---:|---:|---|
| 现红线 0.92 | 87.1% | 6.66% | 当前插件行为 |
| 0.9220(90% 名义) | 70.6% | 5.81% | 弃权层推荐档 |
| 0.9225(85% 名义) | 62.3% | 5.62% | 错误率地板附近 |

- val_soft 上红线档:288/300 决定、2 错(0.69%);realtest 35 条 @0.92 决定 30 条 0 错(@0.90 决定 34 条 0 错)
- **τ 警告(预核算 §2 的直接推论)**:置信带是 τ 的函数。若为了 ECE 把 τ 从 1.2176 调到 0.7–0.8
  (扫描表上的「锐化」档),主 val 上写入率会从 87.1% 涨到 98.5–98.8%、写入错误率从 6.66% 涨到
  **9.4–9.5%**——修 ECE 的顺手动作会把写入门变稀烂。**任何 τ 变更提案必须先过这张审计表。**

### 6. 读数(写进报告的五句话)

1. 99 条错例中 58 条在红线档以上被判「自信」——置信门只能拦住 41 条;压缩带是总根因
2. per-class 解死局:true 类跨集方向相反(§见改动 2 预检),预计负结论
3. conformal 规则迁移后:捕捉 58/99 @ 弃权 29.4%(90% 名义档);红线档 41/99 @ 12.9%
4. 写入错误率在 6.7%(现红线)→5.6%(弃权 38%)之间,曲线浅——弃权层的上限就在这,别指望它单兵救场
5. 「50% @ ≤20%」不可达;凡门槛必须按包络/实测档写,不许硬凑

---

# 本轮改动清单(全部零训练;Python 一律 `D:\Miniconda\envs\laya-ft\python.exe`)

### 改动 1(诊断,必做):选项名极性检查——两个头都跑

新脚本 `kaggle_eval/option_polarity_check.py`,照 `label_swap_check.py` 骨架改。

**与 v4 已有工作的关系**:v4 的「标签面改写增广」+ `label_swap_check.py` 治的是**取值措辞**轴
(冲突↔兼容互换,实测差 0,已过)。arXiv:2609.26758 指出的是另一根轴:**名字体系本身的语义**
(0/1 vs no/yes vs 随机串)。v4 轮换的 4 套说法都是有语义的,模型仍可能依赖词的极性。

**① NLI 头**(ckpt `D:\laya-kaggle-output\laya-nli-conflict-v4`):
- 集合:real 20 + 偏置诊断 14 + val_soft 300;三版:标准版 / 中性名(`{"false":"A","true":"B"}`,
  映射固定)/ 随机串版(每 case 独立,规则见下)
- ✅ **输出键行为已实测**(2026-09-28 单例探针,CPU,`laya.load(v4)`):答案键**恒为 `noul`**(不随
  labels 变),值 = 映射到 true 的选项概率;但**渲染会改变 p 值**——同一输入 0.9225(「冲突/兼容」)
  → 0.8997(A/B)→ 0.8696(随机串)。⇒ 翻转按决策(0.5 两侧)算,**p 位移(mean|Δp|)另报**作观察项
- 随机串规则:大小写字母+数字,长度 6–10,逐 case 独立生成,避免可读词;seed 固定并写进报告

**② multilingual 头**(ckpt `D:\laya-kaggle-output\laya_finetuned_typed_decisions`,本地 CPU 推理):
- **数据源(全量 400 cases/2000 decisions,官方口径)**,拉取方式二选一:
  - `HF_ENDPOINT=https://hf-mirror.com` 后:`load_dataset("LocalLLaMA/typed-decisions","all",split="test")`
    (若环境无 datasets 库走下面 parquet)
  - 直接下 parquet:`https://hf-mirror.com/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet`
    (222,140 B 为拉全的 sanity;`pd.read_parquet` 读)
  - **断言**:行数 = 400;decisions 合计 ≈ 2000(按实际 schema 数);**记录 resolve 到的完整 revision**
    (模型卡引用 `ea930645`,能 pin 就 pin)。本机 `data_local/typed_decisions_test.jsonl`(100 cases)
    只许冒烟
- **先复跑基线**:400-case 全量跑一遍,对照卡面 0.7875/choice 0.752/noul 0.862/score 0.759
  (差得多先查加载/口径,别急着做变体)
- 然后:**choice 题型**做①中性名②随机串两版(论文协议:问题、state、rubric 全不动,只换选项名),
  输出逐题翻转清单;noul/score 题型不动,报基线即可
- 顺带记录两头 pooling 类型(论文发现 mean-pooling 翻转少 4.1×,归档用)

**门槛**:
- NLI 头:real 20 + diag 14 两版各与标准版**差 ≤1 条(双向)**;val_soft 错例数变化 ≤1
- multilingual choice:**翻转总数 ≤2%**(按实际 choice 题量)、McNemar 检验翻转方向不对称性 **p > 0.05**,
  附翻转清单(谁翻成了谁)
- 超门槛 = 读出名字极性,v5 必做名字体系增广,且**如实报 FAIL**——不许拿「swap 过了」搪塞(另一根轴)

**附加观察(可选,≤20 分钟,不设门槛)**:把 NLI 头 labels 的取值词从「冲突/兼容」换成完整的
假设句(如 true→「与已知记忆矛盾,应更新」;false→「与已知记忆一致或不冲突」),在 real 20 + diag 14
上重跑,报 acc 与 p_true 位移。出处:Verdict 2.0 的 v1.4 推理修复(候选改写成 NLI 假设句)先例
(见存档 B)。有差异 → 进 v5「标签措辞」议题;没差异 → 一句话带过。

### 改动 2(诊断,必做):per-class vector scaling——预检升格 + 离线重标定

**定性(不变)**:不是配置层。runtime 实读:温度粒度 = 题型×选项数桶(`noul:2`,无 per-class);
decode 全路径只有 `z = logits[r,:k] / t_scale`(`agent.py:768`),无 bias 项;温度 clamp [0.5,5.0]。
「argmax 不变」只对单 τ 成立,**per-class τ 能改变 argmax**。本轮只做离线实验,**不改 runtime、
不产出接入配置**;若将来要上,最小 runtime diff 属 v5 轮(附回退说明)。

**预检(先做,约 10 分钟;判据已升级)**:
- 画 **矩阵**:对 [主 val, val_soft](realtest 35 条 0 错,只记不判)× {true 类, false 类} 计算
  mean_conf − acc 的方向;需要两个集都**有错例**才有判别力
- **判据**:任一类别在两个集的需修方向**相反** ⇒ FAIL,改动 2 以**负结论报告**收尾(负结论 = 完成);
  每类方向一致才进入拟合
- **预核算初判(已给出,实跑复核即可)**:true 类 主 val **+0.086**(去过信,需软化)vs val_soft
  **−0.070**(欠信,需锐化)→ **方向相反 ⇒ 预计 FAIL**。矩阵+曲线写进 `vector_scaling_v4.txt`
- 为什么预判这么强:置信是压缩带(预核算 §2),per-class 缩放是在压缩信号上再缩放,
  且 bias 项 (b_t−b_f) 等价于先验平移,同样解不了「同类跨集反向」

**若预检 PASS(意外情况)才继续**:dump 原始 logits(注意:s 型得分只从 p 复原不出 per-class 参数,
必须小脚本直出每行 `[l_false, l_true]`,照 temperature_sweep.py 的模型加载骨架写,存
`data_local/noul_logits_v4.json`)→ 在 val_soft **全 300** 上拟合 (τ_t, b_t, τ_f, b_f),目标 = NLL/
置信对齐(不追错误样本;写进报告)→ 报告集 = 主 val 1000 + realtest 35 + diag 14,**与拟合面不相交
(脚本留 assert)**;报 argmax 改变条数;clamp [0.5,5.0] 的部署可行性注记进报告。

**报数口径(若拟合)**:acc=1 集 |1−mean_conf| 下降 ≥0.02 且 Brier/NLL 不劣基线;主 val 上 Brier/NLL
不劣基线;ECE 只作卫生检查。三行对照:基线 τ=1.2176 / 单 τ 最优 / per-class。

### 改动 3(弃权层,必做):split-conformal + 组合规则审计

新脚本 `kaggle_eval/conformal_abstain.py`(RAPS 的 noul 二类简化,split-conformal 足够)。

**拟合与判据(数字全部按预核算重写,不许再用 98)**:
- 拟合面:val_soft **全 300**,得分 s = 1 − p(真类),目标名义覆盖 90% → q̂(≈0.0780,执行时实算);
  拟合面本身不承担判据
- **主判据(在主 val 1000 上,99 条错)**:
  ① **捕捉 ≥50/99(50.5%)** 且 **弃权 ≤35%**(实测档:58/99 @ 29.4%,富余 8 条 / 5.6pp);
  ② AURC + 风险-覆盖曲线,与「无弃权基线」「包络(oracle 上界)」同图;
  ③ **逐条清单**:被弃权的错例逐条列(编号/known/new/p_true/期望)——正式判据,可点验
- **报告项(不设门槛)**:val_soft 自身覆盖率(实测 0.907,名义 90%)与决定率;realtest 35 的
  (覆盖, 弃权);**20% 预算档的数字(包络 49/99,已证 50 不可达——如实写,不许硬凑)**;
  名义 95/85/80% 变体三行(预核算 §4 已给初值)
- **组合规则审计表(必交,模板见预核算 §5)**:写入 = max_conf ≥ t;至少四档
  (0.92 红线 / 0.9220 / 0.9225 / 0.90)+ 现行值;主 val 与 val_soft 各一组;τ 警告一句带上
- **可交换性注记(必须写进报告)**:val_soft(合成)拟合 → 主 val(MNLI 系)评测是分布迁移,
  主 val 数字是「迁移表现」非有限样本保证;可加主 val 对半切(500/500)同分布参照组
- **红线不动**:插件自动写入阈值 ≤0.92 照旧;conformal 是附加层——写入 = 达红线 **且** 未被弃权;
  不许用弃权层放宽门槛,不许把 conformal 阈值直接当写入阈值

### 改动 4(报告,必做):两张模型卡补「同基准对比」节

- 只引用同基准数字(typed-decisions test 口径:我方 0.7875 vs 官方 0.766 vs Jev 0.727);
  跨基准对比显式标注「不同基准,不可比」
- v4 NLI 卡补一句极性/校准层说明(改动 1–3 出数后);可选:加「≥0.92 置信子集错误率」行(数字干净才加)

## 明确不要做

- **本轮不重训**:不动权重、不动 MNLI 冻结集、不改老 20;所有重训结论写预案节
- 不用弃权层/vector scaling 放宽既有分类门槛;**不改 laya runtime 任何代码、不产出接入配置**
- **不动 τ=1.2176**:任何 τ 变更提案必须附预核算 §5 形式的审计表(τ 变 → 写入门变);不许为 ECE 静默换 τ
- **不许在主 val 上调 conformal 的 q̂/threshold**(那是 snooping;q̂ 只能来自 val_soft 或其子集)
- 拟合面与报数集不许相交(脚本留 assert,V3 已踩过一次)
- 引用外部数字必须带链接+写明基准/尺子;polaris 只准用存档表已核文字
- 不引入 tasksource-jev 等外部数据(那是 v5/typed-decisions-v2)
- **不删、不覆盖恢复现场**(mtime 证据留档);`kaggle_eval/rl_agent_config.json/` 野目录不删不碰
- 「50/99 @ ≤20%」已被预核算证明不可达,**不许硬凑**(调拟合面、换判据集都是没活硬整)

---

# v5 训练轮预案(下一轮执行;本轮只存档)

| # | 改动 | 出处 | 触发条件 |
|---|---|---|---|
| 1 | **选项名体系增广**(中性名/随机串按行混入,与现有 4 套语义改写并存) | arXiv:2609.26758 | 改动 1 任一头 FAIL 必做;PASS 也建议做 |
| 2 | **纯 soft CE 对照**(对 teacher 软分布;顺手补 laya#238 <https://github.com/NandhaKishorM/laya/issues/238> 的 `loss=loss_ce/GRAD_ACCUM` 一行 diff);可跑重建参考 = rlcd-lite(GitHub,GRPO+Brier) | Sev <https://huggingface.co/LakoreAI/sev> | v5 训练并行候选,同表对比 |
| 3 | **R-Drop**(两次 dropout KL,λ=0.1 起) | arXiv:2106.14448 | 与 2 同轮消融 |
| 4 | **NOTA/弃权类训练**(conformal 弃权样本 + 双源不一致样本训成显式弃权类;外部证据:Ren et al. 自评 NOTA 优于似然置信(无训练成分)+ Kadavath P(IK) 训练参照;增益以自家消融为准) | arXiv:2312.09300 / arXiv:2207.05221 | 本轮弃权率 >15% 优先上调(参考:90% 名义档实测 29.4%) |
| 5 | **counterfactual 硬约束负例**(回放真实场景,对 known 侧扰动一步造确定标签负例,专补硬约束/否定句) | Web-Shepherd arXiv:2505.15277 | **已触发(v3 4/5、v4 4/5 连续两轮)——v5 必做** |
| 6 | **typed-decisions-v2**:tasksource-jev 2.5M 软标签 + 纯 soft CE + 标签卫生(exact-uniform 行降权 ×0.05)+ **per-language held-out(先 zh)**;数据格式可加 NLI 假设句变体(Verdict 先例) | tasksource-jev;JEV-9B;telepatia/hebrew 先例 | 独立轨道,不阻塞 NLI v5 |
| 7 | 集成 miss 互补分析(v4 头 × typed-decisions 头,零训练先跑互补率) | 一般集成实践;polaris 卡只引「置信门控 + seed 方差报告」 | 任意轮空档 |

**本轮诊断对 v5 的第一优先级提示**:置信压缩带(预核算 §2)意味着「让置信变得可分」比任何后处理
都值钱——soft CE(Sev 线)、温度重训、R-Drop 三件都直接冲这个目标;数量级证据:决策头覆盖率实验
(GitHub「decision-head-rlcd」)显示泛化完全跟训练覆盖走,覆盖不满时 RL 反而伤。

P2 留档(不排期):ArmoRM 式多头门控、AgentRewardBench 四维标签、`/v1/systemone` 兼容端点、
ONNX 导出、Ollaya 式打包渠道。

---

# 生态调研存档 A(HF 侧;2026-09-28,引用必带链接)

## A. Jev 本体(TypeSafe,赛道定义者)

- 2026-09-15 发布,创始人 Diogo Almeida;<https://typesafe.ai/blog/introducing-system-one-models-and-jev>;
  HN <https://news.ycombinator.com/item?id=49717558> **1,984 分 / 520 评论(HN API 实测)**
- 不生成文本,只返回 noul / choice / score 三类决策(与 laya 同构),返回类型化概率,宣称零幻觉+逐决策置信
- 定价 $42/B input tokens(output 免费);官方自报 193.6x faster / 444.6x cheaper
- **架构/权重/论文全不公开**;官方名词 RLCD(Reinforcement Learning for Calibrated Decisions)。
  社区共识(r/LocalLLaMA 质疑帖 <https://www.reddit.com/r/LocalLLaMA/comments/1woe70t/>,本机链路超时未复核):
  encoder+分类头,无隐藏护城河

## B. 直接同类开源模型(官方 typed-decisions test 口径,400 cases / 2000 decisions)

| 模型 | 规模/底座 | acc | ECE/Brier | 要点 | 链接 |
|---|---|---|---|---|---|
| convaiinnovations/laya-typed-decisions(官方) | 421M ModernBERT | 0.766 | 0.213 | RLCD 配方 | <https://huggingface.co/convaiinnovations/laya-typed-decisions> |
| **我们 typed-decisions-multilingual** | 322M mmBERT | **0.7875** | 0.159 | 比官方 +2.1 分;per-language 欠账 | <https://huggingface.co/Modusnsus/laya-typed-decisions-multilingual> |
| LakoreAI/sev | 421M laya | **0.7885** | Brier 0.0495 / NLL 0.8581 | 纯 soft CE 打败 RLCD,见 C-1 | <https://huggingface.co/LakoreAI/sev> |
| manjunathshiva/opendecider-nano | 400M ettin | **0.796** | 0.092 | specialist 最优,17 ms/L40S | <https://huggingface.co/manjunathshiva/opendecider-nano> |
| codepawl/tacet-sonata | 144M mmBERT-small | 0.7625 | 0.095 | 211 vs 14.9 req/s | <https://huggingface.co/codepawl/tacet-sonata> |
| whadupapp/goff-lite | 0.6B Qwen3 | 0.766 | 0.096 | listwise RLCD 改良 | <https://huggingface.co/whadupapp/goff-lite> |
| jaredpalmer/kev-4b | 4B Qwen3.5 | 0.803(locked) | 0.013(域内) | 预注册 + 锁测试集模范生 | <https://huggingface.co/jaredpalmer/kev-4b> |
| autotrust/JEV-9B | 9B 蒸馏 Jev | 教师 90-96% | **0.0007** | 标签卫生:exact-uniform 行降权 ×0.05 | <https://huggingface.co/autotrust/JEV-9B> |
| Mapika/decider-2b | 2B Qwen3.5 | JevBench 0.577 | 0.175 | SFT v1–v8 + 校准感知 RL | <https://huggingface.co/Mapika/decider-2b> |
| damianborek/polaris-1/2/3 | LoRA @ Bespoke-Nimble-9B | v9 = 94 条真实包:seed P3 75/94、P2 73/94、P1 61/94 | — | 双轨决策;0.8 门控非保证(3 条 ≥0.96 unsafe);按 seed 报均值±方差。⚠️ 卡上没有「保留 Jev miss」「集成 0 miss」等文本,勿引用 | <https://huggingface.co/damianborek/polaris-3> |
| meraGPT Decider 1(闭源) | — | 0.768 | Brier 0.052 | leaderboard 榜首 | <https://meragpt.com/models/state-decider-1> |
| TypeSafe Jev 1.13.0(闭源) | — | 0.727 | 0.148 | 官方数据集卡实测(非官网自报) | <https://huggingface.co/datasets/LocalLLaMA/typed-decisions> |

## C. 生态四条主线

1. **RLCD vs 纯 CE 之争**:Sev 研究(minhleduc「Dissecting RLCD」):Laya 官方 RLCD 的 RL 项
   ≈ 噪声平滑的 CE 梯度;teacher 软分布上直接 soft CE 在所有 proper scoring 指标更优(0.7885 vs 0.766)。
   ⚠️ 附带:gold 98.4% 是 teacher argmax 时 hard-label ECE 无效——直接影响 v5 ECE 门槛设计。
   ⚠️ **命名撞车**:检索 RLCD 会撞上 `facebookresearch/RLCD`(Contrastive Distillation 对齐那条线),
   与 TypeSafe 的 RLCD 无关,引用时注明。
2. **蒸馏线**:SargeDev/jev-distill-corpus-v3 74 万行 Jev 输出;JEV-9B 蒸到 ECE 0.0007。
3. **校准是主战场**:官方 0.213 / goff-lite 0.096 / opendecider 0.092 / MacJev 0.032 /
   JEV-9B 0.0007——**至少四种尺子,只列存在性不排名**;手段都是 per-type 温度、Platt、每类偏置。
4. **独立测量文化**:Luni/laya-jev-benchmark 揭穿官方「+16%」是跨基准假对比;kev 系列预注册+锁测试集。
   对外引用必须同基准;我们 0.7875 与 leaderboard 0.727 同口径,可用。

## D. 相邻生态可抄技术(附出处)

- **选项名极性 bug**:arXiv:2609.26758(0/1 改名 no/yes → 每百题翻转 70.4 条,95% CI [67.6,73.1],
  AUC .94→.23;中性名无影响;极性效应 ≥7.4× 中性;mean-pooling 翻转少 4.1×;hosted Jev 同样中招。
  缓解:选项名换随机字符串,精度不掉) → 治理 = 训练名字增广 + 推理前诊断(改动 1)
- **弃权/共形**:RAPS split-conformal <https://arxiv.org/abs/2009.14193>(代码 aangelopoulos/conformal_classification);
  Ren et al. <https://arxiv.org/abs/2312.09300>(自评 NOTA 优于似然置信,无训练成分);
  Kadavath P(IK) <https://arxiv.org/abs/2207.05221>(训练式 NOTA 参照);Guo et al. <https://arxiv.org/abs/1706.04599>
  (vector/matrix scaling)
- **counterfactual 负例挖掘**:Web-Shepherd <https://arxiv.org/abs/2505.15277>(扰动一步造确定标签负例)
- **软标签与多头**:ArmoRM <https://arxiv.org/abs/2406.12845>;Qwen2.5-Math-PRM 软标签
  <https://arxiv.org/abs/2501.07301>(《The Lessons of Developing PRMs》);R-Drop <https://arxiv.org/abs/2106.14448>
- **轨迹评估标签学**:AgentRewardBench <https://arxiv.org/abs/2504.08942>
- **大规模数据源**:tasksource-jev 2.5M 行/670 来源/300+ 任务族(人工投票份额做软标签)
  <https://huggingface.co/datasets/tasksource/tasksource-jev>
- **多语言补课先例**:telepatia-ai/laya-pt-es-typed(per-language held-out 0.807);
  RoeiG/laya-hebrew(NLLB-200 翻译增广)
- **分发**:`/v1/systemone` wire format 成事实标准;Ollaya 式打包;laya 系已有 GGUF/MLX/CoreML/ONNX 端口

---

# 生态调研存档 B(GitHub 侧;2026-09-28,实测,gh API 校准)

## B1. 母体与移植/运行时

- **`NandhaKishorM/laya`(本仓库上游)**:26,555★ / 2,312 forks / Apache-2.0 / 224 open issues;
  multi-language 非自回归决策引擎,router 按请求选 checkpoint
- 移植/运行时(全部在 gallery 之外,stars 实测):mizorewww/laya-mlx 6.5k · mizorewww/laya-coreml 1.5k ·
  receptron/laya(Node/ONNX)505 · ipenywis/laya-ultrafast 215 · FluidInference/FluidUse 203 ·
  wdobry/laya-playground 166 · lkarlslund/laya.cpp 99 · 1Panel-dev/laya-server 75 · 0xBakeer/arbiter 34 ·
  afshinm/laya-mps 22 · apiplant/laya-rs 9 · chneau/docker-laya 4 · codejunkie99/keel(macOS 工作台)285

## B2. 他人 Laya 后训练(与我们同类)

- 语言线:nafi-ullah 孟加拉电商语音(**微调的就是 laya-multilingual**)· shamspias/laya-bangla ·
  telepatia PT/ES(HF)· hebrew(HF)——语言线做的人很少,我们属早期梯队
- 视觉线:r33drichards/laya-vision(60★,换 SmolVLM-256M 骨干,201M,三题型全保留)
- 域线:Talles64/S1-chess · lab-emi/ChipLaya(电路拓扑)· Aurumdev952(临床)· caiovicentino/eikos(金融 4B/27B,MIT)
- **Agent 安全/记忆线(离我们最近)**:Gowtham gyra(421M,判断 destructive commands)·
  Mr-Neutr0n/laya-session-guard(prompt-injection+动作授权)· ganeshdipdumbare/laya-memory
  (**跨会话记忆门控用本地 laya 分类器**)· 0xSarnavo/laya-coding-router · glukicov/laya_router

## B3. 决策模型与工具(可抄点)

| 项目 | 要点 | 落点 |
|---|---|---|
| Heman10x-NGU/openJev-verdict-2.0(290★) | 151M,自报 77.10% / Brier 0.0636 / ECE 0.0144(typed-decisions 口径,自家 harness);v1.4 是**纯推理修复**:候选改写成 NLI 假设句、context 1024→512 | 改动 1 附加观察;v5 标签措辞/数据格式 |
| Astro-Han/decision-head-rlcd(6★) | 3-seed 控制实验:泛化完全跟训练**覆盖率**走,覆盖不满时 RL 反而伤 | v5 预案 1/5 依据 |
| arnabgho/rlcd-lite | GRPO+Brier 奖励 + 校准评测,notebook 级重建(README 有 RLCD 命名撞车警告) | v5 预案 2 参考实现 |
| smkrv/jev-calibrate(32★) | 阈值标定的 holdout 复用 ledger / verdict 分级 / 稳定重跑 | 与我们「拟合/报数不相交」纪律互证 |
| jaredpalmer/kev(7.4k★) | dev/test 冻结、按 dev 选 checkpoint、test 只读一次 | v5 验收方法论引用 |
| bespokelabsai/nimble(1.9k★) | 对比式数据构造(改一个事实让答案翻面)+ 公开 2,676/324 拆分 | v5 数据构造方法 |
| nokia AnyJev(816★) / wfzyx/von(723★) / Mapika decider / FLock this-that-model | 同赛道邻居 | 存在性记录 |

## B4. 索引与注意

- **awesome-jev-gallery(464★,Jev 中心)**:条目质量好但**Laya 侧残缺**——上游 `NandhaKishorM/laya`
  在里面 0 提及,移植仓大多缺席;当「Jev 官方宇宙」索引用,别当 Laya 全集。其它 awesome:
  yibie(1.8k)/v-modal(726)/cobanov(417)/valentynkit(171)/kraayenjon(151),侧重不同
- 外部数字引用纪律同 A 节:链接+基准,两者缺一不许写

---

## 汇报格式(照 V3 结构;数字已按 99 修正)

```
第 4 轮验收汇报(v4 权重不动,SHA256 e9e1c4a5…b13758;零训练轮)
极性诊断:NLI real 20:<n> vs <n>(中性)/<n>(随机);diag 14:<n>/<n>/<n>;val_soft 错例 <n> vs <n> vs <n>
          multilingual choice:翻转 <n>/<N>(门槛 ≤2%),McNemar p=<x>;基线复跑 <acc>(对照卡面 0.7875)
vector scaling:预检矩阵 <FAIL→负结论|PASS>;若 PASS:acc=1 集 |1−mean_conf| <x>→<x>,Brier/NLL <x>→<x>,
          主 val Brier/NLL <x>→<x>(ECE 卫生检查另附)
conformal:val_soft 覆盖 <x>(名义 90%);主 val 捕捉 <n>/99(门槛 ≥50)@弃权 <x>%(门槛 ≤35%);
          AURC <x>;组合规则审计:0.92 档 <x>%/<x>% → 0.9220 档 <x>%/<x>%;逐条清单另附
模型卡同基准对比节:已加/未加(原因);τ 未动(1.2176);红线 0.92 未动
```

后面附:交付物路径、已知代价与残留、预案触发状态、与预核算数字的出入(若有,逐条说明)。

### 交付物清单

- 脚本:`kaggle_eval/option_polarity_check.py`、`conformal_abstain.py`、(`vector_scaling.py`,仅预检 PASS 时需拟合版)
- `data_local/`:极性对话逐条 JSON(含 multilingual 翻转清单与复跑基线)、conformal_v4.json(阈值/覆盖/AURC/被弃错例逐条)、
  vector_scaling_v4.txt(预检矩阵+曲线+结论,负结论也要交)、(`noul_logits_v4.json` 仅 PASS 时)
- 两张模型卡的「同基准对比」节(HF 端同步)
- 本文件末尾补「第 4 轮执行结果」一节(照 V3 写法:门槛对照 + 未过项如实 + 现场记录)
- 被否决的候选/中间产物一律清掉,别留一堆

## 执行注意(只列新增;Kaggle 三坑、b64 内嵌等看 HANDOFF_NLI_V2.md)

0. **现场恢复记录(写给复核方)**:2026-09-28 02:12,V2/V3 文档、全部脚本与 data_local 数据已在盘
   (01:55 排查时发现缺失,目录 mtime 曾停 01:55)。完整性事实:**被跟踪文件与 HEAD 逐字节一致
   (`git status` 除 `HANDOFF_NLI_V4.md` 未跟踪外无任何 diff)**;`kaggle_eval/nli_data/` 整个目录在
   `.gitignore`(第 86 行),其恢复来源 = Kaggle 数据集 `daphnelaurent/nli-conflict-pairs` v9,不在 git 保护网内。
   若再消失:被跟踪文件 `git -C D:/laya checkout -- kaggle_eval`;nli_data 从 Kaggle 回拉。
   汇报前保留 mtime 证据,恢复原因在「执行结果」如实记。
1. 本轮不碰 Kaggle kernel(零训练);multilingual 侧 400 cases CPU 推理即可(322M),超时再切 Kaggle
2. 随机串 seed 固定并写进报告;所有脚本留 assert(行数/不相交/口径)
3. 改动 2 与改动 3 **共用 val_soft 全 300 作拟合面**、共用报数集(主 val + realtest + diag),seed 一致
4. HF 直连不稳走 `HF_ENDPOINT=https://hf-mirror.com`;外部链接逐条可点
5. `kaggle_eval/rl_agent_config.json/` 是 9/25 的野**目录**(内含 laya_finetuned_typed_decisions/),
   v4 真配置(τ=1.2176)在 checkpoint 目录的 rl_agent_config.json **文件**里——不删不碰,汇报点名即可

---

<!-- 执行完成后在此补「第 4 轮执行结果」节(照 V3 写法) -->
