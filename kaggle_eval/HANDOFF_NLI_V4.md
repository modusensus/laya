# Laya 决策头 — 第 4 轮:生态调研落地 + 校准层/诊断任务书(权重不动)

> ⚠️ **本文件已被 `HANDOFF_NLI_V4.1.md`(重写版)替代**,差异摘要见该文件头部。本修补稿仅
> 留档对照,**冲突时一律以 V4.1 为准**。已知作废点(勿照本文件执行):①主 val 错例 = **99**
> (本文件沿袭 V3 的「98」是笔误,凡 /98 分母判据全废);②conformal 主判据改为
> 「捕捉 ≥50/99 且弃权 ≤35%」(本文件「≥50 @ ≤20%」经预核算证明不可达);③「三集 ECE ≤0.04」
> 门槛在 v5 口径下空转,已废;④改动 2 预检升格为「类 × 集」矩阵判据。

> 命名说明:文档名按「交接轮次」编号,与产物版本错位是常态。本文件是第 4 轮:
> **v4 权重保持不变(SHA256 e9e1c4a5…b13758),本轮零训练**,产出 = 三项配置层/诊断交付 +
> v5 训练轮预案。调研背景全文存档在「生态调研存档」节,后续轮次直接引用本节,不用回聊天记录。

## 任务一句话

v4 头已交付并发布 HF,分类门槛全过、ECE 两项因「acc=1.0 撞 0.92 置信天花板」未过(结构性,
非退化);本轮把外部生态调研的可抄点落地:**①选项名极性诊断(两个头)②per-class vector
scaling——降级为离线重标定实验(runtime 不支持纯配置实现,依据见改动 2)③split-conformal
弃权层(给插件写入阈值一个可审计的边界)**。三项全部零训练、零 runtime 改动;v5 数据/配方
改动(软标签、counterfactual 负例、R-Drop、NOTA)写成本文件预案节,下一轮执行。

## 必读前置(不重复,自己去看)

| 文件/节 | 为什么必读 |
|---|---|
| `kaggle_eval/HANDOFF_NLI_V3.md` 全文 | v4 执行结果、温度扫描全表、「v5 门槛重写」验收意见——本轮门槛直接沿用该结论 |
| `kaggle_eval/HANDOFF_NLI_V2.md` | 环境、凭据、Kaggle 三坑、插件接入红线(「v4 待办」第 4 节:自动写入阈值不得 >0.92) |
| `kaggle_eval/temperature_sweep.py` + `data_local/temperature_sweep_v4.txt` | 单一 τ 死局的实测曲线,vector scaling 是它的替代方案 |
| `kaggle_eval/label_swap_check.py` + `data_local/label_swap_v4.json` | v4 已有的标签面交换轴;本轮极性诊断是它的**扩充**而非重复(见改动 1 说明) |
| `kaggle_eval/calib_report.py` | 分切片 ECE 报告,本轮所有校准数字从这里出 |
| `laya/agent.py`(`_decode_answers`,:768 附近)+ `laya/common.py`(温度分桶) | 改动 2 定性依据:温度粒度=选项数桶、decode 无 bias 项、clamp [0.5,5.0]——**别再把 vector scaling 当配置层** |
| 生态调研存档(本文件下方) | 所有外部论文/HF 链接与数字的出处;引用外部数字必须带链接 |

## 起点状态(v4 实测,别重算;数字来自 HANDOFF_NLI_V3.md「v4 执行结果」)

- v4 头:主 val 0.901/ECE 0.0192;偏置诊断 13/14(A1 两方向 0.9225/0.9225);真实 case
  老 20 条 **20/20**、新 10 条 10/10、否定句 4/5;val_soft 297/300(holdout 97/100);
  zh 0.986 / en 1.000
- ECE 未过项(val_soft 0.0703 / realtest 0.0765)= acc 满分撞 0.92 天花板的结构性结果;
  温度扫描:**全网格无单一 τ 同时满足三集**(主 val 要 τ≳1.2,realtest/val_soft 要 τ≲0.8,
  最优解差约 2×);配置保持 τ=1.2176
- v5 门槛重写方向(用户已确认):**只在 acc<1 的切片评 ECE/|acc−mean_conf|**,acc=1 切片
  报 (acc, mean_conf) 二元组;0.92 天花板作为独立跟踪量;不放宽绝对数
- typed-decisions multilingual(另一头,官方 test 400 cases/2000 decisions):
  acc 0.7875(choice 0.752 / noul 0.862 / score 0.759),ECE 0.159,
  温度 choice 1.11 / score 1.05 / noul 1.20;**per-language held-out 评测仍欠着**(模型卡自认)
  ⚠️ **数据口径**:本机 `data_local/typed_decisions_test.jsonl` 只有 **100 cases(≈500
  decisions)**,是 spot 子集;0.7875/0.752 是**官方 400 cases/2000 decisions** 口径。
  改动 1 的 multilingual 侧一律从 HF `LocalLLaMA/typed-decisions`(config `all`,test split)
  拉全量 400 cases,本机 100-case 文件只许冒烟不许出验收数字;直连不稳走
  `HF_ENDPOINT=https://hf-mirror.com`
- 生态定位:0.7875 处于 specialist frontier(0.7885–0.796)与官方 laya-td(0.766)之间;
  ECE 0.159 落后第一梯队(0.092–0.096)——校准是短板

---

# 生态调研存档(2026-09-28,三路网络调研;引用必带链接)

## A. Jev 本体(TypeSafe,赛道定义者)

- 2026-09-15 发布,创始人 Diogo Almeida:<https://typesafe.ai/blog/introducing-system-one-models-and-jev>;HN 1,984 分
- 不生成文本,只返回 **noul / choice / score** 三类决策(接口与 laya 同构),返回类型化概率,宣称 "zero hallucinations" + 逐决策置信度
- 定价 **$42/B input tokens**(output 免费);官方自报 193.6x faster / 444.6x cheaper(社区流传的 "200x/100x" 反而保守)
- **架构/权重/论文全不公开**;官方只给了名词 RLCD(Reinforcement Learning for Calibrated Decisions)。社区共识(r/LocalLLaMA 质疑帖 <https://www.reddit.com/r/LocalLLaMA/comments/1woe70t/>):encoder+分类头,无隐藏护城河

## B. 直接同类开源模型(官方 typed-decisions test 口径,400 cases / 2000 decisions)

| 模型 | 规模/底座 | acc | ECE/Brier | 要点 | 链接 |
|---|---|---|---|---|---|
| convaiinnovations/laya-typed-decisions(官方) | 421M ModernBERT | 0.766 | 0.213 | RLCD 配方;软 acc 落后 Jev(0.471 vs 0.580) | <https://huggingface.co/convaiinnovations/laya-typed-decisions> |
| **我们 typed-decisions-multilingual** | 322M mmBERT | **0.7875** | 0.159 | 比官方 +2.1 分;per-language 欠账 | <https://huggingface.co/slow-stack/laya-typed-decisions-multilingual> |
| LakoreAI/sev(= minhleduc CE 复刻) | 421M laya | **0.7885** | Brier 0.0495 / NLL 0.8581 | 纯 soft CE 打败 RLCD,见 C-1 | <https://huggingface.co/LakoreAI/sev> |
| manjunathshiva/opendecider-nano | 400M ettin 编码器 | **0.796** | 0.092 | 10+ 数据集混合;17 ms/L40S,当前 specialist 最优 | <https://huggingface.co/manjunathshiva/opendecider-nano> |
| codepawl/tacet-sonata | 144M mmBERT-small | 0.7625 | 0.095 | 与官方打平但 211 vs 14.9 req/s | <https://huggingface.co/codepawl/tacet-sonata> |
| whadupapp/goff-lite | 0.6B Qwen3 | 0.766 | 0.096 | listwise RLCD 改良派(proper-scoring reward 为主) | <https://huggingface.co/whadupapp/goff-lite> |
| jaredpalmer/kev-4b | 4B Qwen3.5 | 0.803(locked hard) | 0.013(域内) | 预注册评测 + 锁测试集的模范生 | <https://huggingface.co/jaredpalmer/kev-4b> |
| autotrust/JEV-9B | 9B 蒸馏 Jev 1.13 | 教师 90-96% | **0.0007** | 标签卫生:exact-uniform 占位行降权 ×0.05 | <https://huggingface.co/autotrust/JEV-9B> |
| Mapika/decider-2b | 2B Qwen3.5 | JevBench hard 0.577 | 0.175 | SFT v1–v8 + 校准感知 RL;4 ms/req | <https://huggingface.co/Mapika/decider-2b> |
| damianborek/polaris-1/2/3 | LoRA @ Bespoke-Nimble-9B(Qwen3.5-9B 底座) | v9 = **94 条真实 orchestrator 包**(三个前沿模型多数投票为标签):released seed P3 **75/94**、P2 73/94、P1 61/94;P3 对 v5–v8 全对 | — | coding-agent「conductor」双轨决策(decision: STOP/ASK/DISPATCH;manager: ACCEPT/VERIFY/REJECT/REOPEN/ESCALATE);**0.8 置信门控非保证**(3 条 ≥0.96 仍 unsafe);按 8/4 个训练 seed 报均值±方差。⚠️ 卡上**没有**「保留 Jev miss 作基准」「集成 0 miss」等文本,勿引用 | <https://huggingface.co/damianborek/polaris-3> |
| meraGPT Decider 1(闭源) | — | 0.768(zero-shot SOTA) | Brier 0.052 | HF 数据集卡 leaderboard 榜首 | <https://meragpt.com/models/state-decider-1> |
| TypeSafe Jev 1.13.0(闭源) | — | 0.727 | 0.148(同一 leaderboard) | 官方数据集卡实测,非官网自报 | <https://huggingface.co/datasets/LocalLLaMA/typed-decisions> |

## C. 生态四条主线

1. **RLCD vs 纯 CE 之争**。Sev 研究(minhleduc「Dissecting RLCD」):Laya 官方 RLCD 的 RL 项
   ≈ 噪声平滑的 CE 梯度,直接对 teacher 分布做 soft CE 在所有 proper scoring 指标上更优
   (0.7885 vs 0.766);goff-lite 赢在 proper-scoring reward 混合而非 RL 本身。
   ⚠️ 附带:**gold 98.4% 是 teacher argmax 时 hard-label ECE 无效**——直接影响 v5 ECE 门槛设计。
2. **蒸馏线**。SargeDev/jev-distill-corpus-v3 74 万行 Jev 1.13 输出
   (<https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3>);JEV-9B 蒸到 ECE 0.0007。
3. **校准是主战场**。各家报点:官方 laya-td **0.213**(官方 test,10-bin max-prob)、goff-lite
   **0.096**(自家 held-out)、opendecider-nano **0.092**(自家 general-decision harness)、
   MacJev **0.032**(4K 长输入集)、JEV-9B **0.0007**(蒸馏集 test_set_30k)
   (<https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya>)——⚠️ **这些数字来自至少
   四种不同的尺子,不可横向比**,此处只列存在性不列排名;手段都是 per-type 温度、Platt、
   每类偏置。
4. **独立测量文化**。Luni/laya-jev-benchmark 揭穿 laya 官方 "+16%" 是跨基准假对比
   (83.8% vs 67.8% 来自两个不同基准,<https://huggingface.co/datasets/Luni/laya-jev-benchmark>);
   kev 系列预注册 + 锁测试集。**对外引用 Jev/他卡数字必须同基准**,我们模型卡的 0.7875 与
   leaderboard 的 0.727(Jev)同口径,可用。

## D. 相邻生态可抄技术(附出处)

- **选项名极性 bug**:arXiv:2609.26758「Type-Safe Is Not Error-Free: A Constrained Decision
  Head Follows the Option Name, Not the Rubric Bound to It」(<https://arxiv.org/abs/2609.26758>)。
  选项名与 rubric 解绑实验:0/1 改名 no/yes → 每百题翻转 **70.4** 条(95% CI [67.6, 73.1]),
  AUC **.94 → .23**;中性名无影响;极性效应 ≥7.4× 中性对照;mean-pooling 架构翻转少 4.1×;
  hosted Jev 同样中招(AUC .8146 → .5806,翻转率 24× 重测底噪)。**缓解:选项名换随机字符串,
  精度不掉**。→ 治理手段 = 训练加名字体系增广 + 推理前诊断
- **弃权/共形**:RAPS split-conformal(<https://arxiv.org/abs/2009.14193>,代码
  <https://github.com/aangelopoulos/conformal_classification>)给带有限样本覆盖保证的弃权集;
  Ren et al.(<https://arxiv.org/abs/2312.09300>)证明**自评分数(可含 None-of-the-above 选项)
  优于基于似然的置信度**——注意这是自评/prompt 层机制,**无训练成分**;训练式 NOTA 头的参照
  另见 Kadavath P(IK)(<https://arxiv.org/abs/2207.05221>),对本头的增益以 v5 消融实测为准,
  不引外部数字背书;Guo et al.(<https://arxiv.org/abs/1706.04599>)的 vector/matrix scaling
  是单全局 τ 的标准替代
- **counterfactual 负例挖掘**:Web-Shepherd(<https://arxiv.org/abs/2505.15277>)回放 gold
  轨迹、扰动一步造确定标签 hard negative(WebPRMCollection ~40K 对),7B 验证器打赢 GPT-4o、
  WebArena +26%
- **软标签与多头**:ArmoRM 19 个回归头 + 输入条件门控,每头独立可解释
  (<https://arxiv.org/abs/2406.12845>);Qwen2.5-Math-PRM 用 MC 采样比例做软标签 +
  双源一致性过滤(<https://arxiv.org/abs/2501.07301>《The Lessons of Developing Process
  Reward Models in Mathematical Reasoning》);R-Drop 两次 dropout 前向的 KL,
  encoder 上几乎免费(<https://arxiv.org/abs/2106.14448>)
- **轨迹评估标签学**:AgentRewardBench 四维标签(success / side effects / unnecessary
  repetition / obstructed;<https://arxiv.org/abs/2504.08942>),判官错误学:agreement bias
  (偏信 agent 自报成功)
- **大规模数据源**:tasksource-jev **2.5M 行、670 来源、300+ 任务族、真实人工投票份额做软标签**
  (<https://huggingface.co/datasets/tasksource/tasksource-jev>)
- **多语言补课先例**:telepatia-ai/laya-pt-es-typed(PT/ES per-language held-out 0.807,
  <https://huggingface.co/telepatia-ai/laya-pt-es-typed>);RoeiG/laya-hebrew(NLLB-200
  翻译增广,<https://huggingface.co/RoeiG/laya-hebrew>)
- **分发**:`/v1/systemone` wire format 成事实标准(jebadiah/JevK5/wev/od1/Decida 兼容);
  Ollaya 做 Ollama 式打包(<https://huggingface.co/ollaya-dev>);laya 系已有 GGUF/MLX/CoreML/ONNX 端口

---

# 本轮改动清单(全部零训练;Python 一律 `D:\Miniconda\envs\laya-ft\python.exe`)

### 改动 1(诊断,必做):选项名极性检查 —— 两个头都跑

新脚本 `kaggle_eval/option_polarity_check.py`,照 `label_swap_check.py` 的骨架改。

**与 v4 已有工作的关系,别绕进去**:v4 的「标签面改写增广」(4 套 labels/instructions 轮换)
+ `label_swap_check.py` 治的是**取值措辞**轴(冲突↔兼容互换,实测差 0,已通过)。
arXiv:2609.26758 指出的是另一根轴:**名字体系本身的语义**(0/1 vs no/yes vs 随机串)。
v4 训练时轮换的是 4 套**有语义**的说法,模型仍可能依赖「这词在语料里的极性」。

- 对 **v4 NLI 头**:在 real 20 + 偏置诊断 14 例 + val_soft 300 上,把 labels 换成
  ①中性名(`A/B`)②随机串(每条随机,seed 固定)两版,与标准版逐条对比
- 对 **typed-decisions multilingual**:**数据源 = HF `LocalLLaMA/typed-decisions`
  config `all` 的 test split,全量 400 cases / 2000 decisions**(见起点状态⚠️;本机 100-case
  文件只许冒烟)。choice 题型做 ①中性 ②随机串 两版(论文原文协议:问题、state、rubric
  全不动,只换名),输出**逐题翻转清单**
- 顺带报一个架构观察:论文发现 mean-pooling 翻转少 4.1×——记录两头各自的 pooling 类型进报告

**门槛**:
- NLI 头:real 20 + diag 14,随机串版与标准版**差 ≤1 条(双向)**;val_soft 300
  **错例数变化 ≤1**
- multilingual choice(400-case 全量口径):**翻转总数 ≤2%**(choice ≈800 decisions)且
  McNemar 检验翻转方向不对称性 **p>0.05**,附翻转清单——不用 acc 差值判(n=800 时
  差值判据太钝,翻转计数才能抓到「谁翻成了谁」)
- 超了 = 模型在读名字极性,v5 必须加名字体系增广(预案节),且本轮如实报 FAIL,
  不许用「swap 过了」搪塞——那是另一根轴。

### 改动 2(离线实验,必做):per-class vector scaling —— 降级为离线重标定

**定性(复核 B1,照改)**:本改动**不是**纯配置层。实读 runtime:`laya/agent.py` +
`laya/common.py` 的温度粒度 = 题型 × 选项数桶(`temperature_by_options` 实为「选项数分桶」,
noul 恒为 `noul:2`,**无 per-class 粒度**);decode 全路径只有 `z = logits[r,:k] / t_scale`
一句(`agent.py:768`),**没有任何 bias 项**;温度被 clamp 到 [0.5, 5.0]。即 per-class
(τ_c, b_c) 必须动 `_decode_answers`(代码变更)或出 runtime 之外的后处理。且「argmax 不变」
只对单 τ 成立——**per-class τ 能改变 argmax**。因此本轮:**只做离线重标定实验,不改
runtime、不产出接入配置**;若预检+实验证明值得上,最小 runtime diff(附回退说明)排进
v5 轮;clamp [0.5, 5.0] 对部署可行性的约束写进报告。

**预检(先做,约 5 分钟,复核 B3)**:per-class 能否解跨集死局是**待验证假设**,不是机制
——「主 val 要软化、realtest 要锐化」是集合间冲突,类不对称未必映射得过来。在 val_soft 300
与主 val 1000 上分 true/false 两类各画「置信–正确率」曲线:若两类**同向漂移**(都需软化或
都需锐化),per-class 自由度无戏,改动 2 以**负结论报告收尾——负结论也算完成,不算失败**;
只有两类反向漂移才继续拟合。

**拟合面(复核 C3,照改)**:val_soft **全 300** 作拟合面(该集仅 ~3 错,切 150 后常见
0–2 错,用 1–2 个错误事件拟 4 参不稳);拟合目标 = **NLL/置信对齐,不追错误样本**(此约定
写进报告);报数集 = **主 val 1000 + realtest 35 + diag 14**,与拟合面不相交(脚本留 assert)。

**验收口径(复核 B2,照改)**:v5 门槛重写后「只对 acc<1 切片评 ECE」会使原「三集
ECE ≤0.04」几乎空转(病灶数字 val_soft 0.0703 / realtest 0.0765 恰是 acc=1 切片的 ECE,
被口径排除)。成功量改为:
- acc=1 集(realtest 若满分):**|1−mean_conf| 较 v4 基线下降 ≥0.02**,且该集 **Brier、NLL
  不劣于基线**(Brier/NLL 不受满分天花板影响)
- 主 val(acc<1):**Brier、NLL 不劣于基线**;ECE 只作卫生检查报告,不设门槛
- 同表对照三行:基线(τ=1.2176)/ 单 τ 最优 / per-class(vector scaling);「本集最优 τ」
  仅作参照列,注明选 τ 切片与报数集不相交

若 per-class 也满足不了:如实报死局,升级为 v5 训练轮议题(预案节),不许放宽任何绝对数。

### 改动 3(弃权层,必做):split-conformal 弃权,给插件写入一个可审计的边界

新脚本 `kaggle_eval/conformal_abstain.py`,RAPS 的 noul 二类简化版(split-conformal 足够,
不需要 rank 惩罚)。

**样本量与判据(复核 C1,照改)**:150/150 切分下 90% 名义覆盖的 95%CI 约 ±5pp,
「≥88% 覆盖」与噪声同量级;realtest 20/20 无错 → 「弃权抓错」在其上不可测;val_soft 余 150
错 0–2 条撑不起倍数判据。据此重写:

- **拟合面**:val_soft 全 300 的置信分数,拟合非一致性分位(目标名义覆盖 90%);
  拟合面本身不承担判据
- **主判据(在主 val 1000 上跑,98 条错,统计有效)**:
  ① **错例捕捉率 ≥50%**(≥49/98 条错落入弃权集),弃权预算 **≤20%**(弃权太多等于不干活)
  ② AURC + 风险-覆盖曲线,与「无弃权基线」「oracle(把真错全弃掉的曲线)」同图对照
  ③ **逐条清单**:被弃权的错例每条列出(编号/known/new/p_true/期望)——原「汇报人话」
  要求升格为正式判据,逐条可点验
- **覆盖数字降为报告项**:val_soft(同分布,名义保证适用)、realtest、diag 各报
  (经验覆盖, 弃权率)即可,不设硬门槛
- **可交换性注记必须写进报告**:val_soft(合成)拟合 → 主 val(MNLI 系)评测是**分布迁移**,
  主 val 上的数字是「迁移表现」而非有限样本保证;可加做主 val 对半切(标定 500 / 评测 500)
  作同分布参照组,标明区别
- **红线不动**:插件自动写入阈值 ≤0.92(HANDOFF_NLI_V2.md「v4 待办」第 4 节)照旧;
  conformal 输出是**附加层**——写入 = p_true 达红线 **且** 未被弃权。不许用弃权层放宽
  既有门槛,不许把 conformal 阈值直接当自动写入阈值

### 改动 4(报告,必做):把生态调研结论转成对外可引用的数字纪律

- 两个模型卡各加一节「同基准对比」:只引用同基准数字(typed-decisions test 上的
  0.7875 vs 0.766 vs 0.727),跨基准对比(如 vs Jev 官网自报)显式标注「不同基准,不可比」
- v4 模型卡补一句极性/校准层说明(改动 1-3 出数后)

## 明确不要做

- **本轮不重训**:不动权重、不动 MNLI 冻结集、不改老 20 条;所有「要不要重训」的结论
  写进预案节,下轮再说
- 不用弃权层/vector scaling 放宽既有分类门槛;**本轮不改 laya runtime 任何代码、不产出
  接入配置**(改动 2 已降级离线;runtime 最小 diff 属 v5 轮)
- 拟合面与报数集不许相交(ECE 自证问题,V3 已踩过一次,脚本里留 assert)
- 引用外部模型/论文数字必须带链接并写明**基准/尺子**;不许把论文数字当成自己的实测;
  引用 polaris 系只准用本文件存档表里的已核数字(卡上没有的文本不许复述)
- 不在本轮引入 tasksource-jev 等外部数据(那是 v5/typed-decisions-v2 的事)
- **不删、不覆盖 kaggle_eval/ 与 data_local/ 的恢复现场**(mtime 证据保留到第 4 轮汇报
  归档);`kaggle_eval/rl_agent_config.json/` 是 9/25 的野**目录**(内含
  laya_finetuned_typed_decisions/),真配置在 checkpoint 的 `rl_agent_config.json` 文件——
  本轮不删不碰,汇报点名即可

---

# v5 训练轮预案(第 5 轮执行;本轮只存档,不动手)

按投入产出排序,每项标注出处与本轮诊断的触发条件:

| # | 改动 | 出处 | 触发条件 |
|---|---|---|---|
| 1 | **选项名体系增广**(中性名/随机串按行混入,与现有 4 套语义改写并存) | arXiv:2609.26758 | 改动 1 任一头 FAIL 则必做;PASS 也建议做(零风险预防) |
| 2 | **纯 soft CE 对照**(对 teacher 软分布,替代/对照 RL 项;顺手补 laya#238 <https://github.com/NandhaKishorM/laya/issues/238> 的 `loss=loss_ce/GRAD_ACCUM` 一行 diff) | Sev <https://huggingface.co/LakoreAI/sev> | v5 训练时并行一个候选,同表对比 |
| 3 | **R-Drop**(两次 dropout 前向 KL,λ=0.1 起) | arXiv:2106.14448 | 与 2 同轮,消融表各一行 |
| 4 | **NOTA/弃权类训练**(把改动 3 的 conformal 弃权样本与双源不一致样本训成显式弃权类;外部证据 = Ren et al. 自评 NOTA 选项优于似然置信(无训练成分)+ Kadavath P(IK) 作训练参照;增益以自家消融为准) | arXiv:2312.09300 / arXiv:2207.05221 | 改动 3 弃权率 >15% 时优先级上调 |
| 5 | **counterfactual 硬约束负例**:回放真实 case 场景,对「已知」侧扰动一处造确定标签负例,专补硬约束类与否定句 | Web-Shepherd arXiv:2505.15277 | 否定句连续两轮 4/5 则必做 |
| 6 | **typed-decisions-v2**(另一头):tasksource-jev 2.5M 软标签 + 纯 soft CE + 标签卫生(exact-uniform 行降权 ×0.05)+ **per-language held-out 评测(先 zh)**,补模型卡欠账 | tasksource-jev;JEV-9B;telepatia/hebrew 先例 | 独立轨道,不阻塞 NLI v5 |
| 7 | 集成 miss 互补分析(v4 头 × typed-decisions 头,零训练先跑互补率) | 一般集成实践;polaris 卡可参照的只有置信门控与 seed 方差报告(卡上无集成/互补数字) | 任意一轮空档做 |

P2 留档(不排期):ArmoRM 式多头门控、AgentRewardBench 四维标签(轨迹头 v2 时再议)、
`/v1/systemone` 兼容端点、ONNX 导出、Ollaya 式打包渠道。

---

## 汇报格式(照 HANDOFF_NLI_V3.md 的结构,5 行数字 + 交付物)

```
第 4 轮验收汇报(v4 权重不动,SHA256 e9e1c4a5…b13758)
极性预检:true/false 两类置信-正确率曲线 → 同向/反向漂移(per-class 假设成立/不成立)
极性诊断:NLI real <n>/20 vs <n>/20,diag <n>/14 vs <n>/14,val_soft 错例 <n> vs <n>;
          multilingual choice 翻转 <n>/<N>(门槛 ≤2%),McNemar p=<x>
vector scaling(离线):acc=1 集 |1−mean_conf| <x>→<x>,Brier <x>→<x>,NLL <x>→<x>;
          主 val Brier <x>→<x>,NLL <x>→<x>(ECE 卫生检查另附)
conformal:主 val 错例捕捉 <n>/98(门槛 ≥50%),弃权 <n>/1000(门槛 ≤20%),AURC <x>;
          覆盖/曲线/逐条清单报告另附
模型卡同基准对比节:已加/未加(原因)
```

后面附:交付物路径、已知代价与残留、是否触发预案节各项。

### 交付物清单

- `kaggle_eval/option_polarity_check.py`、`vector_scaling.py`(离线)、`conformal_abstain.py`
- `data_local/`:极性诊断逐条对比 JSON(含 multilingual 翻转清单)、极性预检曲线数据、
  `vector_scaling_v4.txt`(离线报告,含预检结论/三行对照表/clamp 部署可行性注记)、
  `conformal_v4.json`(阈值/覆盖/AURC/被弃错例逐条清单)
- 两个模型卡的「同基准对比」节(HF 端同步)
- 本文件末尾补「第 4 轮执行结果」一节(照 V3 写法:门槛对照表 + 未过项如实写 + 现场恢复记录)
- 被否决的候选一律删掉,别留一堆

## 执行注意(只列新增;Kaggle 三坑、b64 内嵌等看 HANDOFF_NLI_V2.md)

0. **现场恢复记录(A1 已解,写给复核方)**:2026-09-28 02:12,V2/V3 文档、全部脚本与
   `data_local/` 数据已在盘上(**恢复者不明**——01:55 排查时发现缺失,目录 mtime 曾停在
   01:55;本轮开工前已恢复)。完整性已点验:`git status` 干净 = 被跟踪文件与 HEAD 逐字节一致;
   `train_v4.jsonl` 12000 行、val_soft 300 条、realtest_v4 = 20+5+10 逐项核对通过。
   若再消失:被跟踪文件用 `git -C D:/laya checkout -- kaggle_eval` 拉回;训练大文件被
   gitignore,从 Kaggle 数据集 daphnelaurent/nli-conflict-pairs v9 回拉。
   汇报前保留目录 mtime 证据,恢复原因在「第 4 轮执行结果」里如实记一笔。
1. 本轮不碰 Kaggle kernel(零训练);改动 1 的 multilingual 侧推理在本地 CPU 即可
   (322M 头,官方 test 400 cases 单机 CPU 可承受,超时再切 Kaggle)
2. 随机串 seed 固定并写进报告(可复现是验收底线)
3. 改动 2 与改动 3 **共用 val_soft 全 300 作拟合面**、共用同一套报数集(主 val 1000 +
   realtest 35 + diag 14,与拟合面不相交),seed 一致,避免「两套切法」打架
4. 外部引用链接逐条可点;HF 页面本机直连不稳时用 `HF_ENDPOINT=https://hf-mirror.com`;
   官方 test 数据集同走该镜像拉取
5. `kaggle_eval/rl_agent_config.json/` 是个 9/25 的野**目录**(内含
   `laya_finetuned_typed_decisions/`),v4 真配置(`temperature[2]=1.2176`)在 checkpoint
   目录的 `rl_agent_config.json` **文件**里——本轮不删不碰野目录,汇报点名即可
