# HANDOFF_NLI_V7.md — 第 7 轮任务书(NLI conflict head v7:两缺口形状修复 + 对抗泛化补强 + conformal 规则形态预注册评估)

> 读者 = 执行者(无聊天历史)。**开工前必读**:`HANDOFF_NLI_V2.md`(环境/凭据/Kaggle 三坑/插件红线)、
> `HANDOFF_NLI_V3.md`(温度扫描结论)、`HANDOFF_NLI_V4.1.md`(第 4 轮判据与交付)、`HANDOFF_NLI_V5.md`、
> `HANDOFF_NLI_V6.md`(第 6 轮任务书;**其文末「第 6 轮执行结果」的 4 项 FAIL 与负结论 = 本轮输入**;
> v6 执行结果节的三处小数字已随本轮提交修正:极性行 → real 1,1 / diag 0,0;分箱「单调递减」措辞;supp 桶数 28)。
> 冲突时以本文为准。全部数字可由仓内落盘数据复算;复算口径见 §2.6。执行完毕后,在本文件末尾追加「第 7 轮执行结果」节(承 v5/v6 惯例)。

---

## 0. 本轮定位

v4 仍是已交付头(`D:\laya-kaggle-output\laya-nli-conflict-v4`)。v6(纯 CE 主臂,零句面复用口径)8 项 PASS / 4 项 FAIL,
无交付头。第 6 轮复验(独立复核)数字侧全部复现,并给本轮输入清单:

| # | 事项 | 实测 | 来源 |
|---|---|---|---|
| 1 | 新 10 两 miss 同为「硬约束 + 裸意图」,**但现有 HC 的 intent 形状未覆盖两个子形状**:①约束主体 = 第三方(宠物);②约束来源 = 医嘱/康复 | p=0.283(宠物粮食)/ 0.196(运动伤病史) | V6 §6.4-4 |
| 2 | val_soft 5 错全部在 holdout,且**历轮错误全部集中于 degree+barber 两族**(email/desk_floor 历轮 0 错) | v4:3 / v5-A:1 / v5-B:7 / v6:5 见 §2.2 | 复验逐行回查 |
| 3 | **barber holdout 与训练域重叠(自 v5 起)**:训练含理发域 200+(v5)/215(v6) 行(neg 族 barber + hair 族,共享人名池);degree/email/desk_floor 域检干净 | 见 §2.2 注 | 复验新发现 |
| 4 | 极性 val_soft 变化 +2 = #2 同源(同一组 5 错) | 5/5/3 | V6 §6.2 |
| 5 | conformal 同分布对半切仍 FAIL:maxconf(LAC)类规则抓不住高置信错误(97 错中 52 条 conf≥0.92) | 4/54 @ 2.8% | V6 §6.4-6 |

**本轮目标(优先级从高到低)**

1. **HC 补两族(必做)** —— 修 #1:第三方约束族(pet)+ 医嘱/康复约束族(doctor),裸意图形式(不带违逆标记)。§3.1
2. **alt_praise 形状(必做)** —— 修 #2/#3 的 barber 侧:非 holdout 族补「听闻/刷到替代者好评」false 形状。§3.2
3. **metric 族(默认做;小量)** —— 修 #2 的 degree 侧(数值指标变更 = 冲突的抽象迁移)。§3.3
4. **conformal 规则形态预注册评估(必做)** —— 处置 #5:候选分数 s1/s2/s3,拟合半扫描 → 报数半判定;达标即采用,不达标交负结论。§6
5. **回归守卫** —— 老 20 20/20、否定 5/5、数据卫生 0/35、band ≥8pp、swap 0、bias 13/14、τ/0.92/val/val_soft 零改动:全保持。§5

本轮仍是**训练轮**(纯 CE 主臂);runtime、τ、0.92 红线、`val`(1000)/`val_soft`(300) 全部零改动(§9)。

---

## 1. 范围表

| # | 项 | 本轮 | 依据 / 备注 |
|---|---|---|---|
| 1 | HC 补 pet/doctor 两族(+400,裸意图) | **必做** | #1;V6 §6.4-4 方向:「在 hc/cf 块补『裸意图 + 硬约束』true 形状(意图动词不带违逆标记)」的**两个未覆盖子形状** |
| 2 | alt_praise 形状(+240,≥6 非 holdout 族) | **必做** | #2/#3 |
| 3 | metric 族(+160) | 默认做(§10.2 可浮动) | #2 degree 侧;探索性 |
| 4 | conformal 规则形态评估(§6 协议) | **必做(报告为主;达标即采用)** | #5 |
| 5 | 主臂纯 CE(v6 配方) | **必做(唯一必跑训练)** | 承 v6 |
| 6 | RL 对照臂 / R-Drop | 不做 | 承 v5/v6 决策 |
| 7 | τ / 0.92 红线、multilingual 头 | 不做 | 单独立项 |
| 8 | 数据侧 barber/hair 域改动 | 不做(仅报告口径) | #3:最小改动 = 不动数据、报告双读 |

---

## 2. 预核算

### 2.1 主 val(1000)

| 量 | v4 | v5-A | v5-B | v6(本轮基线) |
|---|---|---|---|---|
| 错例 / acc | 99 / 0.901 | 107 / 0.893 | 98 / 0.902 | **97 / 0.9030** |
| band(q90−q10) | 0.62pp | 1.62pp | 3.96pp | **5.14pp** |
| ≥0.92 占比 / 该档错误率 | 87.1% / 6.66% | 90.8% / 7.49% | 87.1% / 6.20% | 84.4% / 6.2%(52/844) |

v6 分位(q10 0.8951 / q50 0.9379 / q90 0.9465 / q99 0.9543 / max 0.9568,复算口径见 §2.6)。

### 2.2 val_soft(300)与 holdout 逐族(复验逐行回查)

| 轮 | 错误数 | 错在哪(族,p_true) |
|---|---|---|
| v4 | 3 | degree 0.075 / degree 0.925 / barber 0.824 |
| v5-A | 1 | degree 0.12 |
| v5-B | 7 | degree 0.104 / 0.552;barber ×5(0.545/0.551/0.577/0.696/0.836) |
| v6 | 5 | degree 0.16 / 0.34 / 0.491;barber 0.857 / 0.704 |

- **历轮错误全部集中于 degree+barber;email/desk_floor 两族历轮 0 错**。
- **注(复验新发现)**:holdout 定义(「v3 起未见类」)对 **barber** 自 v5 起不再严格成立——训练含理发域
  ~200+(v5)/215(v6) 行(neg 族 barber「找谁剪/换了谁」+ hair 族「发型样式」,与 holdout 切片共享人名池);
  degree/email/desk_floor 域检干净(degree 域 0 行;desk_floor 仅泛化词「工位」18 行,无楼层语义)。
- **处置:不动数据侧(v7 最小改动)**;报告改「**holdout 双读**」:4 类冻结口径 + 剔 barber 的 3 类口径(v4–v7 同表,§5.2)。
- v6 其余:consid 中位置信 0.7882;band 16.29pp(q10 0.7859 / q90 0.9488)。

### 2.3 极性(v6,seed 20260928)

real 0/1/1(diff 1,1)· diag 1/1/1(diff 0,0)· val_soft 5/5/3(diff std↔rnd = 2 ⇒ FAIL,与 §2.2 同源);
mean|Δp|:real .0227/.0375 · diag .0026/.0053 · val_soft .0058/.0100。

### 2.4 conformal(v6,对半切)

报数半 4/54 @ 2.8%(k=451 / qhat 0.3141 / T 0.6859);AURC 0.04345(oracle 0.00491;
v4 0.05564 → v5-A 0.04637 → v5-B 0.04402 → v6 0.04345,三连降);旧 val_soft 拟合口径存档 0.7838(legacy)/ 0.6859(拆半)。
**关键分子**:97 错中 52 条 conf≥0.92 ⇒ 任何「写 = 高置信」型阈值规则最多只能弃权——按 AURC 排序在前 35% 里找错。§6 协议专治。

### 2.5 预期效应(报告须逐条给实测对照)

| 项 | 预期 | 备注 |
|---|---|---|
| 新 10 | 10/10(修两 miss) | pet/doctor 形状直接对位 |
| val_soft | ≤ 3(至少修 2 错) | barber 侧(alt_praise)较有把握;degree 侧(metric)不确定 |
| 极性 val_soft | ≤1 变化 | 随 #2 联动 |
| conformal | 候选达标即采用;否则照 FAIL 交负结论 | §6 |
| 其余 | 全保持 | §5 |

### 2.6 复算口径(承 v6 §2.6)

- 分位数:最接近序号法 `sorted(xs)[round(p·(n−1))]`;conf = max(p_true,1−p_true);p_true≥0.5 记 true。
- 泄漏审计:`python kaggle_eval/leak_audit.py`(默认 v5/v4;传路径可审任意训练文件)。
- conformal:`conformal_abstain.py --main … --soft … --real … --diag … --split-half --out …`。
- holdout 逐族回查:`val_soft_probs_*.json`(含 meta.row_id)↔ `nli_conflict_val_soft.meta.json` 边车;v4 行按行序直连。

---

## 3. 数据改造(`kaggle_eval/gen_soft_conflicts_v7.py`;「照着 v6 改」)

输出 `data_local/nli_conflict_train_v7.jsonl`。**val 与 val_soft 不重生成**(逐字节不变)。
v6 已实现且保留:6-labelset 全行轮换、分级软目标(§3.4)、精确配额池、§3.6 泄漏断言、supp/mention 配额。本轮只做 §3.1–3.3 + 断言。

### 3.1 HC 两族(必做;#1;形状可借,原文不入训)

现有 HC 的 intent 形状已覆盖**自约束**(过敏/忌口/戒烟 + 物品),两 miss 的真缺口 = 两个子形状:

- **pet 族(第三方主体,+200 行)**:约束主体 = 用户家的宠物(猫/狗/鹦鹉…)。
  - 例(须为新字面,不得复刻案例句):known「用户家的狗对牛肉过敏」intent「用户打算给它换牛肉味的零食」/ act「用户昨天喂了它鸡胸肉零食」;
  - intent 动词 = 想/打算/准备/计划/想试试;**不带违逆标记**(「不顾」「偏要」「硬要」不进新字面)。
- **doctor 族(医嘱/康复来源,+200 行)**:约束 = 医嘱(静养/忌剧烈运动/忌负重/忌辛辣/复查前忌口)。
  - 例:known「用户腰肌劳损,医生让卧床休息两周」intent「用户准备周末去爬山」/ act「用户昨天去打了羽毛球」;safe 侧 = 遵医嘱(在家休养/改低强度)。
- 结构照 HC(known / intent / act / safe / unrel);safe 侧进 `hc_safe`、unrel 侧进 `hc_unrelated`;intent+act 进 `hardconstraint`。
- 配额默认:两族各 200,kind 分派建议 70/20/10(intent+act / safe / unrel),逐族计数落报告。
- **双断言**:不得与两验收案(猫粮/伤病史)任何句逐字段等值;人工抽查 2 条新字面与案例「形状同、字面异」。

### 3.2 alt_praise 形状(必做;#2 barber 侧)

- 行定义:**听闻/刷到/同事朋友推荐「替代者」好评**(无用户自身状态变更)⇒ 目标 false(consid 语义)。
- 形如「用户听说{B}手艺不错」「用户刷到{B}的评价挺高」「同事推荐了{B}」——**新字面**;
  分布在 **≥6 个非 holdout 族**(默认从 city / job / gym_brand / coffee / car / cat_food / lang 中选 ≥6;
  **不含 hair/barber 域**,不加深 §2.2 注)。
- 配额默认:+240(每族 ~40),meta 加 `shape='alt_praise'`;kind = consid(0.80±0.05)。
- 护栏:不改冲突形状(「用户现在都找{B}剪」类语义保持);不写成「用户在想要不要换{B}」的复读(该形状已有)。

### 3.3 metric 族(默认做;#2 degree 侧;探索性)

- **可量化指标被改写为新值 ⇒ 冲突**的抽象迁移训练:默认新增非 holdout 小族「体检指标」(体重/血糖/血脂/心率;**不用度数/视力域**),
  或摊入 rent 等既有数值族(§10.2)。
- 四侧齐全(known=A 值 / conflict=上调为新值 / consid=担心涨 / compat / unrelated);+160(默认四侧 ×40)。
- conflict 侧 meta 加 `shape='metric_shift'`;kind 按侧归 true / compat / consid / unrelated 既有档位。
- 报告须给「degree 侧前后对照 + 全 val_soft 对照」以证实/证伪迁移;不得为追该族动门槛。

### 3.4 软目标(承 v6 §3.4 表)

新增字面沿用既有 kind 档位:hardconstraint / hc_safe 0.90±0.03 · hc_unrelated 0.96±0.01 · consid 0.80±0.05 · true 0.93±0.02 ·
compat 0.93±0.02 · unrelated 0.96±0.01。硬断言承 v5/v6:`|p(gold)−0.5|≥0.2`;`gold.label==argmax`;档位封闭。

### 3.5 规模与硬不变量

| 块 | v5 | v6 | v7 |
|---|---|---|---|
| MNLI(整块逐字节不变) | 6000 | 6000 | 6000 |
| 合成软冲突(含 alt_praise +240、metric +160) | 5400 | 5400 | **5800** |
| 硬约束(含新 pet/doctor +400) | 600 | 600 | **1000** |
| 否定算子族 | 800 | 1600 | 1600 |
| known 一步扰动(cf) | 600 | 900 | 900 |
| 合计 | 13400 | 14500 | **15300** |

自检(全部 assert,写在生成脚本尾部;承 v6 §3.5 六条 + §3.6,按上表改数):
1. 行数 = 15300;`state` 全局唯一;与 val / val_soft 零交集。
2. gold 计数按 kind 与上表一致;`true : false` 比例记录(不设硬门槛,但要报)。
3. labelset 配比承 v5/v6(语义 4 套合计 ≥50%、LS5 ≥20%、LS6 ≥20%;每 labelset ≥3 kinds)。
4. 每行 `meta` 完整(`cat/kind/lang/labelset`;新块含 `shape`/族标记)。
5. `gold.label == argmax`;`|p(gold) − 0.5| ≥ 0.2`;档位封闭。
6. 新块逐族计数:pet 200 / doctor 200(含子侧)/ alt_praise ≥6 族合计 240 / metric 160;各块正负比。
7. §3.6 泄漏断言全过(35 案 known/new 逐字段等值 = 0;**不过不许写输出**)。

### 3.6 泄漏断言(防回归;承 v6)

```python
# ---- §3.6 泄漏断言(承 v6;追加在 §3.5 断言之后)----
# 口径:35 条验收句(realtest_v2 CASES/NEGATION_CASES + realtest_v4 NEW_CASES)与训练语料
# known/new 逐字段等值(不做子串 —— 子串会假阳性:某超集句式含验收句不等于同一句)。
import json

from realtest_v2 import CASES, NEGATION_CASES
from realtest_v4 import NEW_CASES

def assert_no_case_leak(rows):
    ks, ns = set(), set()
    for (state, _e), _note in list(CASES) + list(NEGATION_CASES) + list(NEW_CASES):
        ks.add(state['known']); ns.add(state['new'])
    hits = []
    for r in rows:
        st = r['state']
        st = json.loads(st) if isinstance(st, str) else st
        if st.get('known') in ks or st.get('new') in ns:
            hits.append((st.get('known'), st.get('new')))
    assert not hits, f'验收句泄漏 {len(hits)} 行,示例: {hits[:2]}'

assert_no_case_leak(v7_train)   # 变量名以 v7 生成器为准
```

- 完成后跑 `leak_audit.py data_local/nli_conflict_train_v7.jsonl` 复跑 0 命中(输出存 `data_local/leak_audit_v7.txt`)。

---

## 4. 训练改造(承 v6;`train_nli_conflict.py` 同一版本即可)

- **主臂 = 纯 CE(v6 配方)**:EPOCHS=4、MICRO_BATCH=16、GRAD_ACCUM=2、LR 承 v4/v5、SIGMA 0.4→0.1、底座不变
  (`convaiinnovations/laya-multilingual`)。RL 项代码保留、默认不跑。
- CLI:`--expected-train 15300`(防静默截断);kernel `max_items` 抬到 16000。
- 尾部承 v5/v6:400 条 calib(seed 20260927)→ τ 拟合 → 落盘 `model.safetensors`/`encoder`/`tokenizer`/
  `rl_agent_config.json`/`metrics.json`;val 1000 逐行 `{gold, p_true}` 落盘 `val_probs_v7.json`。

**止损线(先写死,执行时照判)**:
- 主 val acc < **0.885** ⇒ 判 FAIL,不再叠改动,原样报数。
- **§3.6 泄漏断言失败 ⇒ 立即停修数据,禁止带泄漏训练**。
- 新 10 <10/10、否定 <5/5、val_soft >3、极性变化 >1 ⇒ 对应轴 FAIL 照实报、逐条列 p 值。

---

## 5. 门槛(承 v6 §5.1,只准加)

### 5.1 硬门槛(全过才算交付)

| 集合 / 指标 | v4 | v6 | v7 门槛 |
|---|---|---|---|
| 主 val acc | 0.901 | 0.903 | ≥ 0.896(不放宽) |
| realtest 老 20 | 20/20 | 20/20 | 20/20 |
| realtest 新 10 | 10/10 | 8/10 | 10/10 |
| realtest 否定 5 | 4/5 | 5/5 | 5/5 |
| val_soft 错误数 | 3 | 5 | ≤ 3 |
| 偏置诊断 | 13/14 | 13/14 | ≥ 13/14 |
| label-surface swap diff | 0 | 0 | 0 |
| 极性 real/diag \|diff\| | ≤1 | real 1,1 · diag 0,0 | ≤1 |
| 极性 val_soft 错误变化 | +2 | +2 | ≤1 |
| 分离度 val_soft band | 0.28pp | 16.29pp | ≥ 8pp |
| conformal(同分布对半切,报数半) | 58/99@29.4%(旧口径) | 4/54@2.8% | 捕捉率 ≥50% @ 弃权 ≤35%(采用 §6 结论规则) |
| 训练数据卫生 | 6/35 案复用 | 0/35 | **0/35**(断言全过 + `leak_audit.py` 0 命中) |

### 5.2 报告项(不设硬门槛,但必须给数)

- 分离度全套 + consid 中位置信(承 v6)。
- **holdout 双读**:4 类冻结口径 + 剔 barber 的 3 类口径(逐族错表,承 §2.2 表式;v4–v7 同表)。
- alt_praise / metric 两对位案复盘:val_soft 5 错逐条前后 p 对照。
- conformal:§6 协议全数(三候选 × 两半曲线、采用判定、旧口径存档对照)。
- 极性 mean|Δp| 三集与 §2.3 对照;τ 拟合值;`leak_audit.py` 输出;新块逐族计数;训练曲线。

### 5.3 交付判定

- 承 v6 §5.3:任一硬门槛 FAIL ⇒ 交付物照出、结论写 FAIL(负结论也算完成);不许调门槛/换判据集/挑样本抹平。
- **口径注(必写进报告)**:v6 起验收 35 案与训练语料零逐句交集;v2–v5 数字带既有句式复用口径;
  barber holdout 与训练域重叠自 v5 起(§2.2 注);跨轮对比按此读。

---

## 6. conformal 规则形态预注册评估(必做;#5)

**目标**:验证「存在可用的行级不确定性信号,使弃权规则在报数半 ≥50% 捕捉 @ ≤35% 弃权」;
达标即采用,不达标交负结论。**不许用报数半/验收集做任何拟合或选择。**

1. **候选分数(固定三个,不得事后加第四个)** —— 脚本新增 `kaggle_eval/conformal_alt_probe.py`
   (本机 CPU 可跑;输出 `data_local/conformal_alt_v7.json`,含三候选逐行分数):
   - `s1` = maxconf(现状 LAC 基线);
   - `s2` = **表面一致性**:同一 state 用 3 个固定渲染(①行原 questions;②中性 A/B;③keep/supersede)各取 p,
     记 `s2 = 1 − (max_r p_r − min_r p_r)`;
   - `s3` = `answers.action.act_probability`(辅助头;方向语义执行侧先核实,报告注明该字段含义与使用方向)。
2. **协议**:沿用 seed 20260928 洗牌对半(与 v6 可比;assert 两半互斥)→ 在 **half1** 上为每个候选扫描阈值
   (目标:capture 最大且 abstain ≤35%)→ 每候选定一个规则变体 → 在 **half2** 上评估 capture/abstain/代入错误率。
   任一候选 half2 上 ≥50% capture @ ≤35% abstain ⇒ **该规则为本轮采用规则**(写进 `conformal_abstain.py` 新模式,
   默认开关;报告给全数含两半与旧口径对照);否则保持现状,负结论照交。
3. **护栏**:half2 与 realtest 35 一律不参与拟合;若做了协议外探索,必须标「探索,不得采用」;报告给出三候选完整曲线。

---

## 7. 交付物与归档

**脚本(进仓,`kaggle_eval/`)**
- `gen_soft_conflicts_v7.py`(含 §3.5 全断言 + §3.6 泄漏断言)
- `conformal_alt_probe.py`(**新**;§6)
- 其余沿用(**不改判据**):`train_nli_conflict.py`、`dump_probs.py`、`leak_audit.py`、`option_polarity_check.py`、
  `realtest_v2.py`、`realtest_v4.py`、`noul_bias_diag.py`、`label_swap_check.py`、`conformal_abstain.py`(仅 §6 采用时加新模式)

**数据产物(`data_local/`)**
- `nli_conflict_train_v7.jsonl`(15300)
- `val_probs_v7.json`、`val_soft_probs_v7.json`、`polarity_nli_v7.json`、`memory_conflict_realtest_v7.json`、
  `noul_bias_diag_v7.json`、`label_swap_v7.json`、`conformal_v7.json`、`conformal_alt_v7.json`、
  `conf_band_v7.txt`、`leak_audit_v7.txt`

**Kaggle 资产**
- 数据集 `daphnelaurent/nli-conflict-pairs` **v12**(train 换 v7;val(1000)/val_soft(300) 冻结不变;`_README` 加 v12 行)
- kernel `laya-nli-conflict-ce` 再 push 一版(计账 = **kernel v12**,纯 CE 主臂);`nli_kernel.py` 同步嵌 v7 脚本(不执行)

**模型产物**:`D:\laya-kaggle-output\laya-nli-conflict-v7\`;大小与 SHA256 写进报告。

**HF(实施口径已拍板)**:**达标才推,且一次推齐** —— 卡(同基准对比节 v4–v7 + 「验收 35 案与训练语料零逐句交集」句 +
新头 SHA)+ 新头权重全文件,一次上传;未达标不动。**不单独补推 v5/v6 存档。**

---

## 8. 执行环境与命令(承 v6 §7)

- Kaggle CLI 的 **tmp 必须指到 ASCII 路径**(如 `C:/kaggle_cfg/tmp`);token 走 `KAGGLE_API_TOKEN` 环境变量,不落盘、不进日志。
- **数据集总挂最新版**;README 改动也要出新版本号,否则 kernel 侧拿到旧版。
- **每次 push kernel 都会从头重训**(CUDA 非确定性 ⇒ 权重不可位复现):本机留档副本才是参照物。
- kernel 的 `QUICK_SAVE` 纯文档版无输出(status ERROR 属占位)。
- 本机 CPU 复现:`D:/Miniconda/envs/laya-ft/python.exe`;`pandas` 被系统策略挡,一律 `pyarrow`。
- 本机→kaggleusercontent 大文件链路间歇断流:重试循环 + 串行下载。
- 泄漏审计:`python kaggle_eval/leak_audit.py [train.jsonl …]`(每次生成后跑,期望 0 命中)。
- **新增**:`conformal_alt_probe.py` 本机可跑(例:`python kaggle_eval/conformal_alt_probe.py <ckpt> <out.json>`;细节自定,
  输出须含三候选逐行分数 + 两半划分与阈值扫描表)。

---

## 9. 不做清单(硬)

- 承 v6 §8 全部:不动 τ 与 0.92 红线;不改 `val`(1000)/`val_soft`(300) 任何字节;不在报告集上做任何拟合
  (conformal §6 的 half1/half2 互斥规则为唯一例外,且必须 assert);不引入外部数据;不把 35 案原文写进训练数据(断言强制);
  不重修 multilingual 头;不跑 R-Drop;不放宽任何判据、不删既有负结论记录;不许缩范围。
- **新增**:不删/不改 neg barber 族与 hair 族(#3 只报告不动数据);新块字面不得为两验收案(猫粮/伤病史)的逐字复刻;
  **half2 与 realtest 不参与任何拟合**;**不得事后新增 §6 候选分数**。

---

## 10. 开放决策(不拍板按默认走)

1. **三块配额浮动**:默认 +400 / +240 / +160;模板空间不足按 v5 先例(补形态/加算子)解决,不得低于配额 80%,缺口逐族写报告。
2. **metric 族选型**:默认 = 体检指标新族(体重/血糖/血脂/心率,不用度数域);备选 = 摊入 rent 等数值族(报告注明所用族)。
3. **alt_praise 族清单**:默认 ≥6 族自选(不含 hair/barber 域),每族 ~40。
4. **s2 渲染集**:默认三档(原渲染 / 中性 A-B / keep-supersede);如需增删,须在评估前固定并写明。
5. **split seed**:沿用 20260928;若与 v6 分半不可比,报告说明。
6. **HC 两族字面风格**:同语气/同长度,不引入新领域词(除 pet/医嘱本体);报告附 2–3 条新字面示例。

---

## 附录 A. 版本与文件对照(本轮)

| 对象 | v5 | v6 | v7 |
|---|---|---|---|
| 训练数据 | `nli_conflict_train_v5.jsonl`(13400) | `nli_conflict_train_v6.jsonl`(14500) | `nli_conflict_train_v7.jsonl`(**15300**) |
| 数据生成 | `gen_soft_conflicts_v5.py` | `gen_soft_conflicts_v6.py` | `gen_soft_conflicts_v7.py` |
| conformal 工具 | `conformal_abstain.py` | + `--split-half` | + `conformal_alt_probe.py`(§6) |
| Kaggle 数据集 | v10 | v11 | **v12** |
| Kaggle kernel | v10 / ce v1 | v11 | **v12** |
| 交付头 | 无(两臂 FAIL) | 无(4 FAIL) | 达标才出 |
| val / val_soft | 逐字节不变 | 逐字节不变 | 逐字节不变 |
| 数据卫生 | 12/35(含 1 对) | 0/35(断言) | **0/35(断言)** |
| HF | 不动 | 不动 | **达标时一次推齐(卡+权重)** |

---

## 第 7 轮执行结果

**结论:11 项硬门槛 PASS / 1 项 FAIL(偏置诊断 12/14,回归守卫破口)⇒ 无交付头,v4 保持已交付,HF 未动(§7 未达标不动)。**
v6 的 4 项 FAIL 全部修复(新 10、val_soft、极性、conformal——后者经 §6 协议采用 s2 规则达标),但 pet 族引入了
新的回归:对照 B2(名字一致)翻转。

### 7.1 执行与资产

| 项 | 值 |
|---|---|
| 训练数据 | `nli_conflict_train_v7.jsonl` 15300 行;§3.5 七条断言 + §3.6 泄漏断言全过;`leak_audit.py` **0/35**(`data_local/leak_audit_v7.txt`) |
| 新块落数 | HC pet 200 / doctor 200(逐族计数见生成器输出;kind 分派 pet:140/40/20、doctor:140/40/20 ≈ 70/20/10);alt_praise 240(city/job/gym_brand/coffee/car/cat_food 各 40,**实渲染 240/240 为 ZH**——复验:双语分支未生效(`len(langs)==2` 恒假、`n_en` 恒 0;语言偏斜照报);metric 160(true/consid/compat/unrelated 各 40,conflict 侧强制递增取值、meta.shape='metric_shift') |
| 新字面抽查(§3.1 双断言) | 8 条样例全为「形状同、字面异」,与猫粮/伤病史两案零逐字段等值(§3.6 断言覆盖);注:任务书 pet 族 act 示例「喂了它鸡胸肉零食」(狗对牛肉过敏)与约束不构成违背,执行侧按语义一致原则配对(act 喂过敏原) |
| Kaggle | 数据集 **v12**;kernel `laya-nli-conflict-ce` **v3(= 计账 kernel v12,纯 CE)**,约 26 分钟;`nli_kernel.py` 同步嵌 v7 脚本未执行 |
| checkpoint | `D:\laya-kaggle-output\laya-nli-conflict-v7\`;model.safetensors SHA256 `73a4c637bd946ce510919db1f308c66053f73a7064966f7723c8a3d6c523cee4` |
| τ(noul) | **1.1058**(v4 1.2176 / v6 1.0905) |
| 复验产物 | 全套 `*_v7.json` + `conformal_v7.json`(maxconf 对半切,存档)、`conformal_v7_s2.json`(采用规则门槛)、`conformal_alt_v7.json`(三候选逐行分数+扫描曲线)、`conf_band_v7.txt`、`leak_audit_v7.txt`、`eval_v7.log` |

### 7.2 硬门槛对照(§5.1)

| 指标 | v4 | v6 | v7 实测 | 门槛 | 判定 |
|---|---|---|---|---|---|
| 主 val acc | 0.901 | 0.903 | **0.903**(err 97) | ≥ 0.896 | **PASS** |
| realtest 老 20 | 20/20 | 20/20 | **20/20** | 20/20 | **PASS** |
| realtest 新 10 | 10/10 | 8/10 | **10/10**(运动伤病史 p=0.8819,宠物粮食亦过) | 10/10 | **PASS**(v6 修复) |
| realtest 否定 5 | 4/5 | 5/5 | **5/5** | 5/5 | **PASS** |
| val_soft 错误数 | 3 | 5 | **1**(剩余 1 错 = degree row18,p 0.16→0.31) | ≤ 3 | **PASS**(v6 修复) |
| 偏置诊断 | 13/14 | 13/14 | **12/14**(对照B2 0.1215→**0.8985** 翻转;D2 v6 已 miss,0.8775→0.8663 持平) | ≥ 13/14 | **FAIL**(本轮唯一) |
| label-surface swap diff | 0 | 0 | **0** | 0 | **PASS** |
| 极性 real/diag \|diff\| | ≤1 | 1,1/0,0 | **0/0**(real)/0/0(diag) | ≤1 | **PASS** |
| 极性 val_soft 错误变化 | 3 | 5 | **1**(1/1/1;变化 = −2) | ≤1 | **PASS**(v6 修复) |
| 分离度 val_soft band | 0.28pp | 16.29pp | **16.64pp**(q10 0.7775 / q90 0.9439) | ≥ 8pp | **PASS** |
| conformal(报数半,采用 §6 规则) | 58/99@29.4% | 4/54@2.8% | **s2:34/51 = 66.7% @ 33.4%**(T=0.9912,ge) | ≥50% @ ≤35% | **PASS**(v6 修复,经 §6 采纳) |
| 训练数据卫生 | 6/35 | 0/35 | **0/35** | 0/35 | **PASS** |

### 7.3 §6 协议全数(conformal_alt_v7.json)

- 划分:seed 20260928 对半,half1 err 46 / half2 err 50(probe CPU p1;门槛脚本用 kernel p1 为 46/51);
  p1 与 kernel val_probs 逐行交叉核对 max dev **0.016**(τ 处理一致)。
- 三候选(half1 选方向+阈值:最大捕捉 @ 弃权 ≤35%;half2 报数):
  - **s1 maxconf(基线)**:half1 T=0.9299(ge,cap 32 @ 34.6%)→ half2 **37/50(74%)@ 36.0% 弃权 ⇒ FAIL**(弃权超线 1pp;预注册门槛不动)。
  - **s2 表面一致性 = 1−(max−min)(三渲染:原渲染/中性 A-B/keep-supersede)**:half1 T=0.9912(ge,cap 29 @ 34%)→ half2 **33/50(66%)@ 33.4% ⇒ PASS**;门槛脚本复核(kernel p1)**34/51(66.7%)@ 33.4% ⇒ PASS**。分离度读数:s2 在错误行均值 0.89 vs 正确行 0.98(s1 为 0.86 vs 0.93)——错误行三渲染概率散开,表面一致性抓得住高置信错误,正是 v6 负结论要的信号。
  - **s3 act_probability**:辅助动作头在该 checkpoint 上**全饱和(1000/1000 行 = 1.0)**——只微调了 noul 头,动作头塌缩,无信息;字段语义已核实(`laya.agent._decode_answers`:`act` 为辅助动作头输出,`act_probability` = 其第 0 槽,挂每答案 `action` 扩展)。
- **判定:采用 s2 规则**(§6)。已写进 `conformal_abstain.py --rule s2`(新模式;默认仍 `maxconf` 以保 v4–v6 数字可复现,v4 回归复核 58/99@29.4% 逐位不变);per-row s2 由 probe 落盘,p1/p2/p3 可确定性重导。
- 主 val 置信分布(v7):q10 0.8850 / q50 0.9337 / q90 0.9435,band 5.85pp;≥0.92 占 80.2%、该档错误率 5.0%(v6 84.4%/6.2%);AURC 0.04648(v6 0.04345,±运行方差内)。

### 7.4 holdout 双读(§5.2;4 类冻结口径 + 剔 barber 3 类口径)

| 轮 | 总错 | 分布内 | holdout(4类) | holdout(3类) | email | desk_floor | degree | barber |
|---|---|---|---|---|---|---|---|---|
| v4 | 3 | 0 | 3 | 2 | 0 | 0 | 2 | 1 |
| v5-A | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| v5-B | 7 | 0 | 7 | 2 | 0 | 0 | 2 | 5 |
| v6 | 5 | 0 | 5 | 3 | 0 | 0 | 3 | 2 |
| **v7** | **1** | **0** | **1** | **1** | 0 | 0 | **1** | **0** |

alt_praise 修掉 barber 侧(v6 2 错 → 0);metric 把 degree 从 3 错压到 1 错(剩余 row18「近视400度→新配500度」p 0.16→0.31,迁移部分起效未过线)。双口径下 v7 均为历轮最少。

### 7.5 偏置诊断 FAIL 的归因与读数(本轮唯一 FAIL)

- 两条 miss:**对照B2**(known「用户的猫叫小白」/ new「用户的猫是白色的」,期望 false,p=0.8985)与**对照D2**(花生过敏/吃开心果,p=0.8663)。
- **D2 非 v7 新增**:v6 该条已 miss(0.8775),v7 持平——v6 的「13/14」本来就悬在 D2 这条上。
- **B2 是 v7 新翻转**(0.1215 → 0.8985),机制最可能与 **pet 族直接相关**:pet 族 200 行中 140 行是「用户家的宠物 known + 宠物相关 new ⇒ true」(第三方约束),而 pet 域的负例只有 40 条 hc_safe(全是「用户回避了过敏原」侧,没有「宠物属性陈述 = 一致」侧);B2 恰是全 diag 集唯一的「宠物 known + 宠物属性 new」对照。归因是机制推断(非消融实验),如实注记。
- v8 若立项修此条:pet/doctor 族补「同主体属性一致 = false」控制行(如「用户家的狗对牛肉过敏」+「用户家的狗是柯基」),预期小改动即可回收 B2,且不动其余已达标轴。
- 其余守卫全保持:老 20、否定、卫生、band、swap、极性、τ/0.92/val/val_soft 零改动。

### 7.6 判定

- 11 PASS / 1 FAIL ⇒ **无交付头,v4 保持已交付;HF 未动**(§7 达标才一次推齐)。
- v6 的全部 4 项 FAIL 在 v7 修复,且 conformal 轴首次以预注册规则形态达标(s2 采用);负结论(偏置诊断回归)照实报,不调门槛不换判据(§5.3)。
- 口径注:v7 延续零句面复用(0/35);barber holdout 与训练域重叠自 v5 起(§2.2 注),双读表已列;跨轮对比按此读。
