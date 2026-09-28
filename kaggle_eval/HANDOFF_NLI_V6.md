# HANDOFF_NLI_V6.md — 第 6 轮任务书(NLI conflict head v6:数据卫生重置 + 纯 CE 主臂;训练轮)

> 读者 = 执行者(无聊天历史)。**开工前必读**:`HANDOFF_NLI_V2.md`(环境/凭据/Kaggle 三坑/插件红线)、
> `HANDOFF_NLI_V3.md`(温度扫描结论)、`HANDOFF_NLI_V4.1.md`(第 4 轮判据与交付)、`HANDOFF_NLI_V5.md`
> (第 5 轮任务书;其文末「第 5 轮执行结果」的三条根因与残留 = 本轮输入)。冲突时以本文为准。
> 全部数字可由仓内落盘数据复算;复算口径见 §2.6。执行完毕后,在本文件末尾追加「第 6 轮执行结果」节(承 v5 惯例)。

---

## 0. 本轮定位

v4 仍是已交付头(`D:\laya-kaggle-output\laya-nli-conflict-v4`,SHA256 见 V3)。v5 两臂各 4 项硬门槛 FAIL
(V5 文末),无交付头。第 5 轮复验(独立复核)逐项复现数字侧后,给出 v6 的输入清单:

| # | 事项 | 实测 | 来源 |
|---|---|---|---|
| 1 | **训练语料含验收句复用(数据卫生)** —— 35 案中 12 案被命中(含 1 对完整原文);v4 已有 6 案(遗留) | 171/13400 行(1.3%) | `kaggle_eval/leak_audit.py`(复验新增) |
| 2 | 老 20「无关补充」双双误报 —— 根因:neg/cf 新块缺 unrelated 控制行 | A/B 均 19/20 | V5 文末·根因① |
| 3 | 否定句 4/5 —— 根因:五形状族缺(财产/通勤/宠物反转等不命中) | A/B 均 4/5 | V5 文末·根因② |
| 4 | conformal 塌缩 —— 拟合面与报数集不同分布 | 12/107@2.3%、16/98@3.0% | `conformal_v5*.json` |
| — | RL 项与分级软目标相克(臂 A 崩、纯 CE 臂 B 反超) | A 0.893 / B 0.902 | V5 文末·根因③ |

**本轮目标(优先级从高到低)**

1. **数据卫生全清(必做;力度已拍板 = 全清,含 v3/v4 遗留)** —— 35 案 known/new 句与语料**零逐字段等值**,
   加断言防回归(§3.1 / §3.6)。
2. **unrelated 控制行(必做)** —— 修 #2(§3.2)。
3. **否定五形状逐族补(必做)** —— 修 #3(§3.3)。
4. **conformal 同分布标定(必做)** —— 修 #4:主 val 对半切(§2.4 / §5.1)。
5. **主臂 = 纯 CE(必做)** —— 处置根因③;RL 代码保留、默认不跑(§4)。

本轮仍是**训练轮**:runtime、τ、0.92 红线、`val`(1000)/`val_soft`(300) 全部零改动(见 §8)。

---

## 1. 范围表

| # | 项 | 本轮 | 依据 / 备注 |
|---|---|---|---|
| 1 | 语料泄漏全清(35 案句级零交集)+ 生成器断言 + `leak_audit.py` 复跑记录 | **必做** | 第 5 轮复验新发现;力度 = 全清(含 v3/v4 遗留 6 案) |
| 2 | neg / cf 两块补 unrelated 控制行 | **必做** | V5 文末:「v6:两块各补 unrelated 控制行(约 1:0.5)」 |
| 3 | 否定五形状逐族补模板 | **必做** | V5 文末:「v6:按 NEGATION_CASES 五形状逐族补(形状可借,原文不入训)」 |
| 4 | 主臂纯 CE(v5 臂 B 配方) | **必做(唯一必跑训练)** | V5 文末:「v6 主臂 = 纯 CE,RL 项留作消融对照」 |
| 5 | RL 对照臂 | 可选(默认不跑) | 预算;A/B 同轮对比 v5 已做过 |
| 6 | conformal 同分布标定(主 val 对半切) | **必做** | V5 文末·残留:「弃权层要可用需同分布标定」 |
| 7 | `dump_probs` 行号回查(consid 中位置信工具缺口) | **必做(小改)** | V5 文末·残留 |
| 8 | R-Drop 消融 | 不做 | 方向决策已拍板(代码留 `--r-drop`,不跑) |
| 9 | τ / 0.92 红线、multilingual(typed-decisions)头 | 不做 | 单独立项(承 V5 §8) |

---

## 2. 预核算(v6 门槛的分母;由 v5 落盘数据复算)

### 2.1 主 val(1000 MNLI;`val_probs_v5.json`=臂 A、`val_probs_v5-ce.json`=臂 B)

| 量 | v4 | 臂 A(v5) | 臂 B(v5) |
|---|---|---|---|
| 错例 / acc | 99 / 0.901 | 107 / 0.893 | 98 / 0.902 |
| conf q01 | 0.7408 | 0.7029 | 0.6342 |
| q10 / q50 | 0.9186 / 0.9230 | 0.9220 / 0.9341 | 0.9100 / 0.9420 |
| q90 / q99 / max | 0.9248 / 0.9263 / 0.9279 | 0.9382 / 0.9419 / 0.9486 | 0.9496 / 0.9556 / 0.9600 |
| **band(q90−q10)** | 0.62pp | 1.62pp | 3.96pp |
| ≥0.92 占比 / 该档错误率 | 87.1% / 6.66% | 90.8% / 7.49% | 87.1% / 6.20% |

置信分箱(箱内「错 / 错误率」):
- v4:[0.5,0.7) 5/.625 · [0.7,0.8) 0 · [0.8,0.9) 13/.448 · [0.9,0.92) 23/.258 · [0.92,0.95) 58/.067
- A:[0.5,0.7) 6/.600 · [0.7,0.8) 8/.500 · [0.8,0.9) 12/.400 · [0.9,0.92) 13/.361 · [0.92,0.95) 68/.075
- B:[0.5,0.7) 9/.643 · [0.7,0.8) 9/.474 · [0.8,0.9) 17/.347 · [0.9,0.92) 9/.191 · [0.92,0.95) 49/.063

### 2.2 val_soft(300)

| 量 | v4 | 臂 A | 臂 B |
|---|---|---|---|
| 错误 | 3 | 1 | 7 |
| conf q10 / q50 | 0.9222 / 0.9240 | 0.7805 / 0.9090 | 0.7877 / 0.9184 |
| q90 / q99 / max | 0.9250 / 0.9261 / 0.9265 | 0.9434 / 0.9502 / 0.9531 | 0.9495 / 0.9594 / 0.9608 |
| **band** | 0.28pp | 16.29pp | 16.18pp |

- 臂 B 主 val 分箱错误率单调(0.643/0.474/0.347/0.191/0.063)= 分级软目标起效的直接证据。
- consid 中位置信:v4 = 0.9239;v5 未报(冻结 val_soft 无逐行 meta)—— v6 随 §4 的行号回查补齐。

### 2.3 极性(`polarity_nli_v5*.json`,seed 20260928,`option_polarity_check.py` 口径)

| 集合 | err std/neu/rnd: v4 → A → B | diff neu/rnd: v4 → A → B | mean\|Δp\| neu/rnd(random): v4 → A → B |
|---|---|---|---|
| real(20) | 0/0/1 → 1/1/1 → 1/1/1 | 0/1 → 0/0 → 0/0 | .0805 → .0129 → .0135 |
| diag(14) | 1/1/1 → 1/1/1 → 1/1/1 | 0/0 → 0/0 → 0/0 | .0640 → .0032 → .0053 |
| val_soft(300) | 3/5/5 → 1/1/2 → 7/7/9 | 2/2 → 0/1 → 0/2 | .0458 → .0061 → .0121 |

gate(`real_diag_diff_le_1`)全过;`val_soft_err_change_le_1`:臂 A PASS(变化 1)、臂 B FAIL(变化 2)。

### 2.4 conformal(`conformal_abstain.py`;v4 默认路径回归 = 58/99 @ 29.4%)

| 臂 | capture | 弃权 | 写入错误率 | AURC / oracle |
|---|---|---|---|---|
| v4 | 58/99 | 29.4% | 0.0581 | 0.05564 / 0.00512 |
| A | 12/107 | 2.3% | 0.0972 | 0.04637 / 0.00599 |
| B | 16/98 | 3.0% | 0.0845 | 0.04402 / 0.00502 |

- 迁移塌缩实锤:同一规则在拟合面(val_soft)自弃权 9.3% / 9.0%(A/B),迁到主 val 只剩 2.3% / 3.0%
  —— 分位阈值踩在两集分位数的犬牙交错处会被放大(判据纪律之「拟合面与报数集」的另一面:不同分布也不行)。
- **v6 设计(必做)**:主 val **对半切** —— 半 1 拟合弃权分位、半 2 报数;两半互斥(脚本里 assert);
  seed 固定、切法与两半 err 数写报告。预期分母:报数半 ~500 行、错例 ≈47–54(v5 两臂 98/107 折半);
  门槛见 §5.1。旧 val_soft 拟合口径作废,但数照报(存档对照)。

### 2.5 泄漏审计(`kaggle_eval/leak_audit.py` 复跑;逐字段等值,无子串)

v5:12/35 案命中;171/13400 行(1.3%)含验收句,其中 21 行涉新 10。v4 对照:6/35 案、90/12000 行(0.8%)、
新 10 零命中。命中句全清单(known×n / new×n)= §3.1 的替换对象:

| 组 | 案 | 命中句(值) |
|---|---|---|
| 老 20 | 地点变更 | known×14「用户住在上海」· new×7「用户最近搬到北京工作了」 |
| 老 20 | 关键事实反转 | known×14「用户对花生过敏」 |
| 老 20 | 部署变更 | known×18「用户的项目部署在阿里云」· new×10「项目已经迁移到腾讯云了」 |
| 老 20 | 口味变更 | known×12「用户喜欢喝美式咖啡」 |
| 老 20 | 无关陈述 | known×13「用户住在北京」· new×2「北京有很多历史古迹」 |
| 老 20 | 饮食禁忌松动 | known×37「用户不吃辣」(v4=17,本轮加深) |
| 老 20 | 子集关系 | new×4「用户在上海上班」 |
| 否定 5 | 否定-戒烟后复吸 | known×20「用户已经戒烟了」(v5 新增;指令举例句) |
| 新 10 | 新-会议时间 | known×6「用户的周会固定在周一上午十点」· new×3「周会已经改到周四下午三点了」(**1 对完整原文**) |
| 新 10 | 新-通勤方式 | known×5「用户通勤坐地铁」 |
| 新 10 | 新-快递地址 | known×4「用户的默认收货地址是公司前台」 |
| 新 10 | 新-运动伤病史 | known×4「用户膝盖韧带拉伤,医生要求静养一个月」 |

**口径注(写进报告;跨轮比较必读)**:v6 起验收 35 案与训练语料零逐句交集;**v2–v5 的数字带既有句式
复用口径**(v4 6 案 / v5 12 案),跨轮对比按此注读 —— 受影响案的难度画像与 v6 可能不同向(通常更严)。

### 2.6 复算口径(与 §2 各表同源)

- 分位数:**最接近序号法** `sorted(xs)[round(p·(n−1))]`;conf = max(p_true,1−p_true);p_true≥0.5 记 true。
- 复算输入:`val_probs_v5[-ce].json`(1000×{gold,p_true})、`val_soft_probs_v5[-ce].json`(300)、
  `polarity_nli_v5[-ce].json`;v4 行取 `val_probs_v4.json` / `val_soft_v4.json` / `polarity_nli_v4.json`。
- 泄漏审计:`python kaggle_eval/leak_audit.py`(默认 v5 vs v4;传路径可审任意训练文件)。
- conformal:`conformal_abstain.py --main … --soft … --real … --diag … --out <out>.json`(判据未动)。

---

## 3. 数据改造(`kaggle_eval/gen_soft_conflicts_v6.py`;「照着 v5 改」)

输出 `data_local/nli_conflict_train_v6.jsonl`。**val 与 val_soft 不重生成**(逐字节不变,直接可比)。

v5 已实现且**保留**:6-labelset 全行轮换、分级软目标(§3.4 表)、精确配额池。本轮只做 §3.1–3.3 三件事 + 断言。

### 3.1 语料泄漏全清(必做;本轮第一优先级)

- 目标:**35 案全部 known/new 句 vs 语料逐字段等值 = 0**(含 v3/v4 遗留 6 案)。替换对象 = §2.5 全清单。
- 替换规则:
  - **只换值,不移族**:城市族换城市、云厂商族换厂商、饮品族换饮品、过敏族换过敏原、否定族换对象……
    族的结构/配比/语气保持不变;每个替换值不得等于任何验收句(§3.6 断言兜底)。
  - 「用户不吃辣」「用户已经戒烟了」「用户对花生过敏」这几句同时是 neg 族/过敏族的设计样例 ——
    清理时否定算子与过敏形状的覆盖不许变窄,只移字面值。
  - 涉及生成链:老 20 的 7 句来自 v3/v4 模板库,否定/新 10 相关来自 v5 新块 —— 修复要贯穿依赖链
    (改对应模块的值池或在其上覆写),不只在 v5 文件一处改。
  - v5 的过程修复(精确配额池/拒重抽/第 4 act 等,commit 134312c)不许回退;替换后重跑全部配额断言。
- 完成判据:§3.6 断言全过 + `leak_audit.py data_local/nli_conflict_train_v6.jsonl` 复跑 0 命中
  (输出存 `data_local/leak_audit_v6.txt`)。

### 3.2 unrelated 控制行(必做;根因①)

- 两块各补(按块规模约 1:0.5):**neg 块 +400、cf 块 +300**,合计 +700。
- 行定义 = **同族话题 + 无逻辑冲突**(补充/提及/并行事实)⇒ 目标 false;其中每族 **≥30% 写成
  「同话题无关补充」角度(对齐老 20「无关补充」案的形状,专治该案误报)。
- 新 kind:`neg_unrelated`、`cf_unrelated`(目标值见 §3.4)。

### 3.3 否定五形状逐族补(必做;根因②)

- 按 `NEGATION_CASES` 五形状逐族补模板(养宠物反转 / 从不喝咖啡 / 戒烟后复吸 / 双重否定-一致 /
  无车与通勤兼容):**每族 +80 行,合计 +400**(同类/对侧按形状语义配比,逐族计数写报告)。
- **形状可借,原文不入训**:形状 = 逻辑结构(否定算子在属性/财产/归属上的反转),字面句必须新写
  (§3.6 断言;v5 教训:五形状里有两个形状(财产归属、通勤)完全没进语料)。

### 3.4 分级软目标(承 v5 §3.3 表 + 两个新档)

| meta.kind | v6 p(gold) |
|---|---|
| mnli / contradiction、mnli / entailment-neutral | 0.95(不动) |
| true / compat | 0.93 ± 0.02 |
| unrelated | 0.96 ± 0.01 |
| consid | 0.80 ± 0.05 |
| hardconstraint / hc_safe | 0.90 ± 0.03 |
| hc_unrelated | 0.96 ± 0.01 |
| neg_true / cf_true | 0.90 ± 0.03 |
| neg_false / cf_false | 0.88 ± 0.05 |
| neg_false_boundary | 0.80 ± 0.05 |
| **neg_unrelated / cf_unrelated(新)** | **0.96 ± 0.01** |

- 硬断言承 v5:`abs(p(gold) − 0.5) ≥ 0.2`;`gold.label == argmax`;取值只在表列档位内。
- 已知代价承 v5:val/val_soft 的 gold 仍是 0.95,与模型输出不同尺度 ⇒ 这两集合只承担准确率/错误计数/
  极性判据(ECE 判据重写口径见 §5.2)。

### 3.5 规模与硬不变量

| 块 | v4 | v5 | v6 |
|---|---|---|---|
| MNLI(整块逐字节不变) | 6000 | 6000 | 6000 |
| 合成软冲突(true 1800 / compat 1440 / consid 1080 / unrelated 1080) | 5400 | 5400 | 5400 |
| 硬约束(300 / 180 / 120) | 600 | 600 | 600 |
| 否定算子族 | — | 800 | **1600**(+五形状 400、+unrelated 400) |
| known 一步扰动(cf) | — | 600 | **900**(+unrelated 300) |
| 合计 | 12000 | 13400 | **14500** |

自检(全部 assert,写在生成脚本尾部;承 v5 §3.4 六条并按上表改数,另加 §3.6):
1. 行数 = 14500;`state` 全局唯一;与 val / val_soft 零交集。
2. gold 计数按 kind 与上表一致;`true : false` 比例记录(不设硬门槛,但要报)。
3. labelset 分布满足 v5 配比(语义 4 套合计 ≥50%、LS5 ≥20%、LS6 ≥20%);每 labelset ≥3 kinds。
4. 每行 `meta` 完整(`cat/kind/lang/labelset`)。
5. `gold.label == argmax`;`|p(gold) − 0.5| ≥ 0.2`;档位封闭。
6. 新块逐族计数落报告(五形状 ×80、neg_unrelated 400、cf_unrelated 300、各块正负比)。

### 3.6 泄漏断言(新增;防回归)

```python
# ---- §3.6 泄漏断言(v6 新增;追加在 §3.5 断言之后)----
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

assert_no_case_leak(v6_train)   # 变量名以 v6 生成器为准(v5 里为 v5_train)
```

- 运行位置:生成器尾部(生成器在 `kaggle_eval/` 下运行,`realtest_v2/v4` 为同目录纯数据模块,直接 import)。
- 该断言 = 硬不变量:不过不许写输出文件。`leak_audit.py` 是脚本版复跑对照,两者口径一致(逐字段等值)。

---

## 4. 训练改造(`kaggle_eval/train_nli_conflict.py` → v6)

- **主臂(必做、唯一必跑)**:纯 CE = v5 臂 B 配方 —— `loss = loss_ce / GRAD_ACCUM`(upstream laya#238 的一行),
  数据 = v6;超参承 v4/v5(EPOCHS=4、MICRO_BATCH=16、GRAD_ACCUM=2、LR_ENCODER 2.5e-5、LR_HEAD 1e-4、
  SIGMA 0.4→0.1),底座不变(`convaiinnovations/laya-multilingual`)。RL 项代码保留(开关自选,报告里写清
  每次运行对应 commit/参数)。
- **RL 对照臂**:可选,默认不跑(§9.3)。
- CLI:`--expected-train 14500`(防静默截断);`max_items` 上限抬到 15000。
- 尾部承 v5:400 条 calib 切分(seed 20260927 不变)→ τ 拟合 → 落盘 `model.safetensors`/`encoder`/`tokenizer`/
  `rl_agent_config.json`/`metrics.json`;val 1000 逐行 `{gold, p_true}` 落盘 `val_probs_v6.json`。
- **`dump_probs.py` 行号回查(必做,小改)**:val_soft 输出的 `meta` 带原始文件行号(冻结文件可回查
  consid 明细,补 v5 工具缺口);输出形状保持与 `val_soft_v4.json` 一致 + 行号字段。

**止损线(先写死,执行时照判)**:
- 主 val acc < **0.885** ⇒ 判 FAIL,不再叠改动,原样报数。
- 极性 val_soft err 变化 >1 ⇒ 该轴 FAIL,如实报。
- 否定仍 <5/5 ⇒ 判 FAIL 并列逐条(5 条判定与 p 值)写进报告。
- **§3.6 泄漏断言失败 ⇒ 立即停修数据,禁止带泄漏训练**(新增)。

---

## 5. 门槛(v6 口径;承 v5,只准加)

### 5.1 硬门槛(全过才算交付)

| 集合 / 指标 | v4 | 臂 A | 臂 B | v6 门槛 |
|---|---|---|---|---|
| 主 val acc | 0.901 | 0.893 | 0.902 | ≥ 0.896(不放宽) |
| realtest 老 20 | 20/20 | 19/20 | 19/20 | 20/20 |
| realtest 新 10 | 10/10 | 8/10 | 10/10 | 10/10 |
| realtest 否定 5 | 4/5 | 4/5 | 4/5 | 5/5 |
| val_soft 错误数 | 3 | 1 | 7 | ≤ 3 |
| 偏置诊断 | 13/14 | 13/14 | 13/14 | ≥ 13/14 |
| label-surface swap diff | 0/0 | 0/0 | 0/0 | 0 |
| 极性 real/diag \|diff\| | ≤1 | ≤1 | ≤1 | ≤1 |
| 极性 val_soft 错误变化 | +2 | ≤1 | +2 | ≤1 |
| 分离度 val_soft band | 0.28pp | 16.29pp | 16.18pp | ≥ 8pp |
| conformal(同分布标定,报数半) | 58/99@29.4% | 12/107@2.3% | 16/98@3.0% | 捕捉率 ≥50% @ 弃权 ≤35%(v4 参照 58.6%) |
| **训练数据卫生(新)** | 6/35 案复用 | 12/35(含 1 对完整原文) | 同左 | **0/35**(断言全过 + `leak_audit.py` 0 命中) |

### 5.2 报告项(不设硬门槛,但必须给数)

- 分离度全套:主 val 与 val_soft 的 conf q10/q50/q90/q99/max、band、置信分箱,与 §2 两表逐格对照;
  consid 中位置信(行号回查后单列)。
- conformal 三个数:拟合半(样本内)、报数半(报数)、旧 val_soft 拟合口径(作废存档);AURC 对照。
- 极性 mean|Δp|:三集 × neutral/random,与 §2.3 对照。
- τ 拟合值(与 v4 noul=1.2176、v5 A=1.138 / B=1.065 对照)。
- 训练曲线:每 epoch 平均 loss(跑 RL 臂则同表)。
- `leak_audit.py` 复跑输出 + 五形状逐族计数 + 各块正负/控制比。
- ECE:承 v5 重写口径(只在 `acc<1` 的切片上评;≥0.92 占比独立跟踪)。

### 5.3 交付判定

- 任一硬门槛 FAIL ⇒ 交付物照出、结论写 FAIL(负结论也算完成);不许调门槛/换判据集/挑样本抹平。
- 预核算不可达的门槛不允许硬凑(承 v5 §5.3)。
- **口径注(必写进报告)**:v6 起验收 35 案与训练语料零逐句交集;v2–v5 数字含既有句式复用口径,
  跨轮对比按 §2.5 注读。

---

## 6. 交付物与归档

**脚本(进仓,`kaggle_eval/`)**
- `gen_soft_conflicts_v6.py`(含 §3.5 全断言 + §3.6 泄漏断言)
- `train_nli_conflict.py`(v6:`--expected-train 14500`;纯 CE 主臂)
- `dump_probs.py`(行号回查版)
- `nli_kernel.py` / `nli_kernel_ce.py`(kernel 版,版本表加 v11 行;嵌入 train v6)
- `leak_audit.py`(已在仓;复跑输出存 `data_local/leak_audit_v6.txt`)
- 复验沿用(**不改判据**):`option_polarity_check.py`、`realtest_v2.py`、`realtest_v4.py`、
  `noul_bias_diag.py`、`label_swap_check.py`、`val_breakdown.py`、`conformal_abstain.py`

**数据产物(`data_local/`)**
- `nli_conflict_train_v6.jsonl`(14500 行)
- `val_probs_v6.json`、`val_soft_probs_v6.json`、`polarity_nli_v6.json`、`memory_conflict_realtest_v6.json`、
  `noul_bias_diag_v6.json`、`label_swap_v6.json`、`conformal_v6.json`、`conf_band_v6.txt`、`leak_audit_v6.txt`

**Kaggle 资产**
- 数据集 `daphnelaurent/nli-conflict-pairs` **v11**(train 换 v6;val(1000)/val_soft(300) 冻结不变;
  `_README` 加 v11 行)
- kernel `daphnelaurent/laya-nli-memory-conflict-fine-tune` **v11**(主臂)

**模型产物**:`D:\laya-kaggle-output\laya-nli-conflict-v6\`(`model.safetensors` + `metrics.json` +
`rl_agent_config.json`);总大小与 SHA256 写进报告。

**HF**:达标才更新卡(同基准对比节 + 数据卫生句:「验收 35 案与训练语料零逐句交集」);未达标不动。

---

## 7. 执行环境与命令(承 V5 §7;新增两条)

- Kaggle CLI 的 **tmp 必须指到 ASCII 路径**(如 `C:/kaggle_cfg/tmp`);token 走 `KAGGLE_API_TOKEN`
  环境变量,不落盘、不进日志(细则见 `HANDOFF_NLI_V2.md`)。
- **数据集总挂最新版**;README 改动也要出新版本号,否则 kernel 侧拿到旧版。
- **每次 push kernel 都会从头重训**(CUDA 非确定性 ⇒ 权重不可位复现):本机留档副本才是参照物。
- kernel 的 `QUICK_SAVE` 纯文档版无输出(status ERROR 属占位)。
- 本机 CPU 复现:`D:/Miniconda/envs/laya-ft/python.exe`;`pandas` 被系统策略挡,一律 `pyarrow`。
- **本机→kaggleusercontent 大文件链路间歇断流**:重试循环 + 串行下载(v5 实测教训)。
- **泄漏审计**:`python kaggle_eval/leak_audit.py [train.jsonl …]`(默认 v5/v4;每次生成后跑,期望 0 命中)。

---

## 8. 不做清单(硬)

- 不动 τ 与 0.92 红线;不改 `val`(1000)/`val_soft`(300) 任何字节
- 不在报告集上做任何拟合 —— conformal 的拟合半是为「与报数半互斥」而设的例外,且必须 assert 两半不相交
- 不引入外部数据(tasksource-jev、Sev 等只作**方法**参考,不下载不用)
- 不把 35 案原文写进训练数据 —— **全清之后本条升级为断言强制**(§3.6)
- 不重修 multilingual 头(另立项);不跑 R-Drop
- 不放宽任何判据、不删既有负结论记录
- **不许缩范围**:「unrelated 控制行只加一小撮」「泄漏只清新 10」都算未完成(§9.2 的配额浮动除外)

---

## 9. 开放决策(不拍板则按默认走)

1. **conformal 对半切的切法**:默认 = 固定 seed 洗牌后对半(两半 err 数、acc 差报出;差 >1.5pp 需在报告说明);
   备选 = 另生成同分布标定切片(成本高,默认不走)。
2. **配额浮动**:五形状(+80/族)与 unrelated(+400/+300)为默认配额;模板空间不足时按 v5 先例(补形态/
   加算子)解决,不得低于配额 80%,缺口逐族写报告。
3. **RL 对照臂**:默认不跑;主臂出结果且预算允许时可加跑(加跑则同表对比)。
4. **替换值的风格**:默认 = 与对应族现有行同语气/同长度、不引入新领域词/新实体类型;报告附 2–3 条
   替换前后示例。

---

## 附录 A. 版本与文件对照(本轮)

| 对象 | v4 | v5 | v6 |
|---|---|---|---|
| 训练数据 | `nli_conflict_train_v4.jsonl`(12000) | `nli_conflict_train_v5.jsonl`(13400) | `nli_conflict_train_v6.jsonl`(**14500**) |
| 数据生成 | `gen_soft_conflicts_v4.py` | `gen_soft_conflicts_v5.py` | `gen_soft_conflicts_v6.py` |
| 训练脚本 | v4 版 | v5 版(三开关) | v6 版(纯 CE 主臂) |
| 泄漏审计 | — | — | `leak_audit.py`(断言强制 0 命中) |
| Kaggle 数据集 | v8 | v10 | **v11** |
| Kaggle kernel | v8 / v9 | v10 / ce v1 | **v11** |
| 交付头 | `laya-nli-conflict-v4`(仍交付) | 无(两臂 FAIL) | 达标才出 |
| val / val_soft | 1000 / 300 | 逐字节不变 | 逐字节不变 |
| 数据卫生 | 6/35 案复用 | 12/35(含 1 对) | **0/35(断言)** |
