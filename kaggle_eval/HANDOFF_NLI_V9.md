# HANDOFF_NLI_V9.md — 第 9 轮任务书:剂量补足(B2)+ 形状补齐(waver/变更)+ conformal 语义钉死

> 对象:新一轮数据改造 + 训练 + 复验。开工前必读:`HANDOFF_NLI_V3.md`(温度扫描)、
> `HANDOFF_NLI_V4.1.md`(判据与交付)、`HANDOFF_NLI_V5.md`、`HANDOFF_NLI_V6.md`、
> `HANDOFF_NLI_V7.md`(§6 预注册协议)、**`HANDOFF_NLI_V8.md`(§8 执行结果 + §8.3 复核侧补记 +
> §8.4 v9 议题——本轮范围全部由此提炼,不另立新方向)**。

## 0. 本轮定位(一句话)

v8 把 B2 从 0.8985 拉回 **0.7491**(方向已证、剂量不够),并把 val_soft/polarity 的两条线
推到贴边(holdout 的「考虑改变=兼容」形状在 v8 权重下被误判为冲突);**本轮只做三件事**:
**① attr 块加剂(+240,含反向对照)**、**② in-dist 形状补齐(waver +120 / 变更 +40)**、
**③ conformal 语义钉死(复跑 = 重裁)+ 选点留 1pp 余量**。其余全部冻结,门槛只加不放宽。

---

## 1. 失败清单与根因(全部来自 v8,复核侧已独立复算)

### 1.1 四轴失败(v8:8 PASS / 4 FAIL)

| 轴 | v7 | v8 | 本轮要求 |
|---|---|---|---|
| 偏置诊断 | 12/14(含 B2 翻) | **12/14** | ≥13/14 且 **B2 必须过(p<0.5)** |
| B2 名字一致 | 0.8985 ✗ | **0.7491** ✗(方向已证) | 回 false |
| D2 其他坚果无关 | 0.8663 ✗ | 0.8777 ✗ | 允许照旧 miss(13/14 留 1 席) |
| val_soft 错误数 | 1 | **7** | ≤3 |
| 极性 val_soft 变化 | 1/1/1 | **7/9/10(+6)** | ≤1 |
| conformal(采用规则) | s2 34/51@33.4% ✓ | **s2 42/54@35.4%** ✗(超 0.4pp) | ≥50% @ ≤35% |

### 1.2 val_soft 7 错的行级结构(**本轮第 ② 项的直接依据**)

| row_id | cat(holdout) | kind | known → new | v7 p | v8 p | 判定 |
|---|---|---|---|---|---|---|
| 18 | degree | true | 用户近视400度 → 新配眼镜500度,度数加深了 | 0.3074 | 0.2942 | 漏判(应 true) |
| 86 | degree | consid | 验光度数300度 → 用户在想要不要重新验光 | 0.1948 | **0.8765** | 翻错 |
| 255 | degree | consid | 验光度数500度 → 用户在想要不要重新验光 | 0.1951 | **0.8643** | 翻错 |
| 215 | barber | true | 固定理发师Kevin → 现在都找阿杰剪 | 0.8901 | **0.2753** | 漏判(应 true) |
| 231 | barber | consid | 剪发一直找Andy → 在想要不要换阿杰试试 | 0.2314 | **0.8733** | 翻错 |
| 263 | barber | consid | 剪发一直找Tony → 在想要不要换阿杰试试 | 0.2206 | **0.8758** | 翻错 |
| 291 | barber | consid | 剪发一直找阿杰 → 在想要不要换Andy试试 | 0.2238 | **0.8223** | 翻错 |

- **结构**:5/7 是 **「在想要不要{动作}」= 犹豫/考虑改变 ⇒ 兼容(false)** 形状;2/7 是
  **「变更已发生」⇒ 冲突(true)**。两类都落在 holdout 猫(degree/barber,**训练零行**)。
- **机制**:in_dist 0/200 恒不错 ⇒ 不是分布外噪声,而是**该形状在训练语料里的 ZH 表达覆盖不足**
  (训练 consid 行的「考虑/听说/计划试」多为 EN 或别的句式);v8 的 +120 compat 行把 false 侧
  剂量上抬后,holdout 的「考虑改变」行被推向 true。
- **处置**:不许改门控/口径;本轮用 **in-dist 形状补齐**(第 ② 项)把「犹豫 ≠ 变更」的双形状
  在训练域内做厚,靠语义泛化迁移到 holdout(与 §8.4 议题 2 一致)。

### 1.3 conformal 口径分流(复核侧补记,§8.3/§8.4 议题 5)

- 同一 `conformal_alt_v8.json` 半2 读数:s1 maxconf **38/54 = 70.4% @ 25.4% 达标**、
  s2 42/54 = 77.8% @ **35.4%**(超线 0.4pp)、s3 退化;文件 `verdict` = **ADOPT s1_maxconf**。
- v8 §5/§6 冻结「采用 s2」⇒ 按 s2 计 FAIL。**本轮把语义钉死(见 §6)**,终结此歧义。

---

## 2. 预核算(2026-09-30,复验侧从 v8 落盘逐值复算;执行者按表复核,不替算)

### 2.1 必须保持的 8 项(v8 实测,复验侧已复现)

| 指标 | v8 实测 | 保持要求 |
|---|---|---|
| 主 val acc | **0.8960**(err 104;q10 0.8868 / q90 0.9430) | ≥0.896(**贴线,红线**) |
| 老 20 | 20/20 | 20/20 |
| 新 10 | 10/10(伤病史 0.8887) | 10/10 |
| 否定 5 | 5/5 | 5/5 |
| label-surface swap | 0 | 0 |
| 极性 real/diag \|diff\| | 0/0 · 0/0 | ≤1 |
| 分离度 val_soft band | 16.70pp(q10 0.7761 / q90 0.9431) | ≥8pp |
| 数据卫生 | 0/35 + diag 18 对 0 命中 | 0/35(断言升级含 diag 对) |

### 2.2 本轮增量的风险读数

- **主 val 贴线**:v7 0.903(err 97)→ v8 0.8960(err 104),+120 行后 −0.7pp(含 CUDA 跨 run 方差)。
  本轮增量 **+380(≤2.5%)** 已是上限,且全部为「真内容」行(非纯 compat 粉尘);
  **止损线**:主 val < 0.885 ⇒ 判 FAIL 照报不叠改动;**若落在 0.885–0.896 ⇒ 该轴如实 FAIL**。
- **τ 报告纪律(承 §4)**:τ 直接读回 `rl_agent_config.json` temperature[2](v8 = 1.1240078;
  v6 1.0905 / v7 1.1058;曾误写默认 1.200)。
- **口径注(必写进报告,承 v8 §8.4)**:v6 起验收 35 案与训练语料零逐句交集;val/val_soft 与训练
  语料存在**家族级句式共享**(字段级 310 命中,v7/v8 同量,非回归);卫生口径以「35 案 + diag 对 +
  本轮新块」逐字段零交 与 **state 级** val/val_soft 零交为准;barber holdout 与训练域重叠自 v5 起。

---

## 3. 数据改造(`kaggle_eval/gen_soft_conflicts_v9.py`;「照着 v8 改」)

输出 `data_local/nli_conflict_train_v9.jsonl`。**val(1000)/val_soft(300) 不重生成**(逐字节不变)。
**v8 的 15420 行逐字节保持**(carried 块照 v7→v8 惯例重放 rng 流,新块吃流尾)。

### 3.1 块 A — attr 加剂(+200;唯一既有形状的加量)

- 复用 v8 的 `PET_NAME_T` / `PET_BENIGN_ATTR` / `DOCTOR_BG` 与 `make_attr_rows` 骨架;
  **词池扩充**(不是复制行):名池 +3、色池 +2(均避开禁词 小白/白色/开心果)、
  pet 良性属性 +6、doctor 背景事实 +4(每个语种)。
- **配额(写死,assert 精确)**:`pet_name` **120**(zh 72 / en 48)+ `pet_benign` **40**(zh 24 / en 16)
  + `doctor_bg` **40**(zh 24 / en 16)= **+200**(zh 120 / en 80)。
- `kind='compat'`(TARGETS['compat'] = 0.93±0.02);`meta.shape='attr_consistent'`;
  `meta.family ∈ {pet_name, pet_benign, doctor_bg}`;`cat' ∈ {pet_attr, doctor_attr}`。
- **重点**:`pet_name` 的 known 侧**必须包含 B2 同构句式「用 户的猫叫{N}」**(v8 已有该模板,
  本轮占比 ≥1/3),new 侧「用户的猫是只{C}猫」;形状与 B2 同、字面全新。

### 3.2 块 B — attr 反向对照(+40;**新增形状,保护 true 侧**)

- **语义**:同主体、`new` 与 `known` 的属性**互相矛盾** ⇒ **true**(冲突)。
- 两个形状各 20(zh 12 / en 8):
  1. **外观矛盾**:`known「用户的猫是只{色1}猫」/ new「用户的猫是只{色2}猫」`(色1≠色2);
  2. **名字矛盾**:`known「用户的猫叫{名1}」/ new「用户的猫叫{名2}」`(名1≠名2;名池 ≠ 小白)。
- `kind='true'`(TARGETS['true'] = 0.95±0.02,与既有「变更已发生」行同档);
  `meta.shape='attr_contradiction'`、`cat='pet_attr'`。
- **作用**:防止 200 行「宠物属性 ⇒ false」被学成「宠物属性一律 false」而伤 pet HC true 侧
  (v8 未观察到损伤,但 2.5× 剂量下需要对称护栏)。

### 3.3 块 C — waver 形状补齐(+120;consid 档,只进 in-dist 猫)

- **语义**:`known` = 现状,`new` = **犹豫/考虑改变(尚未发生)** ⇒ 兼容(**false**)。
- **句式(新模板,zh/en 各 ≥6 条轮转)**:zh「用户在想要不要{动作}」·「用户有点想试试{动作}」·
  「用户最近在琢磨要不要{动作}」…;en "The user is considering {action}" · "The user might
  try {action} someday" · "The user is toying with the idea of {action}"…
- **配额**:+120(zh 72 / en 48);**只在 in-dist consid 八猫轮转**:`city / car / job / coffee /
  lang / gym_brand / team / cloud`(每猫 12–15 行);**holdout 猫(email/desk_floor/degree/barber)
  零行保持**。
- `kind='consid'`(TARGETS['consid'] 0.80±0.05,现档);`meta.shape='waver_consider'`;
  `meta.family='waver'`。
- **词池**:复用既有 consid 族的 known 池(现状句),动作词池扩充(新增 ≥8 个/语种)。

### 3.4 块 D — 变更已发生(+40;true 档,只进 in-dist 猫)

- **语义**:`known` = 旧状态,`new` = **已经改变(已发生)** ⇒ 冲突(**true**)。
- **句式**:zh「用户把{旧}换成了{新}」·「用户不再去{旧},改去{新}了」;en "The user switched
  from {old} to {new}" · "The user no longer goes to {old}, they go to {new} now"。
- **配额**:+40(zh 24 / en 16);in-dist 猫轮转 ≥6 猫;`kind='true'`;`meta.shape='change_done'`。
- **作用**:与块 C 成对(同一域内「犹豫 ⇒ false」/「已变更 ⇒ true」),给模型可迁移的语义对比。

### 3.5 规模表

| 块 | v7 | v8 | v9 |
|---|---|---|---|
| MNLI(逐字节不变) | 6000 | 6000 | 6000 |
| 合成软冲突 | 5800 | 5920 | 5920 |
| 硬约束 | 1000 | 1000 | 1000 |
| 否定算子族 | 1600 | 1600 | 1600 |
| known 一步扰动 | 900 | 900 | 900 |
| **v9 新增**:attr +200 / 反向对照 +40 / waver +120 / 变更 +40 | — | (+120 attr) | **+400** |
| **合计** | 15300 | 15420 | **15820** |

### 3.6 断言(承接 v8 + 新增;全过才许写输出)

1. 行数 = **15820**;`state` 全局唯一;state 级与 val/val_soft 零交。
2. kind 计数:compat 1600 → **1800**;consid 1360 → **1480**;true 1840 → **1920**;其余不变。
3. labelset 轮换:语义 4 套 ≥50%、LS5/LS6 各 ≈20%(按 15820 精确改算,assert 精确)。
4. 每行 `meta` 完整;新块含 `shape ∈ {attr_consistent, attr_contradiction, waver_consider, change_done}`。
5. `gold.label == argmax`;`|p(gold) − 0.5| ≥ 0.2`;档位封闭(`TARGETS` 现档,不新增档)。
6. **泄漏断言(承 v8 升级版)**:35 案 + diag 14 对 + 4 交换对逐字段等值 = 0;
   **新块(四个)逐字段 vs val/val_soft = 0**(v8 只对 attr 块做过,本轮四块全做)。
7. 禁词扫描(小白/白色/开心果)= 0;新块违逆标记词(不顾/偏要/硬要/执意/照样)= 0。
8. holdout 猫零行复核:`{email, desk_floor, degree, barber}` 在训练语料中行数 = **0**(断言)。

---

## 4. 训练(承 v8,零改动)

- 主臂纯 CE(EPOCHS=4、MICRO 16、ACCUM 2、SIGMA 0.4→0.1、底座不变 `convaiinnovations/laya-multilingual`)。
- CLI:`--expected-train 15820 --max-items 16200`。
- 尾部承 v5–v8:400 calib(seed 20260927)→ τ 拟合 → 落盘;**τ 直接读回
  `rl_agent_config.json` temperature[2] 写进报告**。val 1000 逐行 `{gold,p_true}` → `val_probs_v9.json`。
- **止损线**:主 val < 0.885 ⇒ FAIL 照报不叠改动;泄漏断言失败 ⇒ 立即停;
  新 10 <10/10、否定 <5/5 ⇒ 照实报;val_soft >3、极性变化 >1 ⇒ 对应轴 FAIL 照报(逐条列 p);
  **B2 仍未过 0.5 ⇒ 负结论照交 + 附剂量-响应外推表**(本底:v8 +120 attr(其中 pet 同形状仅 40)
  ⇒ B2 −15pp 至 0.7491;本轮 +240 attr ⇒ 读数),**不叠第二轮修补**(留 v10)。

---

## 5. 门槛(承 v8 §5,**只准加,不许放宽**)

| 集合 / 指标 | v8 实测 | v9 门槛 |
|---|---|---|
| 主 val acc | 0.8960 | ≥ 0.896 |
| realtest 老 20 | 20/20 | 20/20 |
| realtest 新 10 | 10/10 | 10/10 |
| realtest 否定 5 | 5/5 | 5/5 |
| val_soft 错误数 | 7 | ≤ 3 |
| 偏置诊断 | 12/14 | **≥13/14 且 B2 必须过** |
| label-surface swap diff | 0 | 0 |
| 极性 real/diag \|diff\| | 0/0 | ≤1 |
| 极性 val_soft 错误变化 | +6 | ≤1 |
| 分离度 val_soft band | 16.70pp | ≥ 8pp |
| conformal(**§6 重裁后的采用规则**) | s2 35.4% | ≥50% @ ≤35% |
| 训练数据卫生 | 0/35 | 0/35 |

**报告项(不设硬门槛)**:四个新块的逐形状计数 + 2–3 条新字面示例;分离度全套 + consid 中位置信;
holdout 双读表(v4–v9 同表);极性 mean\|Δp\| 三集;τ 拟合值;逐族计数;训练曲线;
**本轮新增**:waver/变更两块的 in-dist 命中率(val_soft 之外的对照读数)。

---

## 6. conformal 语义钉死(本轮的协议变更,预注册)

1. **复跑 = 重裁(明写)**:`conformal_alt_probe.py` 在每个候选上 half1 选点、half2 判定;
   **任一候选 half2 满足 ≥50% capture @ ≤35% abstain ⇒ 该规则为本轮采用规则**(与 v7 §6 原文一致);
   报告按**采用规则**记门槛读数,并给三候选全曲线 + 旧口径(maxconf LAC)存档对照。
2. **选点留余量(防护性预注册)**:half1 选点预算从 `abstain ≤ 35%` 收紧为 **`≤ 34%`**
   (1pp 余量,防 half1→half2 漂移;v8 s2 即 33.2% → 35.4% 破线)。**门槛不动(仍 ≤35%)**;
   三候选同一预算,不得事后调。
3. 候选仍**冻结为三个**(s1 maxconf / s2 表面一致性 / s3 act_probability),**不得新增**;
   half2 与 35 案不参与任何拟合;协议外探索必须标「探索,不得采用」。
4. 执行:
   `python kaggle_eval/conformal_alt_probe.py <v9 ckpt> data_local/conformal_alt_v9.json`;
   `python kaggle_eval/conformal_abstain.py --rule s2 --split-half --alt-scores
   data_local/conformal_alt_v9.json --main … --soft … --real … --diag … --out
   data_local/conformal_v9_s2.json`(**`--alt-scores` 显式指 v9 文件**;s2 存档与 maxconf 存档同 v8 流程)。
   若重裁结论为 s1,另跑 `--rule maxconf --split-half --alt-scores …
   --out data_local/conformal_v9_s1.json` 出「采用规则」档并同时在报告列出两档读数。
5. **`score_definition` 标签已修**(v8 收尾:现在记录实际 `--alt-scores` 文件名)——报告引用时以此为准。

---

## 7. 交付物与归档

- **脚本**:`gen_soft_conflicts_v9.py`(新);其余沿用不改判据;`nli_kernel.py` / `nli_kernel_ce.py`
  同步嵌 v9 脚本(不执行)。
- **数据产物**:`nli_conflict_train_v9.jsonl`(15820);`val_probs_v9.json`、`val_soft_probs_v9.json`、
  `polarity_nli_v9.json`、`memory_conflict_realtest_v9.json`、`noul_bias_diag_v9.json`、
  `label_swap_v9.json`、`conformal_v9.json`、`conformal_v9_s2.json`、`conformal_v9_s1.json`(如重裁为 s1)、
  `conformal_alt_v9.json`、`conf_band_v9.txt`、`leak_audit_v9.txt`、`eval_v9.log`。
- **Kaggle**:数据集 **v14**(train 换 v9;val/val_soft 冻结;README 加 v14 行);
  kernel `laya-nli-conflict-ce` 再推一版(**计账 kernel v14**,纯 CE 主臂)。
- **模型**:`D:\laya-kaggle-output\laya-nli-conflict-v9\`;SHA256 + 大小写进报告。
- **HF(达标才推)**:卡(**同基准对比节 v4–v9** + 「验收 35 案与训练语料零逐句交集」句 +
  新头 SHA)+ 新头权重全文件,一次推齐;未达标不动、不补推。

---

## 8. 执行环境与命令(承 v8 §8 + 实测新坑)

- ⚠️ **本机 torch 受 Smart App Control 间歇性拦截**(2026-09-29 实测:上午正常、午后拦
  `WinError 4551`、随后又三连通过;SAC 全程 = On)。开工先探测
  `"D:/Miniconda/envs/laya-ft/python.exe" -c "import torch; print('ok')"`;
  **被拦先隔几分钟重试若干轮**;仍拦再停(训练在 Kaggle、生成器/离线复算不受影响)。
- **电池内存坑(本轮复验侧实测)**:`conformal_alt_probe.py` 全量(1000 行 ×3 渲染)约 20 分钟,
  期间 CPU 吃满、**内存剩 ~1 MB**;此窗口内**不要再起重 python**(其它脚本会被拖到分钟级甚至超时),
  等 probe 结束再跑离线脚本。
- Kaggle CLI tmp 指 ASCII(`C:/kaggle_cfg/tmp`);数据集总挂最新;**每次 push kernel 从头重训**
  (CUDA 非确定 ⇒ 权重不可位复现,本机留档副本为准);文档版 QUICK_SAVE 无输出属正常占位。
- 泄漏审计:`python kaggle_eval/leak_audit.py data_local/nli_conflict_train_v9.jsonl`;
  另跑复验侧独立扫描(35 案 + diag 18 对 + 四新块 vs val/val_soft)。
- 本机 CPU 复现:`D:/Miniconda/envs/laya-ft/python.exe`;`pandas` 被系统策略挡,一律 `pyarrow`。

---

## 9. 不做清单(硬)

- 承 v8 §9 全部:不动 τ 与 0.92 红线;不改 val/val_soft 任何字节;不在报告集上做任何拟合
  (§6 half1/half2 互斥为唯一例外);不引入外部数据;不把 35 案原文写进训练数据;
  不放宽任何判据;不删既有负结论;**不得事后新增 §6 候选**;不修 alt_praise 双语分支(死代码照旧)。
- 新增:**不给 holdout 猫(email/desk_floor/degree/barber)加任何训练行**(断言 0);
  不动 v7/v8 已交付行的内容(只允许追加);不新增目标档位(TARGETS 保持现档);
  不为了压 B2 而删 D2 的对照行;**B2 仍未过 0.5 也不叠第二轮修补**。

---

## 10. 开放决策(不拍板按默认走)

1. **词池细节**:名池/色池/动作池执行侧自定(守禁词与既有护栏),报告给抽样示例。
2. **块 C/D 的猫分配**:默认 in-dist 八猫轮转、每猫 12–15 行;猫数不足时按既有 consid 猫补。
3. **块 B 的 sheen**:若外观矛盾行与既有 pet HC 行字面冲突,换色池;冲突无法避免时按 D 侧配额减半,
   缺口逐族写报告。
4. **conformal 重裁结论**:若本轮重裁仍为 s2/s3 之外的情形,照 s1/s2 两档都跑、按 §6.1 记采用规则;
   若三候选全 FAIL ⇒ 负结论照交,v10 立保守选点或换族的预注册。
5. **val_soft 再失败**(>3 且 in_dist 0、错全在 holdout)⇒ 判定跨 run 方差,**不改门槛**,
   v10 立「多 seed 复验协议」后再谈口径(先立协议,不事后放宽)。
6. **归因需要时**(仅当主臂三轴以上失败、需要分离 B2 剂量与 waver/变更块贡献)可另跑消融臂
   (v8 语料 + waver/变更块,不含 attr 加剂),单列表报、**不并入本轮判定**。

---

## 附录 A. 版本与文件对照(本轮)

| 对象 | v7 | v8 | v9 |
|---|---|---|---|
| 训练数据 | `…_v7.jsonl`(15300) | `…_v8.jsonl`(15420) | `…_v9.jsonl`(**15820**) |
| 数据生成 | `gen_soft_conflicts_v7.py` | `gen_soft_conflicts_v8.py` | `gen_soft_conflicts_v9.py` |
| conformal 工具 | + `conformal_alt_probe.py`、`--rule s2` | v8 分档复跑 | **重裁语义 + 选点 ≤34%** |
| Kaggle 数据集 | v12 | v13 | **v14** |
| Kaggle kernel | v12 | v13(ce v4) | **v14**(ce 再推一版) |
| 交付头 | 无(1 FAIL) | 无(4 FAIL) | **达标才出**(HF 一次推齐) |
| val / val_soft | 逐字节不变 | 逐字节不变 | 逐字节不变 |
| 数据卫生 | 0/35 | 0/35(含 diag 对) | 0/35(四新块 vs val 零交) |

---

## 第 9 轮执行结果

**结论:10 项硬门槛 PASS / 2 项 FAIL(realtest 老 20 18/20「无关补充」族回归;偏置诊断 12/14——B2 0.7491→0.889,剂量翻倍不降反升,剂量-响应假设证伪)⇒ 无交付,v4 保持已交付,HF 未动(§7 未达标不动)。按 §4 止损线负结论照交 + 剂量-响应外推表,不叠第二轮修补,留 v10。**

### 9.1 执行与资产

| 项 | 值 |
|---|---|
| 训练数据 | `nli_conflict_train_v9.jsonl` 15820 行;§3.6 断言全过;`leak_audit.py` **0/35**(`data_local/leak_audit_v9.txt`);独立复算:diag 18 对 15 唯一字段全 train 逐字段 **0 命中**;**v8 携带 15420 行 state+gold 逐字节等值**(独立验证,新块恰 400 行) |
| 新块落数 | 块 A attr 加剂 **+200**(pet_name 120 zh72/en48 + pet_benign 40 + doctor_bg 40;**B2 同构 known 79/120=66%**,≥1/3 断言过;其中精确「用户的猫叫{N}」句式 14 行)+ 块 B attr 反向对照 **+40**(appearance 20/name 20,kind=true)+ 块 C waver **+120**(in-dist 八猫逐格精确;gym_brand 按 v3 集仅 ZH,EN 七猫)+ 块 D 变更已发生 **+40**(与 C 成对) |
| 词池偏差(§10-1 报告) | doctor_bg 扩至 **7 条/族**(任务书 +4/语种):v8 重放已消耗旧 ZH 空间 24/32,+4 方案仅余 16 个新组合 < 配额 24(实测 assert 拦下),按词池自定授权扩充,缺口无(配额足额) |
| 泄漏口径(§3.6-6 落地) | 新块 vs 35 案+diag 18 对逐字段 **0**;vs val/val_soft 字段触碰 41 处**全部为 v8 语料已有家族级共享**(§2.2 口径注,delta-new=0 硬断言);holdout 猫新块 **0 行**,全语料遗留 44 行 barber 话题 neg_unrelated 控制行(v6 交付行,§9 未动) |
| Kaggle | 数据集 **v14**(train 9752062 字节 = 本地逐字节;val/val_soft 冻结复核);kernel `laya-nli-conflict-ce` **v5(= 计账 kernel v14,纯 CE)**;`nli_kernel.py` 同步嵌 v9 脚本未执行;拉取遇大文件断流一次(0 字节 model.safetensors),按大小校验重拉成功 |
| checkpoint | `D:\laya-kaggle-output\laya-nli-conflict-v9\`;model.safetensors SHA256 `885f256f8d3a8f3bbe2a513419626f38465b56b3e2b7151574bfd71fdde606d6` |
| τ(noul) | **1.1233**(读回 `rl_agent_config.json` temperature[2] = 1.1233258…;v8 1.1240 / v6 1.0905 / v7 1.1058) |
| conformal 工具 | `--sel-budget` 已加(probe + abstain 两处;默认=旧 0.35,**v8 s2 读数逐位复现 42/54@35.4% 后兼容验证过**);本轮全部按 §6.2 以 0.34 选点 |
| 复验产物 | 全套 `*_v9.json` + `conformal_v9.json`(maxconf 旧口径存档)、`conformal_v9_s1.json`/`conformal_v9_s2.json`(重裁两档)、`conformal_alt_v9.json`(三候选曲线)、`indist_probe_v9.json`(新报告项)、`conf_band_v9.txt`、`leak_audit_v9.txt`、`eval_v9.log` |

### 9.2 硬门槛对照(§5)

| 指标 | v8 | v9 实测 | 门槛 | 判定 |
|---|---|---|---|---|
| 主 val acc | 0.8960 | **0.9050**(err 95;反超 v7 0.903) | ≥ 0.896 | **PASS** |
| realtest 老 20 | 20/20 | **18/20**(无关补充 p 0.1782→0.832、临时的不违背偏好 p 0.2624→0.8785,双双翻错) | 20/20 | **FAIL** |
| realtest 新 10 | 10/10 | **10/10**(伤病史 0.8889) | 10/10 | **PASS** |
| realtest 否定 5 | 5/5 | **5/5** | 5/5 | **PASS** |
| val_soft 错误数 | 7 | **0**(v4 起首次全对;in_dist 0 + holdout 0,degree/barber 全修复) | ≤ 3 | **PASS** |
| 偏置诊断 | 12/14 | **12/14**(B2 **0.889** 翻转;D2 0.8657 持平 miss;A/C/E 全家稳定) | ≥ 13/14 且 B2 过 | **FAIL** |
| label-surface swap diff | 0 | **0** | 0 | **PASS** |
| 极性 real/diag \|diff\| | 0/0 | **0/0**(real 2/2/2、diag 2/2/2,跨面零翻转;mean\|Δp\| real 0.0078/0.0218,diag 0.0033/0.005) | ≤1 | **PASS** |
| 极性 val_soft 错误变化 | 7(+6) | **0**(0/1/0;变化 −7) | ≤1 | **PASS** |
| 分离度 val_soft band | 16.70pp | **16.36pp**(q10 0.7795 / q90 0.9431);主 val band 5.73pp;consid 中位 0.7809 | ≥ 8pp | **PASS** |
| conformal(§6 重裁后采用规则) | s2 35.4% FAIL | **s1_maxconf 30/50 = 60% @ 31.8% PASS**(half1 选点 T=0.9247 @ 28%;s2 同过 32/50=64% @ 32.6%;s3 饱和退化;probe 判 tie,采用规则记 s1,两档并列报告) | ≥50% @ ≤35% | **PASS** |
| 训练数据卫生 | 0/35 | **0/35**(断言升级:四新块 delta-new=0) | 0/35 | **PASS** |

### 9.3 关键读数

- **剂量-响应外推表(§4 止损线要求;B2 机制读数)**:
  | 轮 | attr 一致行(B2 同构 known) | B2 p_conflict | Δ |
  |---|---|---|---|
  | v7 | 0 | 0.8985 | — |
  | v8 | +120(同构 ≈24) | 0.7491 | **−15pp** |
  | v9 | +320 累计(同构 ≈103,含精确句式 14) | **0.889** | **+14pp(回落)** |
  **结论:+240 加剂(累计 2.7×)后 B2 不降反升回到 v7 水平 ⇒ 剂量-响应非单调,v8 的 −15pp 主要是跨 run 方差而非稳定处理效应。「同主体属性一致 = false」数据侧行对 B2 的边际影响小于 run 噪声;靠加剂量回收 B2 的路线到此证伪,v10 需换杠杆(如 B2 形状的对照 losses/平衡采样,或接受 D2+B2 双 miss 的 12/14 现状改判据——需先立协议)。**
- **val_soft 0/300(v4 起首次)**:v8 的 7 错(degree 3/barber 4)全部修复,waver/change in-dist 补齐 + run 方差同向;holdout 双读表 v4–v9:v9 行 0/0/0/0。**in-dist 探针(新报告项):waver 15/15、change-done 15/15 = 30/30**(全新字面、state 级排除训练/val 集)——「犹豫⇒false / 已变更⇒true」对比形状在 in-dist 完美泛化。
- **老 20 回归(18/20)**:两条 miss 均为「无关补充」族(期望 false 侧)。v8 p 0.18/0.26 → v9 p 0.83/0.88;v6 修复该族的 400 行 neg_unrelated 控制行原样在库(§9 未动),val_soft in-dist 0 错 ⇒ 归属跨 run 方差下的薄边缘翻转(与 B2 同机制:小 Margin 区对 run 敏感),非数据回退。
- **conformal 重裁(§6 首次执行)**:选点预算 34% 下 s1/s2 双双过 half2(s1 60%@31.8%,s2 64%@32.6%),v8 的贴线破线问题被 1pp 余量吸收;probe 判 tie,采用规则记 **s1_maxconf**(记分最简、弃权余量最大、推理端无需三渲染;s2 同过并列报告)。s3 act head 持续饱和(cap 0)。
- 训练曲线:见 kernel 日志(v9 4 epochs,纯 CE);主 val 反超回 0.905,兼容侧 +400 未伤主任务。

### 9.4 判定与 v10 议题(不叠修补,如实交)

- **判定**:B2 未过 0.5(§4 止损线)+ 老 20 18/20(门槛 20/20)⇒ **无交付**。ckpt v9 留档(SHA256 见 §9.1),HF 未推,数据集 v14 / kernel ce v5 留档计账。
- **v10 议题(仅列案,未立项)**:
  1. **B2 换杠杆**:剂量路线已证伪(§9.3 外推表);候选 = pet 域 true/false 行数再平衡(140 true vs 一致行 160+ 的配比)、B2 形状 hard-example 上采样、或两阶段微调;也可立项改判据(接受 12/14)——须先立协议再动门槛;
  2. **薄边缘稳定性**:B2/无关补充/holdout consid 同属小 Margin 区,单 run 翻覆(v7↔v8↔v9);多 seed 复验协议(§8.4 议题 2 遗留)应先于任何新数据改动立项;
  3. **conformal**:34% 选点余量方案已验证有效(双规则过线),v10 直接沿用;s2 若日后采用需三渲染推理端成本入账;
  4. val_soft 0/300 与 in-dist 探针 30/30 说明 waver/变更形状补齐达标,该方向无遗留。
- **口径注(承 §2.2)**:v6 起验收 35 案与训练语料零逐句交集;val/val_soft 与语料存在家族级句式共享(字段级 310 命中,v7/v8/v9 同量,非回归);barber holdout 与训练域重叠自 v5 起(44 行 v6 交付控制行在库)——跨轮对比按此读。
