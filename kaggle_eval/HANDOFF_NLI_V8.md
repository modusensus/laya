# HANDOFF_NLI_V8.md — 第 8 轮任务书:属性一致控制行(回收 B2)

> 对象:新一轮数据改造 + 训练 + 复验。开工前必读:`HANDOFF_NLI_V3.md`(温度扫描结论)、
> `HANDOFF_NLI_V4.1.md`(判据与交付)、`HANDOFF_NLI_V5.md`、`HANDOFF_NLI_V6.md`、
> `HANDOFF_NLI_V7.md`(**§7.5 = 本轮唯一失败源的归因与修法提案;§5.2/§7.4 口径注**)。
> 结果回写本文末「第 8 轮执行结果」节。

---

## 1. 本轮定位(一句话)

**唯一目标:回收偏置诊断 B2(v7 唯一 FAIL)**——数据侧只加「同主体属性一致 = false」小族(+120),
其余全部冻结保持;达标即 HF 一次推齐(v4–v8 对比卡 + 权重)。

### 1.1 失败清单(唯一项)

| 项 | v6 | v7 | 本轮要求 |
|---|---|---|---|
| 偏置诊断(总 14 对) | 13/14 | **12/14 FAIL** | ≥13/14,且 **B2 必须过** |
| B2 名字一致 | p 0.1215 ✓ | **p 0.8985 ✗(翻转)** | 回 false(p < 0.5) |
| D2 其他坚果无关 | 0.8775 ✗ | 0.8663 ✗(持平) | 允许照旧 miss(13/14 留有 1 席) |

- B2 全对:`known「用户的猫叫小白」/ new「用户的猫是白色的」,期望 false(一致)`。
- 归因(v7 §7.5,机制推断非消融):pet 族 200 行(140 true)后,模型对「宠物 known + 宠物相关 new」
  倾向 true;B2 是全 diag 唯一的「宠物 known + 宠物属性 new」对照。
- 修法:**补「同主体、非冲突的附加属性」行**(kind=compat,软目标承 0.93±0.02)——
  **形状可借、字面全新**;不得使用 小白 / 白色 / 开心果 等 diag 字面。

---

## 2. 预核算(2026-09-29;从 v7 落盘复算,执行者按表复核)

### 2.1 失败面读数(来自 `noul_bias_diag_v7.json`)

- B2:0.1215(v6)→ **0.8985**(v7),唯一新翻转;D2:0.8775 → 0.8663(持平 miss)。
- 全 diag + swap 共 18 对、36 个字段 vs `nli_conflict_train_v7.jsonl` 逐字段等值扫描:
  **全部 0 命中**(含「用户住在上海」「用户今年28岁」等,值替换已清)——本轮断言按此升级后应直接通过。

### 2.2 必须保持的 11 项(v7 实测,复验侧已复算)

| 指标 | v7 实测 | 保持要求 |
|---|---|---|
| 主 val | 0.9030(err 97;q10 0.8850 / q90 0.9435) | ≥ 0.896 |
| 老 20 | 20/20 | 20/20 |
| 新 10 | 10/10(猫粮 0.8802 / 伤病史 0.8819) | 10/10 |
| 否定 5 | 5/5 | 5/5 |
| val_soft | 1(row18 degree,p 0.3074) | ≤ 3 |
| swap | 0(real 20/20+20/20,diag 12+12) | 0 |
| 极性 real/diag | 0/0 · 0/0 | ≤ 1 |
| 极性 val_soft | 1/1/1 | 变化 ≤ 1 |
| 分离度 band | 16.64pp | ≥ 8pp |
| conformal(采用 s2) | 34/51@33.4% | ≥50% @ ≤35% |
| 数据卫生 | 0/35 | 0/35 |

---

## 3. 数据改造(`kaggle_eval/gen_soft_conflicts_v8.py`;「照着 v7 改」)

输出 `data_local/nli_conflict_train_v8.jsonl`。**val 与 val_soft 不重生成**(逐字节不变)。

### 3.1 唯一新增块:「同主体属性一致」(+#120;#1)

- **语义**:`known` 与 `new` 描述**同一主体**(宠物 / 当事人),两条陈述可同时为真、不违约束 ⇒
  目标 **false(兼容)**。
- **三个形状(全用)**:
  1. **姓名-外观一致**(B2 同形状、换字面):「用户家的猫叫{名}」/「用户的猫是只{色}猫」——
     名池 ≠ 小白(建议:咪咪/汤圆/年糕),色池 ≠ 白色(建议:黑/橘/灰/三花);
  2. **被约束主体 + 良性属性**(§7.5 提案形状):「用户家的狗对牛肉过敏」/「用户家的狗是柯基」
     ·「…是去年领养的」·「…今年三岁」;
  3. **医生语境 + 背景事实**:「用户胃炎,医生要求忌辛辣」/「用户去年做过一次胃镜」·
     「用户在控血压,医生叮嘱少盐忌酒」/「用户口味一直偏清淡」。
- **配额(写死,assert 精确,逐形状计数落报告)**:**合计 +120** = pet_attr **80**
  (zh 48 / en 32)+ doctor_attr **40**(zh 24 / en 16),lang_mix {ZH: 0.60, EN: 0.40}。
- **kind / 目标**:`kind='compat'`(用 `TARGETS['compat']` = 0.93±0.02 现档);
  `meta`:`cat='pet_attr'|'doctor_attr'`、`shape='attr_consistent'`、lang、labelset(6-labelset 轮换)。
- **护栏**:不得含违逆标记词(不顾/偏要/硬要/执意/照样);不得与 35 案、diag 14 对、val/val_soft
  任何字段等值;**字面禁词:小白、白色、开心果**。

### 3.2 少动原则(复用优先)

- v7 已有 `__main__` guard(导入无副作用):**凡 v7 模块级可导入的表/函数一律
  `from gen_soft_conflicts_v7 import …` 复用**(PET_HC、DOCTOR_HC、NEW_HC_SUBFAMS、ALT_PRAISE_T、
  ALT_PRAISE_FAMS、make_alt_praise_rows、METRIC_CATS、make_metric_rows 等),不要复制粘贴;
  main 骨架照 v7 改,只插新块。
- **alt_praise 双语分支 = 已知死代码**(复验注记:双语分支 `len(langs)==2` 恒假,输出 240/240 为 ZH):
  **本轮不修**(数据保持与 v7 逐字节同);EN 侧覆盖另议。

### 3.3 断言(承接 + 新增;全过才许写输出)

1. 行数 = **15420**;`state` 全局唯一;与 val / val_soft 零交集(承 v7 实现)。
2. kind 计数:compat 1480 → **1600**,其余全部不变;新块逐族计数 pet_attr 80 / doctor_attr 40。
3. labelset 配比:语义 4 套合计 ≥50%、LS5/LS6 各 20%(数值按 15420 改算,assert 精确)。
4. 每行 `meta` 完整(`cat/kind/lang/labelset`;新块含 `shape='attr_consistent'`)。
5. `gold.label == argmax`;`|p(gold) − 0.5| ≥ 0.2`;档位封闭。
6. **泄漏断言(升级)**:35 案字段 + **diag 14 对照 + 4 交换对的 known/new 逐字段等值 = 0**。
   diag 对从 `noul_bias_diag` 导入(`from noul_bias_diag import CONTROL_PAIRS, SWAP_PAIRS`)——
   该模块有 `__main__` guard、`import laya` 为惰性包,不会触发 torch,可安全导入。
   §2.1 已证 train_v7 对 18 对字段零命中,升级断言应直接通过;任何命中 = 回归,查新块。
7. 新块禁词扫描(小白 / 白色 / 开心果)= 0。

### 3.4 规模表

| 块 | v6 | v7 | v8 |
|---|---|---|---|
| MNLI(整块逐字节不变) | 6000 | 6000 | 6000 |
| 合成软冲突(含 alt_praise 240 + metric 160) | 5400 | 5800 | **5920**(+attr 120) |
| 硬约束(含 pet/doctor) | 600 | 1000 | 1000 |
| 否定算子族 | 1600 | 1600 | 1600 |
| known 一步扰动(cf) | 900 | 900 | 900 |
| **合计** | 14500 | 15300 | **15420** |

---

## 4. 训练改造(承 v7,零改动)

- **主臂 = 纯 CE**:EPOCHS=4、MICRO_BATCH=16、GRAD_ACCUM=2、LR 承 v4/v5、SIGMA 0.4→0.1、底座不变
  (`convaiinnovations/laya-multilingual`)。RL 项代码保留、不跑。
- CLI:`--expected-train 15420 --max-items 16000`(防静默截断)。
- 尾部承 v5/v6/v7:400 条 calib(seed 20260927 不变)→ τ 拟合 → 落盘
  `model.safetensors`/`encoder`/`tokenizer`/`rl_agent_config.json`/`metrics.json`;
  val 1000 逐行 `{gold, p_true}` 落盘 `val_probs_v8.json`。
- **τ 报告纪律(新)**:τ 拟合值**直接读回 `rl_agent_config.json` 的 `temperature[2]` 写进报告**——
  v6/v7 报告曾误写默认值 1.200(实值 v6 1.0905 / v7 1.1058),勿重蹈。

**止损线(先写死)**:主 val < **0.885** ⇒ 判 FAIL 照报不叠改动;**泄漏断言失败 ⇒ 立即停**;
新 10 <10/10、否定 <5/5、val_soft >3、极性变化 >1 ⇒ 对应轴 FAIL 照实报、逐条列 p;
**B2 修复不成(仍 12/14)⇒ 负结论照交,不叠第二轮修补**(留 v9 决策)。

---

## 5. 门槛(承 v7 §5.1,只准加)

| 集合 / 指标 | v7 实测 | v8 门槛 | 说明 |
|---|---|---|---|
| 主 val acc | 0.903 | ≥ 0.896 | 不放宽 |
| realtest 老 20 | 20/20 | 20/20 | |
| realtest 新 10 | 10/10 | 10/10 | pet 轴 true 侧由此案兜底(猫粮) |
| realtest 否定 5 | 5/5 | 5/5 | |
| val_soft 错误数 | 1 | ≤ 3 | |
| 偏置诊断 | 12/14 | **≥13/14 且 B2 必须过** | D2 允许 miss |
| label-surface swap diff | 0 | 0 | |
| 极性 real/diag \|diff\| | 0,0 | ≤1 | |
| 极性 val_soft 错误变化 | 1/1/1 | ≤1 | |
| 分离度 val_soft band | 16.64pp | ≥ 8pp | |
| conformal(采用 s2) | 34/51@33.4% | ≥50% @ ≤35% | §6 复跑口径 |
| 训练数据卫生 | 0/35 | 0/35 | 断言全过 + `leak_audit.py` 0 命中 |

**报告项(不设硬门槛)**:attr 块逐形状计数 + 2–3 条新字面示例(与 B2 同形状字面异);
分离度全套 + consid 中位置信;holdout 双读表(v4–v8 同表);极性 mean\|Δp\| 三集;τ 拟合值;
逐族计数;训练曲线。**口径注(必写进报告)**:v6 起验收 35 案与训练语料零逐句交集;
barber holdout 与训练域重叠自 v5 起(§2.2 注);跨轮对比按此读。

---

## 6. conformal 复跑口径(无新候选)

- 候选**冻结 = v7 §6 的三个**(s1 maxconf / s2 表面一致性 / s3 act_probability),
  **不得事后新增**;协议同 v7:seed 20260928 对半、half1 选、half2 判。
- 执行:`python kaggle_eval/conformal_alt_probe.py <v8 ckpt> data_local/conformal_alt_v8.json`;
  门槛脚本:`conformal_abstain.py --rule s2 --alt-scores data_local/conformal_alt_v8.json
  --main … --soft … --real … --diag … --out data_local/conformal_v8_s2.json`
  (**`--alt-scores` 必须显式指 v8 文件**——其默认值仍指 `conformal_alt_v7.json`,忘传 = 静默用旧分)。
  maxconf 对半切存档照 v7(`conformal_v8.json`,命令同 v7,不带 --rule)。

---

## 7. 交付物与归档

**脚本(进仓,`kaggle_eval/`)**:`gen_soft_conflicts_v8.py`(新;含 §3.3 全断言);
其余沿用(**不改判据**);`nli_kernel.py` / `nli_kernel_ce.py` 同步嵌 v8 脚本(不执行)。

**数据产物(`data_local/`)**:`nli_conflict_train_v8.jsonl`(15420);`val_probs_v8.json`、
`val_soft_probs_v8.json`、`polarity_nli_v8.json`、`memory_conflict_realtest_v8.json`、
`noul_bias_diag_v8.json`、`label_swap_v8.json`、`conformal_v8.json`、`conformal_alt_v8.json`、
`conformal_v8_s2.json`、`conf_band_v8.txt`、`leak_audit_v8.txt`。

**Kaggle 资产**:数据集 `daphnelaurent/nli-conflict-pairs` **v13**(train 换 v8;val(1000)/val_soft(300)
冻结;`_README` 加 v13 行);kernel `laya-nli-conflict-ce` 再 push 一版(**计账 = kernel v13**,纯 CE 主臂)。

**模型产物**:`D:\laya-kaggle-output\laya-nli-conflict-v8\`;大小与 SHA256 写进报告。

**HF(达标才推)**:卡(**同基准对比节 v4–v8** + 「验收 35 案与训练语料零逐句交集」句 + 新头 SHA)
+ 新头权重全文件,一次上传;未达标不动、不补推。

---

## 8. 执行环境与命令(承 v7 §8 + 本机新坑)

- ⚠️ **本机 torch 现被 Smart App Control 拦截**(2026-09-29 复核实测:`import torch` → WinError 4551;
  pandas 更早已被拦)。本机全套复验(realtest / polarity / bias / probe / dump / band 均依赖 torch)
  在此状态下**跑不了**。**开工先探测**:`"D:/Miniconda/envs/laya-ft/python.exe" -c "import torch; print('ok')"`;
  若被拦 → **停下报告**(需先处理 SAC;训练在 Kaggle、生成器与离线复算不受影响)。
  诊断:`wevtutil qe "Microsoft-Windows-CodeIntegrity/Operational" /c:200 /f:text /rd:true | grep "attempted to load"`。
- Kaggle CLI 的 tmp 必须指 ASCII(`C:/kaggle_cfg/tmp`);**数据集总挂最新版**;README 改动也要出
  新版本号;**每次 push kernel 会从头重训**(CUDA 非确定性 ⇒ 权重不可位复现;本机留档副本为准);
  `QUICK_SAVE` 纯文档版无输出(status ERROR 属占位)。
- 泄漏审计:`python kaggle_eval/leak_audit.py data_local/nli_conflict_train_v8.jsonl`(期望 0 命中;
  若环境导入受限,可注入假 laya 模块后 exec 运行,见 v7 复验记录)。
- 本机 CPU 复现:`D:/Miniconda/envs/laya-ft/python.exe`;`pandas` 被系统策略挡,一律 `pyarrow`;
  本机→kaggleusercontent 大文件链路间歇断流(重试循环 + 串行下载)。

---

## 9. 不做清单(硬)

- 承 v7 §9 全部:不动 τ 与 0.92 红线;不改 `val`(1000)/`val_soft`(300) 任何字节;不在报告集上做任何
  拟合(§6 的 half1/half2 互斥为唯一例外);不引入外部数据;不把 35 案原文写进训练数据;
  不重修 multilingual 头;不跑 R-Drop;不放宽任何判据、不删既有负结论记录;不许缩范围;
  **half2 与 realtest 不参与任何拟合**;**不得事后新增 §6 候选**。
- 新增:**不修 alt_praise 双语分支**(死代码照旧;EN 覆盖另议);**不动 v7 已交付的
  pet/doctor/alt_praise/metric 行**;**不要求修 D2**;B2 修复不做第二轮修补。

---

## 10. 开放决策(不拍板按默认走)

1. **attr 块配额**:默认 pet_attr 80 / doctor_attr 40(合计 +120);模板空间不足按 v5 先例(补形状)
   解决,不得低于配额 80%,缺口逐族写报告。
2. **词池细节**:名池/色池/品种池执行侧自定(守 §3.1 护栏与禁词)。
3. **doctor 语境分配**:默认贴合既有四子族(rest/postop/gastritis/bp 各约 1/4)。
4. **B2 未回收的处置**:负结论照交(不调门槛、不叠修补),v9 再决策。

---

## 附录 A. 版本与文件对照(本轮)

| 对象 | v6 | v7 | v8 |
|---|---|---|---|
| 训练数据 | `nli_conflict_train_v6.jsonl`(14500) | `…v7.jsonl`(15300) | `…v8.jsonl`(**15420**) |
| 数据生成 | `gen_soft_conflicts_v6.py` | `gen_soft_conflicts_v7.py` | `gen_soft_conflicts_v8.py` |
| conformal 工具 | + `--split-half` | + `conformal_alt_probe.py`、`--rule s2` | 沿用(v8 分档复跑) |
| Kaggle 数据集 | v11 | v12 | **v13** |
| Kaggle kernel | v11 | v12 | **v13**(ce 再推一版) |
| 交付头 | 无(4 FAIL) | 无(1 FAIL:B2) | **达标才出**(HF 一次推齐) |
| val / val_soft | 逐字节不变 | 逐字节不变 | 逐字节不变 |
| 数据卫生 | 0/35(断言) | 0/35(断言) | 0/35(断言,升级含 diag 对) |

---

## 第 8 轮执行结果

(执行者填写:执行与资产 / 硬门槛对照 / 关键读数 / 判定;格式照 v7 §7。)
