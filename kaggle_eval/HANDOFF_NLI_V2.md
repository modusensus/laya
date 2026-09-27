# Laya NLI 记忆冲突检测头 — v2 修复任务交接

> 给下一个执行者的完整上下文。所有文件路径、数字、坑都是实测得来，不是猜的。

## 任务一句话

Laya 0.3B 判别模型微调出的「记忆冲突检测头」v1 在真实场景不及格（9/20），按上游 issue 的已验证方案做 v2：**换标签词 + 修数据分布**，目标是真实 case 实测 ≥85%。

## 背景（1 分钟版）

- Laya 是 0.3B 判别式决策模型，30ms 出决策，固定选项。
- 我们在 Kaggle 2×T4 用 MultiNLI 蕴含/矛盾对微调了 noul 头，验证集 accuracy 0.935 / ECE 0.069。
- 但 20 条中文真实 case 实测只有 9/20：真冲突 7/8 对，不冲突 2/12——**逢问必"冲突"，conf=0.91**。
- 根因（两个，叠加）：
  1. 训练数据把 MNLI 的 neutral 类全扔了，模型没见过"无关"；
  2. 上游 #156 实锤的布尔标签词偏置：`true`/`false` 词对会碾过输入内容。

## 上游 issue 依据（都已核实原文）

| Issue | 结论 | 对我们的动作 |
|---|---|---|
| [#156](https://github.com/NandhaKishorM/laya/issues/156)（closed, completed） | `true/false`、`yes/no` 标签对让模型无视 state 永远答 negative，三人独立复现，维护者称"最重要 open bug"；#163 已合并修复=允许自定义 labels | v2 的 noul 问题改用 `labels: {"false": "兼容", "true": "冲突"}` |
| [#377](https://github.com/NandhaKishorM/laya/issues/377)（open） | 否定句失效（"do NOT cancel"→cancel 0.9998），模型层限制无修复 | 实测集单独加否定句 case；接插件时不可逆动作走人工 |
| [#238](https://github.com/NandhaKishorM/laya/issues/238) | RLCD 配方本质是监督学习，维护者承认；社区在做 CE-only 对照 | 可选：v3 试 CE-only，训练更快更稳 |
| #220 | `tools/noul_diag.py` 偏置诊断工具的 PR | v2 训练完先跑偏置诊断（正/负对照对）再实测 |

## 现有资产（全在本机，已就位）

| 资产 | 路径 |
|---|---|
| v1 checkpoint（1.3GB，含 metrics.json） | `D:\laya-kaggle-output\laya-nli-conflict-v1\` |
| 训练数据（6k train + 1k val JSONL） | `D:\laya\data_local\nli_conflict_{train,val}.jsonl` |
| 训练脚本（DDP，官方 RLCD 配方） | `D:\laya\kaggle_eval\train_nli_conflict.py` |
| Kaggle kernel 主文件（含内嵌训练脚本 b64） | `D:\laya\kaggle_eval\nli_kernel.py` |
| kernel metadata | `D:\laya\kaggle_eval\nli_kernel_metadata.json` |
| Kaggle 数据集（已上传） | `daphnelaurent/nli-conflict-pairs` |
| Kaggle kernel（v3 已跑通全流程） | `daphnelaurent/laya-nli-memory-conflict-fine-tune` |
| 真实 case 实测结果 | `D:\laya\data_local\memory_conflict_realtest.json` |
| 本地 conda env（torch CPU/transformers/laya 可编辑安装） | `D:\Miniconda\envs\laya-ft\python.exe` |
| 实测脚本模式 | 见下文「实测脚本」节 |

## v2 改动清单（核心不到 20 行）

### 1. 数据：保留 neutral，重新生成 JSONL

MNLI 取数时：
- 蕴含 (entailment) → gold `false`（不冲突，互补）
- **neutral → gold `false`（不冲突，无关）← v1 就是漏了这批**
- 矛盾 (contradiction) → gold `true`（冲突）
- 比例调成 矛盾:非矛盾 ≈ 1:2（模拟真实记忆流里"大部分新信息不冲突"的先验）
- 数量不变：6k train + 1k val 足够（v1 训练只用了 330 秒）

数据集脚本把 `{"known":..., "new":...}` 两句塞进 state JSON 字符串、questions 用 `conflict` noul、gold 带 probability——**沿用 v1 的 JSONL 结构一行不改**，只换抽样逻辑。

### 2. 标签词：noul 问题换语义标签

v1 JSONL 里 questions.conflict 是：
```json
{"type": "noul", "instructions": "新信息(new)与已有记忆(known)是否冲突？true=冲突，false=不冲突/互补"}
```
v2 改成：
```json
{"type": "noul", "instructions": "新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关", "labels": {"false": "兼容", "true": "冲突"}}
```
（`labels` 键即 #163 合并的修复；先用 `laya.load()` 加载 v1 ckpt 在本地验证 `labels` 键被接受、输出变成中文词，再重新生成数据。）

### 3. 训练：复用现有 kernel，只换数据

- 重新生成两个 JSONL → `kaggle datasets version -p <dir> -m "v2 neutral included"` 更新数据集
- kernel 不用改逻辑（它是从挂载路径读 JSONL 的），直接 `kernels push` 重跑
- **注意 kernel 的三个坑（v3 已趟平）**：
  1. Kaggle 脚本 kernel 只暂存主代码文件 → 训练脚本必须内嵌（nli_kernel.py 里已是 b64 内嵌，改 train_nli_conflict.py 后要重新 b64 替换内嵌块）
  2. torchrun 传参必须用命名参数（`--model-dir` 等），v2 就死在位置参数上
  3. 409 冲突=旧版本还在 saving/running，等或删了重建
- 轮询：`kernels status` → COMPLETE 后 `kernels output` 拉 `/kaggle/working/laya-nli-conflict/`

### 4. 验收（硬门槛，不许降）

1. Kaggle val accuracy ≥ 0.90（v1 是 0.935，v2 不应倒退）
2. **偏置诊断 PASS**：正/负对照对（如"用户住在上海"+"用户搬到北京" vs "用户住在上海"+"上海有很多大学"）在交换输入时输出必须跟着输入变，而不是钉死一个标签
3. **真实 case 实测 ≥ 16/20（85%）**——用下面的 20 条 case 原样重跑，注意其中"无关补充/无关陈述"这几条 v1 全错，v2 必须翻正
4. 否定句 case（新增 3-5 条，如"用户没有养宠物" vs "用户的狗昨天打了疫苗"）单独统计，准确率不作硬门槛但必须报告

### 5. 交付物

- v2 checkpoint（Kaggle output 拉回，放 `D:\laya-kaggle-output\laya-nli-conflict-v2\`）
- `data_local/memory_conflict_realtest_v2.json`（20+否定句 case 逐条结果）
- 一段 5 行以内的数字汇报：Kaggle val acc / 偏置诊断结果 / 实测 n/20 / 否定句 n/N

## 实测脚本（原样可跑，改 checkpoint 路径即可）

```python
import json, laya
CASES = [  # (state, 期望, 说明) — 与 v1 实测完全同一条
    (({'known': '用户住在上海', 'new': '用户最近搬到北京工作了'}, 'true'), '地点变更'),
    (({'known': '用户主要用 Python 写代码', 'new': '用户现在主力语言换成了 Rust'}, 'true'), '偏好变更'),
    (({'known': '用户每周三去健身房', 'new': '用户的健身时间改到了每周五'}, 'true'), '时间变更'),
    (({'known': '用户对花生过敏', 'new': '用户说他可以吃花生酱了'}, 'true'), '关键事实反转'),
    (({'known': '用户的项目部署在阿里云', 'new': '项目已经迁移到腾讯云了'}, 'true'), '部署变更'),
    (({'known': '用户喜欢喝美式咖啡', 'new': '用户现在只喝拿铁'}, 'true'), '口味变更'),
    (({'known': '用户的猫叫小白', 'new': '用户的猫叫小黑'}, 'true'), '名字矛盾'),
    (({'known': '用户今年 28 岁', 'new': '用户今年 35 岁'}, 'true'), '年龄矛盾'),
    (({'known': '用户喜欢喝咖啡', 'new': '用户今天早上喝了一杯咖啡'}, 'false'), '一致行为'),
    (({'known': '用户养了一只猫', 'new': '用户的猫昨天打了疫苗'}, 'false'), '补充细节'),
    (({'known': '用户在学人工智能', 'new': '用户今天学了微积分'}, 'false'), '无关补充'),
    (({'known': '用户住在北京', 'new': '北京有很多历史古迹'}, 'false'), '无关陈述'),
    (({'known': '用户是产品经理', 'new': '用户在做一款记账 App'}, 'false'), '工作相关'),
    (({'known': '用户每天早上七点起床', 'new': '用户今天七点起床后去跑步了'}, 'false'), '习惯一致'),
    (({'known': '用户喜欢科幻电影', 'new': '用户昨晚看了一部科幻片'}, 'false'), '偏好一致'),
    (({'known': '用户用 Windows 电脑', 'new': '用户给电脑装了新键盘'}, 'false'), '兼容细节'),
    (({'known': '用户喜欢安静的环境', 'new': '用户在考虑去热闹的音乐节'}, 'false'), '临时的不违背偏好'),
    (({'known': '用户不吃辣', 'new': '用户今天想尝试一下微辣'}, 'true'), '饮食禁忌松动'),
    (({'known': '用户有一个妹妹', 'new': '用户是家里的独生子'}, 'true'), '家庭结构矛盾'),
    (({'known': '用户住在上海浦东', 'new': '用户在上海上班'}, 'false'), '子集关系'),
]
agent = laya.load(r'D:\laya-kaggle-output\laya-nli-conflict-v2', device='cpu')
correct = 0
for (state, expect), note in CASES:
    a = agent.predict(state, {'conflict': {'type': 'noul',
        'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
        'labels': {'false': '兼容', 'true': '冲突'}}})['answers']['conflict']
    # v2 输出的键是中文标签「冲突/兼容」，取概率时注意
    got = 'true' if a.get('冲突', a.get('noul', 0)) >= 0.5 else 'false'
    ok = got == expect
    correct += ok
    print('OK ' if ok else 'MISS', note, got, a)
print(f'TOTAL: {correct}/{len(CASES)}')
```

（predict 返回结构若与上述键名不符，以 `print(a)` 实际输出为准调整，不要瞎猜键名。）

## 环境与凭据注意事项

- Python 一律用 `D:\Miniconda\envs\laya-ft\python.exe`（torch 2.14 CPU + transformers + laya 可编辑安装）。
- kaggle CLI 在含非 ASCII 字符的路径下会挂：临时目录必须指到 ASCII 路径（v3 用的是 `C:/kaggle_cfg/tmp`），token 从 `%USERPROFILE%\.kaggle\kaggle.json` 读后以 `KAGGLE_API_TOKEN` env 传入。
- HF/Kaggle token 一律走环境变量，不要写进任何文件或日志；用完提醒用户吊销。
- 用户偏好：中文交流；诚实汇报失败；重要决策先给数字和门槛再动手。

## 明确不要做

- 不要动 `laya/` 核心库（本次任务范围只有数据 + kernel + 实测）。
- 不要尝试把任务改成多分类或换模型架构——上游 #156 的结论是标签词问题，先按已验证方案走。
- 不要在没跑偏置诊断的情况下直接报"修好了"。
- 实测 case 必须原样用上面 20 条（可加不可减），否则和 v1 的 9/20 没有可比性。

## v2 执行结果(2026-09-27)

三轮训练(Kaggle kernel `daphnelaurent/laya-nli-memory-conflict-fine-tune` v2/v3/v4,数据集同名 v2 数据已传两个版本):

| 候选 | 配方 | val acc | 偏置诊断(修正集) | 真实 case | 否定句 |
|---|---|---|---|---|---|
| e3 | 3 epochs | 0.895 | 11/14(不钉死) | **18/20(90%)** | 3/5 |
| **e4(交付)** | 4 epochs | **0.902 ✓** | 12/14(不钉死) | 16/20 | 3/5 |
| e3r | 3 epochs,train 重抽样 | 0.895 | 10/14 | 15/20 | 3/5 |

- **交付 checkpoint**:`D:\laya-kaggle-output\laya-nli-conflict-v2\`(= e4;val_ece 0.021)
- **结论**:布尔词钉死偏置已修复(#156);"逢问必冲突"翻转(非冲突 case v1 对 2/12 → v2 对 11/12)
- **未过项(如实)**:真实 case 16/20 = 80%,未到 85% 目标;偏置诊断 12/14——残留错误集中在「用户住在上海 vs 搬到北京工作」这类**时间性/变更型软冲突**,两个方向都错(e3/e4/e3r 一致)。根因是数据分布:MNLI 矛盾对几乎全是词汇否定,没有"状态更新"型冲突。若要翻正 A1 类 case,需要在数据里合成软冲突对(超出本交接的"只换抽样逻辑"范围,留给 v3;#238 的 CE-only 对照也可一并试)
- **验收/诊断脚本**(均在 `kaggle_eval/`):`noul_bias_diag.py`(正/负对照+输入交换,14 例)、`realtest_v2.py`(20 条+5 条否定句)、`val_breakdown.py`(val 逐类:矛盾类召回是瓶颈,e4 为 85%)、`gen_nli_conflict_data_v2.py`
- **注意**:全仓库已迁至 `D:\laya`(可编辑安装已重指 `pip install -e D:/laya`);`%USERPROFILE%\Desktop\laya` 只剩空壳(进程占用,重启后可删);Kaggle CLI 2.2.4 走 Python API + `KAGGLE_API_TOKEN`(内容为 `.kaggle/kaggle.json` 的 key),CLI 入口的认证不可用;临时目录仍指 `C:/kaggle_cfg/tmp`

### Kaggle 资产版本对照(防混淆,必读)

| kernel 版本 | 数据集版本 | 产出 | 状态 |
|---|---|---|---|
| v1 | v1(neutral 丢弃) | v1 头(9/20) | 已废弃 |
| v2 | v2(seed 20260927) | e3:0.895 / 18/20 | 备选 |
| **v3** | **v2** | **e4:0.902 / 16/20 = 交付版** | **交付** |
| v4 | v3(train 重抽样 seed 20260928) | e3r:0.895 / 15/20 | 已否决 |
| — | v4(= v3 内容 + _README.md) | 无训练,仅文档 | 当前数据集最新版 |

- ⚠️ CLI `kernels output` 只拉**最新** kernel 版本(当前 v4 = 已否决的 e3r)。
  **取交付 checkpoint 用本地 `D:\laya-kaggle-output\laya-nli-conflict-v2\`,或网页端下载 kernel version 3 的 output。**
- kernel 页面本身没有描述字段,源码即文档;数据集页的 subtitle/description 已写全
  (含上述对照表摘要),数据集内 `_README.md` 随下载走。
- 更新数据集页描述:改 `kaggle_eval/nli_data/dataset-metadata-page.json` 后调
  `dataset_metadata_update`(参考 C:\kaggle_cfg\stage\nli_data 的现成配置);
  `keywords` 字段会被 Kaggle 拒(非受控词表),不要加。

### 2026-09-27 补充:kernel v5(文档版 quick save)

- kernel 最新版为 **v5**:用 SDK 的 `KernelExecutionType.QUICK_SAVE` 推送的**纯文档版**
  (存版本不运行,零 GPU)。源码头部带完整 VERSION MAP(含数据集版本对照)和两个坑的说明;
  内嵌脚本已恢复 EPOCHS=4,与交付版 v3 配方一致。
- **`kernels status` 现在显示 ERROR 是正常的**:那是"未执行版本"的占位状态,不是失败
  (failureMessage 为空、无输出)。同理 **`kernels output` 对 v5 只返回源码日志**,拿不到模型——
  这是有意的防呆:CLI 默认拉最新版,再也不可能静默拉到被否决的 e3r。
- **拿交付 checkpoint 的两个途径**:本地 `D:\laya-kaggle-output\laya-nli-conflict-v2\`
  (= kernel v3 输出),或 kernel 网页 Versions → v3 → Output 下载。
- 若日后要重训:注意数据集挂载永远是最新版(v4 = train 重抽样 seed 20260928 + README),
  与交付版 v3 用的原始抽样(seed 20260927)不同,直接 push 不会复现交付权重;
  先用 `gen_nli_conflict_data_v2.py` 以 SEED=20260927 重新生成上传再 push。

## v3 数据工位(2026-09-27,只做数据,训练未启动)

按"软冲突数据无论如何不亏,训练载体后定"的决策,先完成数据:

- 生成器:`kaggle_eval/gen_soft_conflicts_v3.py`(seed 20260929,确定性;每类含 conflict/
  compatible/consideration/unrelated 四种配对句式,20 个属性类 × 中英双语)
- `data_local/nli_conflict_train_v3.jsonl`:9000 条 = v2 的 6000 条 MNLI **原样** ⊕ 3000 合成
  (1000 软冲突 + 800 兼容细节 + 600 变更意图 + 600 无关提及),全局矛盾:非矛盾 = 1:2.00
- `data_local/nli_conflict_val_soft.jsonl`:300 条(100 真/200 假);其中 **4 个属性类
  (email/desk_floor/degree/barber)只在 val 出现**,专门度量对没训过的属性类型的泛化;
  切片信息在 `nli_conflict_val_soft.meta.json`(顺序与行一一对应)
- **交付 v2 模型在 val_soft 上的基线**(`data_local/val_soft_v2_baseline.json`):
  软冲突 78%(meanP 0.717)/ 兼容细节 87% / **变更意图 44.6%** / **无关提及 50%**;
  holdout 类 62% < in-dist 72%。两个弱点与真实 case 的残留错误完全对应
  (「在考虑/想试试」被当成冲突;提到另一个取值就被当变更),训练价值已被基线锁定:
  v3 目标 = true ≥90%(holdout ≥85%)、consid ≥80%、unrelated ≥85%、compat ≥87% 保持

**接手 v3 训练前必读**:
1. kernel 内嵌脚本的 `max_items=8000` 上限必须先改到 ≥9500(9000 条 train 会被截断),改完重新 b64 内嵌;
2. 数据集挂载永远是最新版:上传 v3 数据前想清楚要不要覆盖(当前 Kaggle 最新 = v2 数据 + README,与交付版一致);
3. val(主 1000 条)保持 v2 原样不动,保证与 0.902 可比;soft 指标看 val_soft;
4. 诚实提醒:合成模板的属性类是从 20 条真实 case 的缺口反推的,v3 的真实 case 提升属于
   "定向修复验证",泛化能力看 val_soft 的 holdout 切片;验收时最好再补几条全新的真实 case;
5. CE-only 对照(#238)如果要一起做:把 loss 改成 `loss = loss_ce / GRAD_ACCUM` 一行即可。

## v3 训练结果(2026-09-27,交付版更新)

kernel v6 × 数据集 v5(9k train = 6000 MNLI + 3000 合成软冲突,EPOCHS=4,RLCD 配方不变,
`max_items` 8000→9500)→ **v3 头,全门槛通过,接替 e4 成为交付版**:

| 指标 | v2(e4) | **v3** | 门槛/目标 |
|---|---|---|---|
| val(主 1000,与 v2 完全相同) | 0.902 / ECE 0.021 | **0.902 / ECE 0.0146** | ≥0.90 ✓ |
| 偏置诊断(修正集) | 12/14 | **13/14 PASS** | PASS ✓ |
| 真实 case 20 条 | 16/20 | **19/20(95%)** | ≥85% ✓✓ |
| 否定句 5 条 | 3/5 | 4/5 | 报告 ✓ |
| val_soft 软冲突 / 变更意图 / 无关提及 | 78% / 44.6% / 50% | **99% / 93.8% / 100%** | 全达标 |
| val_soft holdout 类(未训练) | 62% | **95%** | 泛化 ✓ |

主 val 逐类无回退(false 92.6% / true 85.3%)。已修:地点变更、临时的不违背偏好、
工作相关、习惯一致全部翻正,「地点变更」在偏置诊断的两个输入方向均为 0.92。
**已知代价**:「饮食禁忌松动」(不吃辣 vs 想尝微辣)翻成 miss——consid 控制组把
"意图≠冲突"学得偏强,遇到"意图违反硬约束"时会漏报;净收益 +3 条。v4 若继续:
给合成块加一类"意图违反硬约束(过敏/安全)"且标注为冲突的对照行。
诚实提醒:20 条真实 case 的属性类曾用于设计合成模板,v3 的真实 case 提升属于定向修复
验证;泛化证据是 val_soft holdout 95%;下次验收建议补全新真实 case。

**Kaggle 资产终态**:数据集 v7(README 含完整对照表)为最新;kernel v6 = v3 训练(交付
输出来源),kernel v7 = 文档版 quick save(最新版,无输出,status ERROR 属占位)。
取交付模型:本地 `D:\laya-kaggle-output\laya-nli-conflict-v3\`,或网页 Versions → v6 → Output。

## v4 待办(2026-09-27 规划,未执行)

上一节结论未动;本节只列 v4 动作项,按性价比排序。

### 1. 立刻可做的两件(跨项目借鉴,纯数据/评测层,不改模型结构)

| # | 动作 | 来源 | 为什么 | 验收 |
|---|---|---|---|---|
| A | **标签面改写增广**:训练行内把 label 的 key/取值在同义集合里随机替换(冲突/矛盾/不一致 ↔ 兼容/一致/无关;key 名也可换) | Hammer 的 function masking(把函数名换成随机串,逼模型读描述) | 同构治我们踩过两次的病:#156 布尔词偏置、v3「意图≠冲突」学太死 | 增广版重跑:主 val 不掉 + 真实 case ≥19/20 |
| B | **标签面交换验收**:把 `labels` 两个取值互换后重跑 realtest + 偏置诊断 | Hammer 的 robustness 评测协议 | 现有诊断只有「输入交换」和正/负对照,缺「标签字面」这一轴;这是唯一能证明学到语义而非标签面的实验 | 交换前后 accuracy 差 ≤1 条 |

### 2. 稀有类按比例放大(不要只调阈值)

Hammer 给「无合适函数可调」(label=[]) 专门加了 7500/60k ≈ 12.5% 的 abstain 样本才稳;
我们的合成块里最小的一类是「无关提及」600/9000 ≈ 6.7%,同档量级,说明配方方向对。
v4 把四类合成块整体 ×1.8(软冲突 1000→1800、兼容细节 800→1440、变更意图 600→1080、
无关提及 600→1080),块内维持 冲突:非冲突 = 1:2,则全局仍是 1:2。

> 勘误:本节早前写的「软冲突块 354/3000 ≈ 12%」是错的——354 只是「带明确变更动词」这一族
> 措辞的正则命中数,合成块里标为真冲突的实际有 1000 条。比例结论不变,数字已按上段更正。

### 3. v4 验收表要补的校准数字(此前只在主 val 上算过)

`kaggle_eval/calib_report.py`(stdlib 即可跑,`python calib_report.py [data_local 目录]`,只读现成 JSON):

| 集合 | v2 baseline ECE | v3 ECE |
|---|---|---|
| val_soft 全体 | 0.2140 | 0.0625 |
| ├ kind=true(软冲突) | 0.1628 | 0.0684 |
| ├ kind=compat | 0.0605 | 0.0780 |
| ├ kind=consid(在考虑/意图) | 0.4205 | 0.0203 |
| ├ kind=unrelated | 0.3902 | 0.0791 |
| └ holdout 4 类(email/desk_floor/degree/barber) | 0.2864 | 0.0310 |
| 分语言 zh / en | 0.2228 / 0.2079 | 0.0569 / 0.0782 |
| realtest 20 条 | — | 0.0309 |

两条读数:①v3 在分布外集合上 ECE 0.0625,比主 val 的 0.0146 差 4 倍——**集成时别拿 0.0146 定阈值**;
②zh acc 97.7% / en 100%(v2 为 67.4% / 72.2%),中英均无塌陷。分语言报数是 M-Prometheus 的教训
(英文中心 judge 在非英文上退化),v4 起固定进验收表。

### 4. 集成红线(与 v4 无关,接插件时立即生效)

软冲突正例的 `p_true` 全部挤在 0.92 附近:100 条内 max 0.925 / 中位 0.922,**无一条 > 0.95**;
偏置诊断 A1 同为 0.9217。即 0.92 是本版头的软冲突置信天花板 ⇒ **接 dsh-mneme 时自动写入门槛
不得 > 0.92**(否则软冲突永不自动入库)。建议双阈值:≥0.90 自动 / 0.60-0.90 人工复核。

### 5. 发布/上线习惯(来自 Ollaya,可选)

- checkpoint 随附 sha256(现在只有目录名),权重固定到 commit
- 把「冲突?/是否更新旧记忆?/置信度」打包成**一次前向的 preset**,而非散问 N 次(省 N 倍延迟,顺带吃满 prefix cache)
- instruction 前缀措辞固定;改了措辞缓存全废

### 6. 明确不抄(省得返工)

- kev 的基座路线(LoRA-on-Qwen3.5 + flash-linear-attention):CUDA/ROCm 绑死、训练长度仅 384,换基座只换来新基建坑;我们已在 laya 上跑通全链路
- von 的 8k 长窗:输入是短记忆对,不需要
- Prometheus 那类 7B+ judge 做写入决策:成本/延迟不适合放进记忆写入路径(它能教的是数据配方,不是架构)
- DiffusionGemma 的约束解码:我们本来就是 typed head,绕回去了

### 7. 与既有 v4 方向的关系

「v3 训练结果」里留的「意图违反硬约束(过敏/安全)对照行」仍是主项,与本节的 A/B 及放大动作并行:
硬约束对照行解决**漏报**(数据覆盖),标签面增广/交换解决**表面依赖**(数据形态),两者不互相替代。
