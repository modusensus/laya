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
| v1 checkpoint（1.3GB，含 metrics.json） | `C:\Users\石晴\Desktop\laya-kaggle-output\laya-nli-conflict-v1\` |
| 训练数据（6k train + 1k val JSONL） | `C:\Users\石晴\Desktop\laya\data_local\nli_conflict_{train,val}.jsonl` |
| 训练脚本（DDP，官方 RLCD 配方） | `C:\Users\石晴\Desktop\laya\kaggle_eval\train_nli_conflict.py` |
| Kaggle kernel 主文件（含内嵌训练脚本 b64） | `C:\Users\石晴\Desktop\laya\kaggle_eval\nli_kernel.py` |
| kernel metadata | `C:\Users\石晴\Desktop\laya\kaggle_eval\nli_kernel_metadata.json` |
| Kaggle 数据集（已上传） | `daphnelaurent/nli-conflict-pairs` |
| Kaggle kernel（v3 已跑通全流程） | `daphnelaurent/laya-nli-memory-conflict-fine-tune` |
| 真实 case 实测结果 | `C:\Users\石晴\Desktop\laya\data_local\memory_conflict_realtest.json` |
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

- v2 checkpoint（Kaggle output 拉回，放 `C:\Users\石晴\Desktop\laya-kaggle-output\laya-nli-conflict-v2\`）
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
agent = laya.load(r'C:\Users\石晴\Desktop\laya-kaggle-output\laya-nli-conflict-v2', device='cpu')
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
- kaggle CLI 在含中文的用户名路径下会挂：临时目录必须指到 ASCII 路径（v3 用的是 `C:/kaggle_cfg/tmp`），token 从 `C:\Users\石晴\.kaggle\kaggle.json` 读后以 `KAGGLE_API_TOKEN` env 传入。
- HF/Kaggle token 一律 [REDACTED]，不要写进任何文件或日志；用完提醒用户吊销。
- 用户偏好：中文交流；诚实汇报失败；重要决策先给数字和门槛再动手。

## 明确不要做

- 不要动 `laya/` 核心库（本次任务范围只有数据 + kernel + 实测）。
- 不要尝试把任务改成多分类或换模型架构——上游 #156 的结论是标签词问题，先按已验证方案走。
- 不要在没跑偏置诊断的情况下直接报"修好了"。
- 实测 case 必须原样用上面 20 条（可加不可减），否则和 v1 的 9/20 没有可比性。
