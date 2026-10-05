# 团队 Onboarding — Laya NLI 记忆冲突检测头

> 给新加入的 AI 协作者（以及未来的我们自己）：这一页是进入项目的入口。
> 权威数字一律以 §2「信息地图」列出的真值源为准，本文快照只做定位。
>
> 最后更新：2026-10-04（v11 三跑协议判定 FAIL 落盘后）。
> **状态变更时更新本文档对应小节，不要只写各 agent 的私有记忆**——私有记忆不共享。

## 0. 一分钟概览

上游 **Laya**（NandhaKishorM/laya）是一个 0.3B 本地决策引擎，接口为 `/v1/systemone` 风格的 noul/choice/score 三类决策。本 fork 的项目：为记忆插件场景微调一个 **NLI 记忆冲突检测头（noul 头）**——判断新信息与用户既有记忆冲突/兼容/无关。

工作方式：数据与评测工具在本仓 `kaggle_eval/`，语料推 Kaggle 数据集，训练跑 Kaggle 免费 GPU kernel，产物回本机评测，按**预注册验收协议**逐轴判定；达标才交付（推 Hugging Face），否则负结论照交、现役版本不动。

**当前状态一句话**：v4 已交付并公开在 HF（现役）；v5–v11 共 7 轮迭代未达交付门槛；最新 v11 走完三跑预注册协议，13 轴 10 PASS / 3 FAIL 判 FAIL——协议在正确工作，不是事故。

## 1. 团队与分工

| 成员 | 角色 | 备注 |
|---|---|---|
| 用户 | 唯一决策人 | 交付、HF 推送、协议变更、算力花费都由用户拍板；中文交流 |
| ZCode | 执行主力 | 数据制备、评测、编排器、文档；维护私有记忆 + 本文档 |
| Hermes | 验收代理 + 任务书作者 | 逐数字复算交付物；也会自己写任务书/脚本/直接 commit——接手前先 `git log`，别重写它已做的东西 |
| dsh | 新加入（2026-10） | 职责由用户分配；从本文档和 §5 启动清单进入 |

## 2. 信息地图（什么在哪、以谁为准）

| 内容 | 位置 | 效力 |
|---|---|---|
| 项目权威上下文 | `kaggle_eval/HANDOFF_NLI_V11.md`（当前轮）+ `HANDOFF_NLI_V2~V10.md`（历史） | **权威**。汇报数字不要靠记忆复述，回真值源取 |
| 验收判定 | `kaggle_eval/protocol_verdict.py`；v11 判定日志 `data_local/protocol_verdict_v11.log` | 复现命令见 §4 |
| 工具链 | `kaggle_eval/*.py`（gen_* 语料生成、eval_kernel、conformal_abstain、family_diag、leak_audit、band_report、hf_publish、hf_archive_index 等） | 脚本即文档，改动走 commit |
| 语料真值源 | `data_local/nli_conflict_train_v11.jsonl`（15914 行）；Kaggle 数据集 `daphnelaurent/nli-conflict-pairs` v16 | 携带链与卫生断言见 V11 交接书 |
| 本机 checkpoint | `D:\laya-kaggle-output\`（v4 现役 + v5–v11 各臂） | SHA256 见 `kaggle_eval/archive_sha256_manifest.txt` |
| 公开发布 | HF `Modusnsus/laya-nli-memory-conflict`（v4 现役）+ v5–v10 九个归档仓 + 合集 | 归档卡面数字取自本地 metrics 真值源 |
| 工作日志 | 本 fork 独有的提交历史（round-N task book → execution → record 节奏） | `git log` 就是流水账 |
| AI 私有记忆 | 各 agent 自管（如 ZCode 在 `~/.zcode/cli/memories/`） | **不共享**。重要结论必须落盘到 HANDOFF 或本文档 |

## 3. 环境速查（本机）

- 仓库：`D:\laya`（origin = GitHub fork `modusensus/laya`，SSH over 443；upstream = `NandhaKishorM/laya`）。**`C:\Users\石晴\Desktop\laya` 是 C 盘空壳，别用。**
- Python：一律 `D:\Miniconda\envs\laya-ft\python.exe`（torch CPU + transformers + datasets + laya 可编辑安装）。
- **C 盘空间敏感**：大文件一律放 D 盘；例外 `C:\kaggle_cfg\` 是 ASCII 路径暂存区，Kaggle 工具链依赖它。
- 凭据：Kaggle `C:\kaggle_cfg\kaggle.json`（经环境变量使用，值不落盘不进日志）；HF 已 `hf auth login`（classic write token，注意两处同步：`~/.cache/huggingface/token` 与 `D:\hf_cache\token`，HF_HOME 指向后者）。
- 账号：GitHub `modusensus`、HF `Modusnsus`、Kaggle `daphnelaurent`（均为公开身份）。
- 已知坑：Git Bash 自带 ssh 读不到 `~/.ssh/config`（中文用户名路径编码 bug），hf.co 的 git 操作要用 Windows OpenSSH（`C:\Windows\System32\OpenSSH\ssh.exe`）。

## 4. 现状快照（2026-10-04）

- **v4 现役**：HF `Modusnsus/laya-nli-memory-conflict`（公开，apache-2.0）。软冲突 p_true 置信天花板 ≈0.92——下游把头接进记忆插件时，自动写入阈值不得 >0.92。
- **v11 三跑协议判定 FAIL（终局，2026-10-04 落盘）**：13 轴 10 PASS / 3 FAIL。
  - FAIL：轴 1 主 val 三跑中位 0.895 < 0.896（差 1 例）；轴 3 new10（r2=9/10）；轴 4 negation（r2=4/5）——后两轴是「三跑全满分」口径下的薄边单例翻转。
  - PASS 亮点：老 20 三跑 60/60（杠杆 1b 根治）；**家族轴三跑全绿**（family mean p 0.2169 / 0.0939 / 0.0794，中位 0.0939，零高置信案）vs 旧单案 B2 读数摆幅 0.544↔0.107——家族率仪器设计被三跑实证；conformal 六档全过。
  - 负结论照交：v4 不动、HF 不动、不叠修补。复现：`python kaggle_eval/protocol_verdict.py --tags r1,r2,r3 --prefix v11_`（日志 `data_local/protocol_verdict_v11.log`）。
- 训练通道：Kaggle 免费 GPU，r1–r3 = kernel `laya-nli-conflict-ce` version 9/10/11（约 17 分钟/跑）；数据集 v16（15914 行）。
- fork 已同步上游 0.3.24–0.3.26（2026-10-04，merge commit b113c93）。

## 5. 新会话启动清单

1. 读本文档。
2. 读 `kaggle_eval/HANDOFF_NLI_V11.md`（当前轮上下文；需要历史与「三大坑」背景时再读 V2/V3）。
3. `git log --oneline -30` + `ls kaggle_eval/`——确认别的 agent 已经做了什么，别重写。
4. 需要报数字时回 §2 真值源取，并保证本机可复现。
5. 不确定的事问用户；用户没拍板的事不动手。

## 6. 踩过的坑（前人血泪，按层分类）

### 6.1 方法论层

- **单案探针无判定力**。B2 单案三跑 0.889 / 0.3877 / 0.6951，极差 50pp 且横跨 0.5 线——任何单案读数都不构成结论。解法 = 家族轴（9 案平均）+ 中位口径（v11 已实证）。
- **薄边轴「三跑全满分」在 n=3 下是绑定约束**：1 例翻转即判死（v11 轴 3/4 就死在这）。v12 议题：薄边家族扩容增稳，或协议中位化。
- **门槛设计避免绝对数**：acc=1 时 ECE≡1−置信上限，不含模型质量信息——改条件式门槛（如只在 acc<1 的切片评 ECE）。
- **数据泄漏形态清单**：unrelated 控制行缺失、句面复用、holdout 混入。每轮语料必须过 `leak_audit.py`（35 案 0 命中断言）+ 家族/诊断字段零交检查。
- **RL 项与分级软目标相克**（v5 臂 A 崩、纯 CE 臂 B 反超）→ 主臂用纯 CE，别再叠 RL。
- **置信压缩带是历轮死局总根因**：软冲突 max_conf 全体 0.918–0.928、无区分力。后处理救不了，主攻 = 让置信可分（分级 soft CE 已把 band 0.28→16.2pp）。
- **「run 方差」归因要靠重复 miss 检验**：同配置 2/3 跑重复 miss 同一案 = 语料真实弱点，不是方差（protocol_verdict 抓出过两案）。
- **下「不可满足」结论前先枚举推理期配置层杠杆**（温度、阈值、校准、decode 路径），配置层有解就别下架构级结论。
- runtime 硬事实：laya 温度粒度 = 题型×选项数桶、decode 无 bias 项——任何 per-class (τ,b) 校准都要动 `_decode_answers`，不存在纯配置层做法。

### 6.2 工程层

- **Kaggle CLI**：`TMPDIR/TMP/TEMP` 都要指到 ASCII 路径（`C:\kaggle_cfg\tmp`），否则上传在非 ASCII 临时路径上失败；`kernels_push` 收文件夹（内含 kernel-metadata.json）；**CLI 只能拉最新版输出 → 推序纪律：拉完上一版输出才推下一版**（违者只能靠 session output API 按版本抢救）；api.kaggle.com 偶发 SSL EOF/IncompleteRead，重试即可，别改配置；GPU 会话上限 2，并发推 kernel 会撞 cap。
- **拉取失败分类**：网络类失败留在原相位跨 tick 重试，只有完整性断言（权重字节数、指纹）才进 ERROR——网络抖动不该冻结流水线等人工。
- **eval 电池 rc = 工件存在性**：gate FAIL 的非零退出码仅 informational，裁决权在 protocol_verdict。别把 rc=1 当停机信号。
- **脚本化 str.replace 补丁必须 grep 验证替换落地**——定义处替换静默 no-op、调用点全改，会出 TypeError（r2 电池崩溃教训）。
- **HF**：fine-grained token 对合集 API 一律 403，必须手动建 classic "Write" token；`HF_HOME=D:\hf_cache`（C 盘保护）；上传走 xet 分块，约 1.5MB/s。
- **后台 shell 被中断会卡死子进程**（挂在断管道上）——taskkill 后分段重跑。
- 本机 pandas 被 Windows 应用控制策略挡，parquet 走 pyarrow。

### 6.3 调研引用纪律（两次实锤教训）

- 子代理/搜索工具报的论文、模型卡数字是**二手转述，幻觉高发区**（polaris 案：卡上根本不存在的「精彩对比数字」被写进任务书，Hermes 逐条核对抓出）。
- 工具返回被截断时，细节一律视为未获取，**不许用先验印象补全**（kev 案：凭生态印象脑补出整套不存在的损失函数/基准）。
- 规则：驱动决策或写入正式文档的每条外部数字，先回原始来源（arXiv 摘要页、HF raw README）抽查；无法核实的标「未核实」或删除。

## 7. 协作纪律与偏好

- **中文交流**；重要决策先给数字和门槛再动手。
- **诚实汇报失败**：未过门槛就是未过，不凑数。改门槛 = gate-shopping，禁止；门槛变更走协议预注册流程（草案 → 定稿 → 回测 → 开跑）。
- **汇报数字必须本机可复现**，附产物路径和 SHA256。
- **交付与 HF 推送 = 用户人工确认门**，任何 agent 不得自作主张（v11 verdict PASS 后也仍需用户点头）。
- 接手新任务先看 git log 和 `kaggle_eval/` 现状；**Hermes 定稿的任务书优先于我方原稿**（V4.1 替代 V4 的先例）；它交付的脚本优先复用（temperature_sweep 先例）。
- 对 `laya/` 核心代码的修改遵循上游 `AGENTS.md`（英文 conventional commit、CI gates、不整文件重排版）。fork 内 `kaggle_eval/` 的提交同样用 conventional 风格：`feat/fix/docs(kaggle-eval): ...`。
- **文档同步纪律**：状态变更要更新 HANDOFF 对应轮次 + 本文 §4 快照；只写进私有记忆 = 下一个 agent 看不见。

## 8. 当前计划与待办（2026-10-04）

1. **v12（已立项，任务书定稿 2026-10-06）**：`HANDOFF_NLI_V12.md` = 否定冲突杠杆 36 行 + 新 10/否定 5 聚合口径中位化 + 否定家族 8 案新仪器（轴 #13，卫生断言扩 0/51）。两项开放决策已由用户按推荐拍板（§4）。待执行（ZCode）：gen 语料 v17 → verdict 工具升级 → v11 前置回测 → 三跑（kernel version 12/13/14）。
2. **待用户拍板**：v1/v2/v2-e3/v3 早期 ckpt 与 typed_decisions 目录是否补传 HF 归档。
3. **已清理（2026-10-06）**：定时任务 automation-f6e9077a 已删除。
4. **可选**：上游 tag v0.3.24/25/26 已 fetch 到本地，是否推 fork 待定。
5. **上游新能力评估（0.3.24–0.3.26）**：`LAYA_JEV_STRICT`（serve 端严格 Jev wire contract）、selective-classification 评测指标（Brier/AURC）、histogram-binning 置信重校准、`tests/test_conformal_abstention.py`——与 v12 方向的相关性待评估。
