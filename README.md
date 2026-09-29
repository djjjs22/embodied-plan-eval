# embodied-plan-eval

具身任务规划评测

符号状态机执行 + 客观真值判定 · 无需仿真器 / GPU / 机器人

具身任务规划评测。用符号状态机执行模型生成的动作序列，以最终状态是否达成目标作为客观真值。**不需要仿真器、不需要 GPU、不需要机器人。**

---

## 数据许可

`dataset/` 下的评测样本派生自 [ALFRED](https://askforalfred.com/)（MIT License），由 `scripts/import_alfred.py` 从 ALFRED 的 `json_2.1.0` 轨迹中抽取，仅保留高层符号动作与目标条件，不含任何图像。原始版权归艾伦人工智能研究院（AI2）。

本仓库代码与评测集同以 MIT 许可发布。

---

## 这个项目要回答什么

具身系统分两层。高层规划把「把加热过的面包放到桌上」拆成动作序列，是纯符号推理；底层执行是机械臂真的去抓，需要仿真和硬件。

第一层的质量同样决定系统成败，但它缺少可规模化的评测手段。常见做法是让一个模型给另一个模型的输出打分，而**这把尺子本身没人验证过**。

本项目换个思路：ALFRED 的高层动作是符号化的，那就可以写一个状态机去**真的执行**模型生成的动作序列，看最终状态有没有达成目标。执行结果不会因人而异，也不需要任何模型来发表意见。

这跟「SQL 项目里用执行结果比对」是同一个思想，只是换了个任务域。

---

## 为什么不用仿真

跑 Habitat 或 AI2-THOR 要装 Docker、配 GPU、下几十 GB 场景资源，光环境就可能耗掉几天。而高层规划的输入输出都是文本，用 LLM API 就能测。

代价是：**只测规划，不测执行**。这个边界在报告和简历里必须写清楚，不能把规划成功率说成机器人成功率。

---

## 快速开始

```bash
pip install -e ".[dev]"
export MINIMAX_API_KEY=...        # 或写进项目根 .env，已在 .gitignore 里

# 1. 从 ALFRED 抽取 300 条分层样本
python scripts/import_alfred.py --root ../data/alfred --per-type 50

# 2. 硬门槛：用 gold 序列验证状态机，必须接近 100%
#    过不了就回去修 simulator.py 的前置条件规则，别往下走
python scripts/verify_gold.py

# 3. 先跑 10 条打通链路
python scripts/run_eval.py --limit 10 --workers 2

# 4. 全量
python scripts/run_eval.py
```

ALFRED 数据（Lite 版 35.7MB，只含轨迹 JSON）：

```
https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z
```

解压需要 7z 工具。macOS 上建议装系统版 `p7zip`，**不要用 py7zr 全量解压**——它处理近两万个小文件极慢，实测十几分钟跑不完还会把磁盘 IO 占满。

---

## 三层打分

| 层 | 判什么 | 怎么判 |
| --- | --- | --- |
| L0 可执行性 | 动作序列能不能跑 | 符号状态机，前置条件逐条校验 |
| L1 目标达成 | 最终状态对不对 | 按 `pddl_params` 检查目标条件 |
| L2 路径质量 | 绕不绕、冗余不冗余 | Rubric 评分，见 `rubric/plan_quality_v1.md` |

L0 和 L1 由代码判定，L2 才交给 Judge。能用代码判的绝不交给模型。

完整指标定义见 `docs/METRICS.md`，含每个指标的已知局限。

---

## 核心指标

除了常规的成功率和编辑距离，这个项目有两个值得单独说的指标：

**前置条件违反率**。量化模型生成了物理上执行不了的动作，比如没拿到东西就要放下、切片时手上没刀。这是具身规划独有的幻觉类型，文本生成任务里没有对应物。

**pass^k**。同一条样本连续跑 k 次都成功才算通过。τ-bench 用它发现前沿 Agent 单次看着还行，连续八次掉到 25% 以下。pass@1 与 pass^k 的差值就是不稳定性。

---

## 难度分层

直接用 ALFRED 官方的 7 种任务类型，它们天然就是由易到难的阶梯，不需要另外设计：

| 类型 | 难度 |
| --- | --- |
| pick_and_place_simple / look_at_obj_in_light | L1 |
| pick_clean_then_place / pick_heat_then_place / pick_cool_then_place | L2 |
| pick_two_obj_and_place / pick_and_place_with_movable_recep | L3 |

---

## 目录

```
evalkit/
  simulator.py   符号状态机，整个项目的地基
  alfred.py      ALFRED 数据解析
  models.py      LLM 客户端与 prompt 模板
  metrics.py     Wilson 区间、Cohen's Kappa、编辑距离
  report.py      HTML 报告
scripts/
  import_alfred.py   抽取分层样本
  verify_gold.py     Day 2 硬门槛，gold 必须跑通
  run_eval.py        跑评测出报告
rubric/              评分量表，含锚点样例
docs/METRICS.md      指标卡
```

---

## 已知边界

- **只测高层规划，不代表机器人真能执行成功**。抓取失败、视觉误检都不在评测范围内
- 状态机规则是对 ALFRED 语义的保守简化，若 gold 跑不通以数据为准修规则
- 路径效率依赖 gold 长度作参照，若 gold 不是最短路径该维度会失真
- 高层评测无法捕捉需要视觉反馈的失败（比如物体被遮挡）
