# Prompt 清单与设计说明

这个目录把所有 prompt 集中放着，改 prompt 只需要改这里的 `.md` 文件，不用碰 Python 代码。

---

## 一、一共 8 个 prompt，分四组

| 文件 | 用在哪 | 给谁 | 要它做什么 |
| --- | --- | --- | --- |
| `plan_system.md` + `plan_user.md` | `scripts/run_eval.py` | **被测模型** | 生成动作序列 |
| `judge_system.md` + `judge_user.md` | `scripts/run_judge_audit.py` | **裁判模型** | 判断这条序列能不能达成目标 |
| `pairwise_system.md` + `pairwise_user.md` | `scripts/run_judge_audit.py` | **裁判模型** | 两条序列哪条更好（位置偏差实验用） |
| `rubric_system.md` + `rubric_user.md` | `scripts/run_rubric_check.py` | **裁判模型** | 按四个维度给序列打分 |

**关键区别**：给被测模型的和给裁判的，内容不一样。裁判会额外拿到「被测模型的输出」和「目标条件」，而被测模型拿不到这些。

---

## 二、设计要点（面试官常问）

### 1. 为什么 prompt 里要写物理约束

`plan_system.md` 里那段「硬性约束」（一次只能拿一个、加热要站微波炉前）**不能省**。

理由：不告诉模型这些规则，它违反规则是我们没说清楚，不是它规划能力差。**把规则写进 prompt，才能保证测的是能力而不是信息缺失。**

这条有实测支撑：前置条件违反率 21.9%，这些是模型在知道规则之后仍然犯的错，才是真实的能力问题。

### 2. 为什么裁判的 prompt 里要强调「不要重新规划」

`judge_system.md` 里有一句「不要重新规划，只判断给出的这条序列行不行」。

理由：我们要的是裁判的**判定**能力。如果让它自己另写一条，它可能会写一条更好的然后判「能达成」，那就不是在评价被测模型的输出了。

### 3. 为什么 rubric 要求「逐维度独立打分，不要合成总分」

`rubric_system.md` 里明确要求了。

理由：总分无法定位问题。分维度之后才能说清楚这个模型是「爱绕路」还是「爱漏步骤」——这两种问题的解法完全不同。

### 4. temperature 的设定

- **评测和审计用 temperature=0**（确定性输出，保证可复现）
- **pass^k 实验必须用 temperature>0**（默认 0.7）。因为 temperature=0 时每次输出相同，跑 3 次得到 3 个一样的结果，根本测不出稳定性

`scripts/run_passk.py` 里有这个检查，填 0 会直接报错退出。

---

## 三、改 prompt 的注意事项

1. **改完必须升版本号**。`evalkit/config.py` 里的 `PROMPT_VERSION`（当前 `embodied-plan-v1`）参与缓存键计算。不升版会命中旧缓存，你以为改了生效，其实跑的还是旧结果——这个坑很隐蔽。

2. **改完重跑 gold 验证**。如果改动影响输出格式，先跑 `python scripts/verify_gold.py` 确认状态机仍能解析。

3. **一次只改一处**。同时改多个 prompt 会分不清是哪个改动带来的效果差异。

---

## 四、代码怎么读这些文件

`evalkit/config.py` 里的 `load_prompt()` 函数：

```python
from evalkit.config import load_prompt
system = load_prompt("plan_system")   # 读 prompts/plan_system.md
```

文件不存在时会报错并列出可用清单，不会静默返回空字符串——那是另一种隐蔽的坑。
