# 关卡 0：读懂 apply_action

这是整个项目的心脏。`evalkit/simulator.py` 第 225 行到第 343 行，一百来行。
读完之后，这个项目在你眼里就不是一团了。

预计时间 30 分钟。不用写代码，只用读和填表。

---

## 怎么读一个函数

别一行一行硬啃。按这三步：

1. **看签名** —— 进来的是什么，出去的是什么
2. **看有几个 if 分支** —— 每个分支是一类情况
3. **看 return 出现在哪** —— 什么时候成功，什么时候失败

`apply_action` 的签名：

```python
def apply_action(state, action, args) -> tuple[bool, FailReason, str]:
```

进来三个东西：当前世界状态、动作名、参数列表。
出去三个东西：**成不成功**、**失败原因**、**一句人话说明**。

这个"出去三个东西"的设计很关键。只返回 True/False 的话，
你只知道错了，不知道错在哪。多返回一个原因，才能做 bad case 归因。

---

## 骨架：七个分支，每个分支做同一件事

```python
act = action.strip()
a0 = args[0] if len(args) > 0 else ""    # 第一个参数，通常是物体
a1 = args[1] if len(args) > 1 else ""    # 第二个参数，通常是容器

if act == "GotoLocation":   ...      # 移动
if act == "PickupObject":   ...      # 拿起
if act == "PutObject":      ...      # 放下
if act in TOOL_RECEPTACLES: ...      # 清洁 / 加热 / 冷却，三个共用一段
if act == "SliceObject":    ...      # 切片
if act == "ToggleObject":   ...      # 开关
if act in ("NoOp", "None", "End"): ...  # 空操作
return False, UNKNOWN_ACTION, ...    # 走到这说明动作名没认出来
```

**每个分支都在做同一件事，就两步：**

1. **检查前置条件** —— 物理上做得到吗
2. **做得到就改状态，做不到就返回失败原因**

明白这一点，七个分支就是同一段代码写了七遍。

---

## 我带你读 PickupObject（第 257 行）

```python
if act == "PickupObject":
    obj = state.find(a0)                                    # ①
    if obj is None:                                         # ②
        return False, FailReason.OBJ_NOT_FOUND, f"场景里找不到 {a0}"
    if state.inventory is not None:                         # ③
        return False, FailReason.HANDS_FULL, f"手上已有 {state.inventory}，无法再拿 {a0}"
    state.inventory = obj.obj_id                            # ④
    obj.parent = None                                       # ⑤
    return True, FailReason.OK, f"拿起 {obj.obj_type}"       # ⑥
```

**① `obj = state.find(a0)`**
`a0` 是动作的第一个参数。动作写 `PickupObject(cloth)`，`a0` 就是 `"cloth"`。
`find()` 拿着这个名字去场景物体里找（第 149 行），先比完整名再比类型名，忽略大小写。
找不到返回 `None`。

**② 找不到 → 失败**
返回三件套 `(False, 失败原因, 说明)`。`False` 会被 `run_plan` 记一次"前置条件违反"。

**③ 手上有东西 → 失败**
`inventory` 是"手上拿着什么"，`None` 表示空手。不是 `None` 说明已经有东西了。
这是 ALFRED 的硬规则：一次只能拿一个。

**④ 记到手上**
`obj_id` 是完整名，如 `Cloth_2476119a`。

**⑤ `obj.parent = None`**
`parent` 表示"这个物体在哪个容器里"。设成 `None` 表示它现在不在任何容器里——
它在手上。这一行必须写，不然物体会同时"在柜子里"和"在手上"。

**⑥ 成功**
返回 `True`。

**一句话总结这个分支：**
先确认东西存在、手是空的，然后把它从原来的位置摘下来，记到手上。

---

## 现在你自己读剩下六个

打开 `evalkit/simulator.py`，从第 248 行读到第 343 行，把这张表填了：

| 动作 | 检查什么（前置条件） | 成功时改哪些字段 |
| --- | --- | --- |
| GotoLocation | | |
| PickupObject | 物体存在、手是空的（已带你读完） | `inventory`、`obj.parent` |
| PutObject | | |
| CleanObject / HeatObject / CoolObject | | |
| SliceObject | | |
| ToggleObject | | |
| NoOp | 什么都不检查 | 什么都不改 |

填的时候注意两个地方，容易看漏：

- `PutObject` 那段有几行被注释掉了（第 274-275 行），那是关卡 1 动过的地方，想想为什么注释掉
- `Clean/Heat/Cool` 那段有个 `TOOL_RECEPTACLES` 字典（第 78 行），要翻上去看

---

## 答案

| 动作 | 检查什么 | 成功时改什么 |
| --- | --- | --- |
| GotoLocation | 参数不能为空 | `state.location` |
| PickupObject | 物体存在、手是空的 | `state.inventory`、`obj.parent` |
| PutObject | 物体存在、容器存在 | `obj.parent`、`state.inventory` 清空 |
| Clean/Heat/Cool | 物体存在、物体在手上、**站在对应器具前** | `obj.cleaned` / `heated` / `cooled` |
| SliceObject | 物体存在、**手上拿的是刀** | `obj.sliced` |
| ToggleObject | 物体存在、**物体类型可开关** | `obj.toggled` |
| NoOp | 无 | 无 |

加粗的两条最容易漏，也是模型最常犯的错：
想起来加热却忘了先走到微波炉前，想切片却没拿刀。

---

## 怎么验证自己读对了

挑一条样本，看它的每个动作分别触发了哪个分支：

```bash
python scripts/show_steps.py trial_T20190908_233657_231915
```

输出里每一步都会打印哪个字段从什么变成了什么。
那个"从什么变成什么"，就是你在表里填的"成功时改哪些字段"。

再换一条看看别的分支：

```bash
python scripts/show_steps.py --failed 3      # 带清洁动作的
python scripts/show_steps.py --failed 4      # 带开关动作的
```

`--failed` 不带数字会列出全部 4 条 gold 跑不通的 id。

---

## 读完这个你能讲什么

> 状态机的核心是一个 `apply_action` 函数，进来一个动作，
> 先检查物理前置条件，做到了才改状态，做不到就返回具体原因。
> 七个动作分支结构完全一样。
> 关键是返回值带了一个失败原因枚举——
> 统计这些原因的分布，就能知道模型到底是「不理解物理约束」
> 还是「场景感知错了」，两种问题的改进方向完全不同。

---

## 下一关

关卡 1 你做过了（注释掉两行看数字怎么变）。
关卡 2 是修 `CleanObject` 空参数那个真 bug，等你填完这张表再动手。
