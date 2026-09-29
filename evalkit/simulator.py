"""符号执行器 —— 整个项目的地基。

================================================================================
这个文件解决什么问题
================================================================================

评测的核心难题是「怎么判定模型输出对不对」。业界默认做法是让另一个模型来
打分（LLM-as-a-Judge），但那把尺子本身没被验证过：评判模型有位置偏差、
长度偏差、自偏好偏差，换一个评判模型分数就变。

本项目换个思路：找一个**能执行**的东西当尺子。

具身任务的高层动作是符号化的 —— PickupObject、PutObject 这类名字，参数是
物体名。既然是符号，就可以用代码推演它的效果，不需要 3D 仿真器也不需要
机器人。这个文件做的就是这件事：给一串动作，推演它执行完之后世界变成
什么样，再和目标条件比对。

对应方案里的位置：这是「客观真值」的来源，是整个项目唯一不可替换的部分。

================================================================================
怎么验证这个文件是对的
================================================================================

用 gold 序列反向验证：ALFRED 每条任务都有专家标注的正确动作序列，把它丢进
状态机跑，能跑通且达成目标，说明规则写对了。当前 315 条里 311 条通过
（98.7%，2026-09-28 复跑确认）。跑通率低于 95% 就说明规则有问题，必须先修这里。

================================================================================
和 SQL 项目的对应关系（有助于理解）
================================================================================

SQL 项目里，客观真值来自「把 SQL 丢进数据库执行，比对结果集」。
本项目里，客观真值来自「把动作序列丢进状态机执行，比对最终状态」。
两件事的结构完全一样：找一个可执行的判定器，替代主观打分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# 模型输出时爱在每行前面加序号，比如：
#     1. PickupObject(Apple_1)
#     2、GotoLocation(Fridge_1)
# 不剥掉的话动作名会变成 "1. PickupObject"，后面全部匹配失败。
# 注意这个正则必须在解析动作名之前应用，顺序反了就失效。
_LEADING_ENUM = re.compile(r"^\s*(?:\d+[\.、\)]|[-*•])\s*")


class FailReason(str, Enum):
    """动作执行失败的原因分类。

    这个枚举不是给程序看的，是给归因分析用的。它直接对应方案里 bad case
    归因的第一层：模型到底错在哪一类。

    比如统计出来 NOT_HOLDING 占多数，说明模型的问题是「没物理常识」；
    如果 OBJ_NOT_FOUND 占多数，说明问题在「场景理解」。
    两种问题的改进方向完全不同，所以必须分开记。
    """

    OK = "ok"
    UNKNOWN_ACTION = "未知动作"
    UNPARSEABLE = "无法解析"
    OBJ_NOT_FOUND = "物体不在场景中"
    RECEP_NOT_FOUND = "容器不在场景中"
    HANDS_FULL = "手上已有物体"
    NOT_HOLDING = "手上不是该物体"
    HANDS_EMPTY = "手上是空的"
    WRONG_LOCATION = "位置不对"
    NEED_KNIFE = "需要持刀才能切片"
    NOT_TOGGLEABLE = "该物体不可开关"


# 清洁、加热、冷却这三个动作不是随便哪都能做的，
# 必须站在对应的器具前面。这是 ALFRED 的物理约束，
# 也是模型最容易漏掉的一步（拿了东西直接加热，忘了先走到微波炉前）。
TOOL_RECEPTACLES = {
    "CleanObject": {"sink", "sinkbasin", "bathroom sink"},
    "HeatObject": {"microwave"},
    "CoolObject": {"fridge"},
}

# 只有这些类型的物体能被开关（开灯、开水龙头）。
# 对桌子或苹果做 ToggleObject 没有意义，要拒绝。
TOGGLEABLE = {"desklamp", "floorlamp", "lamp", "faucet", "tv", "television", "lightswitch"}


def normalize(name: str | None) -> str:
    """把物体名归一化成统一的比较键。

    为什么要归一化：同一个物体在不同地方写法不一样。
        scene 里叫  "AlarmClock_1a2b3c"  （类型名_哈希）
        类型名是    "AlarmClock"
        gold 参数写 "alarmclock"        （纯类型名小写）
    不统一的话，gold 里的 "alarmclock" 永远匹配不上场景里的物体，
    结果就是全部样本都被判成「物体不在场景中」——我第一版就踩了这个坑，
    当时 140 条样本 0% 通过，查了半天才发现是名字对不上。

    做法：转小写，再去掉下划线、连字符、空格、竖线这些分隔符。
        "AlarmClock_1a2b3c" -> "alarmclock1a2b3c"
        "Desk Lamp"         -> "desklamp"
    """
    # ① 空值直接返回空字符串。后面 find("") 就靠这个判「压根没给名字」
    if not name:
        return ""
    # ② 转小写 + 去首尾空格。"AlarmClock" 和 " alarmclock " 必须算同一个东西
    out = name.lower().strip()
    # ③ 挨个删掉分隔符。竖线是 ALFRED 里偶尔出现的写法，一并处理掉
    for ch in ("_", "-", " ", "|"):
        out = out.replace(ch, "")
    return out


@dataclass
class ObjectState:
    """场景里一个物体的状态。

    parent 表示它当前在哪个容器里，None 表示正被拿在手上。
    其余几个布尔字段是物体的物理状态，会被 Clean / Heat / Cool /
    Slice / Toggle 这几个动作改变，也是目标条件可能检查的内容。
    """

    obj_id: str                 # 完整名，如 "AlarmClock_1a2b3c"
    obj_type: str               # 类型名，如 "AlarmClock"
    parent: str | None = None
    cleaned: bool = False
    heated: bool = False
    cooled: bool = False
    sliced: bool = False
    toggled: bool = False

    @property
    def key(self) -> str:
        return normalize(self.obj_id)


@dataclass
class WorldState:
    """世界状态。

    三个字段就是具身规划里全部需要跟踪的东西：
        location   机器人在哪（决定了能不能清洗、加热、冷却）
        inventory  手上拿着什么（ALFRED 规则：最多一个）
        objects    场景里所有物体及其状态
    """

    location: str | None = None
    inventory: str | None = None
    objects: dict[str, ObjectState] = field(default_factory=dict)

    def find(self, name: str) -> ObjectState | None:
        """按名字找物体。

        先按完整 id 匹配，匹配不上再退化成按类型名匹配。
        两级匹配是因为 gold 动作的参数用的是类型名（"alarmclock"），
        而场景里登记的是完整 id（"AlarmClock_1a2b3c"）。
        """
        # ① 空名字直接返回 None，不进循环
        if not name:
            return None
        # ② 先把要找的名字归一化，和场景里存的 key 用同一套规则
        key = normalize(name)
        # ③ 两级匹配：先比完整 id（"alarmclock1a2b3c"），
        #    比不上再退化成比类型名（"alarmclock"）。
        #    必须有两级 —— gold 参数写的是类型名，场景里存的却是完整 id
        for obj in self.objects.values():
            if obj.key == key or normalize(obj.obj_type) == key:
                return obj
        # ④ 两级都没中 → 场景里真没这个东西
        return None

    def clone(self) -> WorldState:
        """深拷贝一份状态。用于需要保留执行前快照的场景。

        注意 objects 要逐个重建，只复制字典的话里头的 ObjectState
        还是同一批对象，改一个另一个跟着变。
        """
        # ① 位置、手持这两个是标量，直接抄
        # ② objects 必须逐个 ObjectState(**vars(v)) 重建，不然是浅拷贝
        return WorldState(
            location=self.location,
            inventory=self.inventory,
            objects={k: ObjectState(**vars(v)) for k, v in self.objects.items()},
        )


@dataclass
class StepResult:
    """单个动作的执行结果。raw 保留原始文本，出错时用来定位是哪一行。"""
    index: int
    raw: str
    action: str | None
    args: list[str]
    ok: bool
    reason: FailReason = FailReason.OK
    detail: str = ""


@dataclass
class RunResult:
    """整条动作序列的执行结果。

    violations 是本项目最有价值的数字，见 run_plan 的说明。
    """
    steps: list[StepResult]
    final_state: WorldState
    goal_achieved: bool
    violations: int              # 前置条件违反次数
    executed: int                # 成功执行的动作数
    goal_detail: dict = field(default_factory=dict)


def parse_action_line(line: str) -> tuple[str | None, list[str]]:
    """把一行动作文本拆成 (动作名, 参数列表)。

    能处理的输入形态：
        "PickupObject(Apple_1)"         -> ("PickupObject", ["Apple_1"])
        "PutObject(Apple_1, Fridge_1)"  -> ("PutObject", ["Apple_1", "Fridge_1"])
        "1. PickupObject(Apple_1)"      -> ("PickupObject", ["Apple_1"])
        "NoOp()"                        -> ("NoOp", [])
        ""                              -> (None, [])

    解析不出来返回 (None, [])，由调用方记为 UNPARSEABLE，
    而不是抛异常中断整条序列 —— 一条动作坏了不代表后面的都该放弃。
    """
    # ① 剥行首序号。这一步必须排在解析动作名之前，
    #    顺序反了会把 "1. PickupObject" 整个当成动作名，后面全部匹配失败
    s = _LEADING_ENUM.sub("", line.strip()).strip().strip(".,")
    # ② 空行（或剥完只剩一个标点）→ 返回 (None, [])，调用方记为 UNPARSEABLE
    if not s:
        return None, []
    # ③ 标准写法：有括号且以右括号结尾。从括号里抠出参数，按逗号切分
    if "(" in s and s.endswith(")"):
        head, _, rest = s.partition("(")
        args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
        return head.strip(), args
    # ④ 退化路径：模型没写括号，比如 "PickupObject Apple_1"。
    #    把逗号也当空格处理，再按空格切
    parts = s.replace(",", " ").split()
    return (parts[0], parts[1:]) if parts else (None, [])


def apply_action(state: WorldState, action: str, args: list[str]) -> tuple[bool, FailReason, str]:
    """执行单个动作，原地修改 state。返回 (是否成功, 失败原因, 说明)。

    ==========================================================================
    这是整个项目最关键的函数。每个动作分支都在做同一件事：
        1. 检查前置条件（物理上做得到吗）
        2. 条件满足才修改状态，不满足就返回具体失败原因
    ==========================================================================

    为什么前置条件这么重要：模型经常生成「看起来合理但物理上做不到」的
    动作，比如手里还拿着苹果就去拿面包。人类一眼能看出不对，但如果只是
    比对 gold 序列的字符串，这种错误只会被算成「和 gold 不一样」，
    说不清错在哪。状态机能明确指出「第 2 步手上已有物体」，
    这就是可解释性，也是 bad case 归因的基础。

    规则来源：ALFRED 的语义，做了保守简化。
    若 gold 跑不通，优先怀疑这里，而不是怀疑数据。

    七个分支结构完全一样，读懂一个剩下六个自然通。
    每个分支独立从 ① 开始编号。
    """
    # ① 动作名去首尾空格。后面全靠字符串比对，多一个空格就匹配不上
    act = action.strip()
    # ② 第一个参数，几乎总是「物体」。没给就当空字符串，find("") 会返回 None
    a0 = args[0] if len(args) > 0 else ""
    # ③ 第二个参数，几乎总是「容器」。只有一个参数的动作拿不到，同样是空
    a1 = args[1] if len(args) > 1 else ""

    # ---------------------------------------------------------------- 移动
    if act == "GotoLocation":
        # ① 唯一的前置条件：得给个位置。没给就不知道往哪走
        if not a0:
            return False, FailReason.UNPARSEABLE, "缺少位置参数"
        # ② 改机器人的位置。这一步看着没用，其实决定了后面能不能洗/热/冷
        state.location = a0
        return True, FailReason.OK, f"移动到 {a0}"

    # ---------------------------------------------------------------- 拿起
    if act == "PickupObject":
        # ① 拿着参数里的名字去场景找物体，找不到返回 None
        obj = state.find(a0)
        # ② 场景里没这东西 → 失败
        if obj is None:
            return False, FailReason.OBJ_NOT_FOUND, f"场景里找不到 {a0}"
        # ③ 手上已经有东西 → 失败。ALFRED 硬规则：一次只能拿一个
        if state.inventory is not None:
            return False, FailReason.HANDS_FULL, f"手上已有 {state.inventory}，无法再拿 {a0}"
        # ④ 【已知漏洞，暂未修复】这里本该校验「是否已走到物体所在位置前」。
        #
        #    不校验的后果：模型可以完全不写 GotoLocation 就直接拿起东西，
        #    路径凭空短两三，还能被判成功。路径多样性分析实测发现，
        #    94% 的「比 gold 更短」的成功样本就是这么来的。
        #
        #    2026-09-27 尝试加上校验，结果 gold 通过率从 98.7% 掉到 67.9%，
        #    门槛拦住了。原因是 ALFRED 的 GotoLocation 语义比预想复杂：
        #    它的目标可以是**物体本身**（GotoLocation(bread) 走到面包前），
        #    而不只是容器。按「必须站在物体所在容器前」来校验就过严了。
        #
        #    正确修法需要先把「物体 → 所在位置」的映射建对
        #    （ALFRED 的 object_poses 里没有 parentReceptacles 字段，
        #    当前物体的 parent 多数靠动作序列推断），再回过头来加校验。
        #    已记为待改进项，见 docs/DATA_CARD.md 的已知偏差一节。
        #
        # ⑤ 记到手上，存的是完整名（Cloth_2476119a）
        state.inventory = obj.obj_id
        # ⑥ 离开原来的容器。设 None 表示「不在任何容器里，在手上」。
        #    这行不写的话，物体会同时「在柜子里」又「在手上」
        obj.parent = None
        return True, FailReason.OK, f"拿起 {obj.obj_type}"

    # ---------------------------------------------------------------- 放下
    if act == "PutObject":
        # ① 找物体。注意 a0 是被放的东西，a1 是放到哪，别搞反
        obj = state.find(a0)
        # ② 场景里没这东西 → 失败
        if obj is None:
            return False, FailReason.OBJ_NOT_FOUND, f"场景里找不到 {a0}"
        # ③ 手上拿的不是它 → 失败。没拿到东西就说要放下，是最典型的规划幻觉。
        #    关卡 1 让你把这两行注释掉跑一遍，通过率会涨 ——
        #    那不是模型变强，是尺子变松了
        if state.inventory != obj.obj_id:
            return False, FailReason.NOT_HOLDING, f"手上不是 {a0}，当前持 {state.inventory}"
        # ④ 找容器（放到哪）
        recept = state.find(a1)
        # ⑤ 容器找不到也判失败，不能放行。
        #    模型引用一个场景里不存在的容器，本身就是幻觉。
        #    会不会误伤 gold？不会 —— gold 引用过的固定器具（微波炉、水槽）
        #    已经在 alfred.py 的 _augment_scene 里补全进场景了。
        #    这一行曾经被误改成「找不到就用传入的名字」，
        #    结果通过率从 98.4% 变 98.7%，数字一动就被门槛抓住了
        if recept is None:
            return False, FailReason.RECEP_NOT_FOUND, f"场景里找不到容器 {a1}"
        # ⑥ 【已知漏洞，暂未修复】放下也本该校验位置，原因与 ④ 相同，
        #    一起记为待改进项。当前不校验的后果是「路径效率」指标失真。
        # ⑦ 物体的所在容器改成目标容器
        obj.parent = recept.obj_id
        # ⑧ 手清空
        state.inventory = None
        return True, FailReason.OK, f"把 {obj.obj_type} 放到 {a1}"

    # ------------------------------------------- 清洁 / 加热 / 冷却
    # 这三个动作共用一段代码，因为逻辑一模一样，只有两处不同：
    #   允许的器具不同（见第 78 行的 TOOL_RECEPTACLES）
    #   改的状态字段不同（cleaned / heated / cooled）
    if act in TOOL_RECEPTACLES:
        if a0:
        # ① 找物体
            obj = state.find(a0)
        elif state.inventory:
            obj = state.find(state.inventory)
        else:
            obj = None
        # ② 场景里没这东西 → 失败
        if obj is None:
            return False, FailReason.OBJ_NOT_FOUND, f"场景里找不到 {a0}"
        # ③ 物体必须拿在手上。隔空洗东西不算
        if state.inventory != obj.obj_id:
            return False, FailReason.NOT_HOLDING, f"{act} 要求手上持有 {a0}"
        # ④ 取出这个动作允许的器具集合，比如加热只允许 microwave
        allowed = TOOL_RECEPTACLES[act]
        # ⑤ 当前位置归一化。state.location 可能是 "Microwave_1a2b3c" 带哈希，
        #    而 allowed 里存的是 "microwave"，所以要转小写再比对
        loc = normalize(state.location or "")
        # ⑥ 必须站在对应器具前面。
        #    模型最容易漏的就是这个：拿了东西直接加热，忘了先走到微波炉前
        if not any(x in loc for x in allowed):
            return (
                False,
                FailReason.WRONG_LOCATION,
                f"{act} 需要在 {'/'.join(allowed)} 前执行，当前在 {state.location}",
            )
        # ⑦ 三关全过，改对应的状态字段
        if act == "CleanObject":
            obj.cleaned = True
        elif act == "HeatObject":
            obj.heated = True
        else:
            obj.cooled = True
        return True, FailReason.OK, f"{act} 作用于 {obj.obj_type}"

    # ---------------------------------------------------------------- 切片
    if act == "SliceObject":
        # ① 找物体
        obj = state.find(a0)
        # ② 场景里没这东西 → 失败
        if obj is None:
            return False, FailReason.OBJ_NOT_FOUND, f"场景里找不到 {a0}"
        # ③ 手里拿的必须是刀。判断方式是看 inventory 里带不带 knife 字样
        #    （ButterKnife 也算，是刻意放宽的）
        knife = state.inventory
        if knife is None or "knife" not in normalize(knife):
            return False, FailReason.NEED_KNIFE, f"切片需要手持刀，当前持 {state.inventory}"
        # ④ 打上切片标记。注意切片不消耗刀，刀还在手上
        obj.sliced = True
        return True, FailReason.OK, f"切片 {obj.obj_type}"

    # ---------------------------------------------------------------- 开关
    if act == "ToggleObject":
        # ① 找物体
        obj = state.find(a0)
        # ② 场景里没这东西 → 失败
        if obj is None:
            return False, FailReason.OBJ_NOT_FOUND, f"场景里找不到 {a0}"
        # ③ 类型必须可开关。对苹果按开关没有意义，要拒绝。
        #    白名单在第 86 行的 TOGGLEABLE
        if normalize(obj.obj_type) not in TOGGLEABLE:
            return False, FailReason.NOT_TOGGLEABLE, f"{obj.obj_type} 不是可开关物体"
        # ④ 取反。关着的打开，开着的关掉
        obj.toggled = not obj.toggled
        return True, FailReason.OK, f"切换 {obj.obj_type}"

    # ------------------------------------------------------------- 空操作
    # 不做任何检查，也不改任何状态。留着它是为了让模型有个「结束」可输出，
    # 不然它会硬凑一个动作出来
    if act in ("NoOp", "None", "End"):
        return True, FailReason.OK, "空操作"

    # ---------------------------------------------------------------- 兜底
    # 走到这说明七个分支一个都没匹配上，动作名不认识。
    # 判失败而不是忽略：模型编了个不存在的动作，这本身就是问题
    return False, FailReason.UNKNOWN_ACTION, f"未识别的动作 {act}"


def build_initial_state(scene_objects: list[dict]) -> WorldState:
    """从场景物体列表构造初始状态。

    入参由 alfred.py 抽取，每项形如：
        {"objectId": "Apple_1", "objectType": "Apple", "parentReceptacles": [...]}

    注意容器初始值在这里并不太重要：任务是让物体最终出现在目标容器里，
    而初始位置会被 PutObject 覆盖。真正关键的是物体「存在」。
    """
    # ① 从空状态开始：机器人没位置、手是空的、物体字典空的
    state = WorldState()
    # ② 逐个物体登记进字典
    for i, o in enumerate(scene_objects):
        # ③ 取完整 id。两个字段都拿不到时兜底造一个 obj0、obj1……，
        #    宁可名字难看也别让这一条样本整个崩掉
        oid = o.get("objectId") or o.get("objectType") or f"obj{i}"
        # ④ 取初始所在容器。ALFRED Lite 里这个字段基本是空的
        #    （原始 JSON 压根没写 parentReceptacles），所以多半落到 None
        parents = o.get("parentReceptacles") or []
        # ⑤ 用归一化后的名字当字典键，find() 查的时候也用同一套规则。
        #    注意只取 parents[0] —— 一个物体同时挂在多个容器下的情况不处理
        state.objects[normalize(oid)] = ObjectState(
            obj_id=oid,
            obj_type=o.get("objectType") or o.get("objectId") or oid,
            parent=parents[0] if parents else None,
        )
    return state


def check_goal(state: WorldState, params: dict) -> tuple[bool, dict]:
    """检查目标是否达成。params 来自 ALFRED 的 pddl_params。

    pddl_params 的典型内容：
        object_target   要操作的物体（如 "AlarmClock"）
        parent_target   最终要放到哪（如 "Desk"）
        mrecep_target   可移动容器目标（如 "Basket"），有的话优先于 parent_target
        toggle_target   要开关的物体（如 "DeskLamp"）
        object_sliced   是否要求物体被切片

    为什么要返回每个子项的达成情况，而不只是一个布尔值：
    方案里的「子目标完成率」指标需要这个粒度。一条任务失败了，但完成了
    3 个子目标中的 2 个，和完全没动，是完全不同的两回事。只报成功率会把
    这个差别抹掉 —— 这正是 GM-100 引入 PSR 指标要解决的问题，
    本项目独立地也想到了这一点。
    """
    # ① 逐项检查的结果都塞进这个字典，最后一起返回
    detail: dict[str, bool] = {}

    # ② 把完成条件里的几个字段取出来，取不到就当空字符串。
    #    ALFRED 用空字符串表示「这项不要求」，不是 None，两种都要兜住
    obj_target = params.get("object_target") or ""
    parent_target = params.get("parent_target") or ""
    mrecep_target = params.get("mrecep_target") or ""
    toggle_target = params.get("toggle_target") or ""
    need_sliced = bool(params.get("object_sliced"))

    # ③ 先确认目标物体在不在场景里
    obj = state.find(obj_target) if obj_target else None
    if obj_target:
        detail["目标物体存在"] = obj is not None

        # ④ 决定「应该放在哪」。可移动容器目标优先，没有才用普通容器。
        #    这个 or 的优先级顺序是 mrecep 任务能判对的关键
        final_parent = mrecep_target or parent_target
        if final_parent:
            # ⑤ 找出目标容器在场景里的完整 id，好跟物体的 parent 比对
            recept = state.find(final_parent)
            recept_id = recept.obj_id if recept else None
            # ⑥ 三个条件全满足才算放对：物体存在、它有归属、
            #    且归属归一化后等于目标容器归一化后的名字
            placed = (
                obj is not None
                and obj.parent is not None
                and normalize(obj.parent) == normalize(recept_id or final_parent)
            )
            detail[f"放置在{final_parent}"] = placed
        # ⑦ 切片是独立的一项要求，和「放在哪」互不干涉
        if need_sliced:
            detail["已切片"] = bool(obj and obj.sliced)

    # ⑧ 开关类任务额外查一项
    if toggle_target:
        t = state.find(toggle_target)
        detail[f"已切换{toggle_target}"] = bool(t and t.toggled)

    # ⑨ 全部子项都 True 才算达成。一项都没查（detail 空）算没达成，
    #    不能因为「没要求」就白送一个通过
    achieved = all(detail.values()) if detail else False
    return achieved, detail


def run_plan(
    lines: list[str],
    scene_objects: list[dict],
    params: dict,
    stop_on_violation: bool = True,
) -> RunResult:
    """执行一整条动作序列，返回执行结果。

    stop_on_violation 默认为 True：遇到前置条件违反就中止。
    为什么不继续推演：违反之后的动作是在一个「不可能存在的状态」上继续
    算的，算出来的结论没有意义，还会污染后续的失败原因统计。

    ==========================================================================
    返回的 violations 是本项目的核心指标「前置条件违反率」的来源。
    ==========================================================================
    它量化的是模型的规划幻觉：生成了一串物理上执行不了的动作。
    这个指标在文本生成类任务里不存在，因为文本没有物理约束。
    """
    # ① 先把场景读进来，得到一份初始记账本
    state = build_initial_state(scene_objects)
    # ② 三个计数器：逐步结果记 steps，成功数 executed，违反数 violations
    steps: list[StepResult] = []
    violations = 0
    executed = 0

    # ③ 一行一行执行
    for i, line in enumerate(lines):
        # ④ 拆成 (动作名, 参数)。拆不出来 action 是 None
        action, args = parse_action_line(line)
        if action is None:
            # ⑤ 解析失败：记一条 UNPARSEABLE 的坏步，算一次违反
            steps.append(StepResult(i, line, None, [], False, FailReason.UNPARSEABLE, "空行或无法解析"))
            violations += 1
            # ⑥ 默认遇到违反就停。继续推演是在一个不可能存在的状态上算，
            #    算出来的东西没意义，还会污染失败原因统计
            if stop_on_violation:
                break
            continue

        # ⑦ 真正执行这一步，原地改 state
        ok, reason, detail = apply_action(state, action, args)
        steps.append(StepResult(i, line, action, args, ok, reason, detail))
        if ok:
            executed += 1
        else:
            violations += 1
            if stop_on_violation:
                break

    # ⑧ 不管执行到哪（哪怕中途断了），都拿当前状态去对完成条件算一次账
    achieved, detail = check_goal(state, params)
    return RunResult(
        steps=steps,
        final_state=state,
        goal_achieved=achieved,
        violations=violations,
        executed=executed,
        goal_detail=detail,
    )
