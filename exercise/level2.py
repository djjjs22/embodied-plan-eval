"""关卡 2：清洁 / 加热 / 冷却 这三个动作的目标解析。

这一关修的是 4 条 gold 失败里**唯一一条真 bug**。

背景（先读，别急着写代码）：
    315 条样本里，CleanObject / HeatObject / CoolObject 一共出现 135 次。
    其中 134 次是带参数的，长这样：
        CleanObject(cloth)
    只有 1 次不带参数，长这样：
        CleanObject()
    这 1 条的原始 JSON 是 {'action': 'CleanObject', 'args': ['']}，
    参数是个空字符串，不是没有参数。

    现在 simulator.py 里 a0 = "" 传进 state.find("")，
    find 的第一行 `if not name: return None` 直接返回 None，
    于是报「物体不在场景中」。但这是误报 —— 物体在手上，只不过没写名字。

    ALFRED 的语义：CleanObject 清洗的就是手里拿着的东西，
    所以参数为空时应该默认取 inventory。

规则：先自己写，写完再对照 evalkit/simulator.py 里 `if act in TOOL_RECEPTACLES:` 那一段。
      直接抄答案的话测试会过，但脑子里是空的。

验收：python -m pytest tests/test_level2.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Obj:
    name: str
    otype: str
    parent: str | None = None
    cleaned: bool = False
    heated: bool = False
    cooled: bool = False


@dataclass
class World:
    location: str | None = None
    inventory: str | None = None      # 手持物体的 name
    objects: dict[str, Obj] = field(default_factory=dict)

    def find(self, name: str) -> Obj | None:
        """按名字找物体。先匹配 name，再匹配类型名，忽略大小写。"""
        if not name:
            return None
        key = name.lower()
        for o in self.objects.values():
            if o.name.lower() == key or o.otype.lower() == key:
                return o
        return None


# 这三个动作必须站在对应器具前面才能做，这是 ALFRED 的物理约束。
# 模型最容易犯的错就是拿了东西直接加热，忘了先走到微波炉前。
TOOL_RECEPTACLES = {
    "CleanObject": {"sink", "sinkbasin", "bathroom sink"},
    "HeatObject": {"microwave"},
    "CoolObject": {"fridge"},
}


def resolve_target(world: World, arg: str) -> Obj | None:
    """确定动作作用在哪个物体上。

    两档逻辑：
        1. arg 非空  -> 用 world.find(arg) 找，找不到返回 None
        2. arg 为空  -> 默认作用在手上持有的物体（world.inventory）
                        手上没东西就返回 None

    为什么需要第 2 档：ALFRED 语义里 CleanObject 洗的就是手里的东西，
    有些标注省略了参数。参数为空不代表动作无效。

    返回 None 时，调用方会报「物体不在场景中」。注意区分两种情况：
        - arg 非空但找不到  -> 真的不在场景里，或者名字没对齐，是真错误
        - arg 为空且手上是空的 -> 手上没东西却要清洗，也是错误
    这两种都返回 None，但报错文案要区分开，否则排查时看不出是哪一种。
    """
    if arg:
        return world.find(arg)
    else:
        if world.inventory is not None:
            return world.find(world.inventory)
        else:
            return None


def apply_tool_action(world: World, action: str, arg: str) -> tuple[bool, str]:
    """执行清洁 / 加热 / 冷却。成功返回 (True, "")，失败返回 (False, 失败原因)。

    按顺序检查三件事：
        1. 目标物体存不存在（用 resolve_target）
        2. 目标是不是拿在手上（对比 world.inventory 和 obj.name）
        3. 当前位置对不对（world.location 里要含有 TOOL_RECEPTACLES[action] 中的一个）

    三关都过了才改状态：
        CleanObject -> obj.cleaned = True
        HeatObject  -> obj.heated = True
        CoolObject  -> obj.cooled = True

    失败原因写成人话，比如：
        "手上不是 Bread_1，当前持 Apple_1"
        "HeatObject 需要在 microwave 前执行，当前在 countertop"

    位置比对提示：world.location 可能是 "Microwave_1a2b3c" 这种带哈希的，
    而 TOOL_RECEPTACLES 里存的是 "microwave"，所以要转小写后做子串包含判断。
    """
    obj = resolve_target(world,arg)
    if obj is None:
        return (False,"找不到目标物体")
    if obj.name != world.inventory:
        return (False, f"手上不是 {obj.name}，当前持 {world.inventory}")
    allow_locations = TOOL_RECEPTACLES[action]
    is_location_ok = False
    for loc in allow_locations:
        if loc in (world.location or "").lower():
            is_location_ok = True
            break
    #         位置检查完全没生效。模型说"站在餐桌前用微波炉加热苹果"，你的状态机会说"没问题"。
    #
    # 这就是关卡 1 那两行注释掉的代码——尺子松了，量出来的分数好看但没意义。一模一样的坑，换了个形式又出现一次。
    if not is_location_ok:
        return (False, f"{action} 需要在 {'/'.join(allow_locations)} 前执行，当前在 {world.location}")

# 三关都过了，执行状态更改
    if action == "CleanObject":
        obj.cleaned = True
    elif action == "HeatObject":
        obj.heated = True
    elif action == "CoolObject":
        obj.cooled = True

    # 最终交卷
    return (True, "")