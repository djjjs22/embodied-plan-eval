"""关卡 1：动作解析与两个核心动作的状态转移。

这一关是整个项目的起点。写完之后你会明白「符号状态机」到底在干什么。

规则：先自己写，写完再对照 evalkit/simulator.py。
      直接抄答案的话测试会过，但脑子里是空的。

验收：python -m pytest tests/test_level1.py -v
"""
from __future__ import annotations
import re


from dataclasses import dataclass, field


@dataclass
class Obj:
    """场景里的一个物体。"""

    name: str          # 如 "Apple_1"
    otype: str         # 类型名，如 "Apple"
    parent: str | None = None   # 所在容器，None 表示拿在手上


@dataclass
class World:
    """世界状态。ALFRED 规则下一次只能手持一个物体。"""

    location: str | None = None
    inventory: str | None = None      # 手持物体的 name
    objects: dict[str, Obj] = field(default_factory=dict)

    def find(self, name: str) -> Obj | None:
        """按名字找物体。先匹配 name，再匹配类型名。

        注意 gold 动作里的参数是纯类型名小写（如 "apple"），
        而场景里的物体叫 "Apple_1"，类型叫 "Apple"。
        所以匹配时要忽略大小写。
        """
        if not name:
            return None
        key = name.lower()
        for o in self.objects.values():
            if o.name.lower() == key or o.otype.lower() == key:
                return o
        return None


def parse_action_line(line: str) -> tuple[str | None, list[str]]:
    """把一行动作文本拆成 (动作名, 参数列表)。

    要处理的输入形态：
        "PickupObject(Apple_1)"        -> ("PickupObject", ["Apple_1"])
        "PutObject(Apple_1, Fridge_1)" -> ("PutObject", ["Apple_1", "Fridge_1"])
        "1. PickupObject(Apple_1)"     -> ("PickupObject", ["Apple_1"])   行首有序号
        ""                             -> (None, [])

    提示：
    1. 先剥掉行首可能的序号（数字加点、顿号、右括号，或者 - * •）
    2. 有括号就从括号里取参数，按逗号切分
    3. 空行返回 (None, [])
    """
    line = line.strip()  # 移除整个字符串的空格
    if not line:
        return (None, [])

    # 1. 去除行首序号
    line = re.sub(r"^\s*(?:\d+[\.、\)]|[-*•])\s*", "", line)

    # 2. 按第一个左括号切割
    if "(" in line:
        head, _, rest = line.partition("(")
        # 移除末尾的右括号
        rest = rest.rstrip(")")
        # 按逗号分割并去除每个参数的空格
        args = [a.strip() for a in rest.split(",") if a.strip()]
        return (head.strip(), args)
    else:
        # 没有括号的情况（根据提示，我们可能不需要处理，但为了安全起见）
        # 提示说：“有括号就从括号里取参数”，所以假设都有括号
        return (line.strip(), [])


def apply_pickup(world: World, arg: str) -> tuple[bool, str]:
    """拿起物体。成功返回 (True, "")，失败返回 (False, 失败原因)。

    要检查两件事：
    1. 场景里有没有这个物体
    2. 手上是不是空的（ALFRED 规则：一次只能拿一个）

    成功时要做两件事：
    1. world.inventory 设成这个物体
    2. 物体的 parent 设成 None（离开原来的容器）

    失败原因写成人话，比如 "场景里找不到 Apple_1"、
    "手上已有 Bread_1，无法再拿 Apple_1"。
    """

    # 1. 拿 arg 去场景里找物体
    obj = world.find(arg)

    # 2. 找不到的情况
    if obj is None:
        return (False, f"场景里找不到 {arg}")

    # 3. 手上有东西的情况（注意这里改成了 '手上'，且直接引用 world.inventory）
    if world.inventory is not None:
        return (False, f"手上已有 {world.inventory}，无法再拿 {arg}")

    # 4. 成功：w.inventory 设成这个物体的 name（注意这里按测试要求存名字）
    world.inventory = obj.name

    # 5. 成功：物体的 parent 设成 None
    obj.parent = None

    # 6. 返回成功
    return (True,"")
    #raise NotImplementedError

def apply_put(world: World, obj_arg: str, recep_arg: str) -> tuple[bool, str]:
    # ① 先找被放的那个物体
    obj = world.find(obj_arg)
    # ② 场景里没这东西 → 失败
    if obj is None:
        return (False, f"场景里找不到 {obj_arg}")
    # ③ 手上拿的不是它 → 失败。
    #    没拿到东西就说要放下，这是最典型的规划幻觉，必须拦住
    if world.inventory != obj.name:
        return (False, f"手上不是 {obj_arg}，当前持 {world.inventory}")
    # ④ 找容器。找不到不算错 —— 微波炉水槽这类固定器具压根没登记，
    #    这种情况直接用动作里写的名字当容器标识
    recept = world.find(recep_arg)
    container = recept.name if recept else recep_arg
    # ⑤ 物体挪进容器
    obj.parent = container
    # ⑥ 手空了
    world.inventory = None
    return (True, "")
