"""ALFRED 数据解析。

================================================================================
这个文件解决什么问题
================================================================================

评测需要题目和标准答案。这个文件负责把 ALFRED 的原始 JSON 转成能直接
喂给状态机的样本。

只依赖 Lite 版（35MB 轨迹 JSON），不读图片和特征，所以不需要 GPU 也
不需要仿真器。这是整个项目能在一周内做完的前提。

================================================================================
为什么要单独抽一个文件，而不是在脚本里直接读 JSON
================================================================================

数据解析里有三个坑（见下方各处注释），每一个都可能导致全量样本失效。
集中在一个文件里，改动只改一处，也方便定位问题。

================================================================================
一条 ALFRED 样本的结构
================================================================================

    task_type           7 种任务类型之一，天然就是难度分层
    pddl_params         目标条件（object_target / parent_target /
                        mrecep_target / toggle_target / object_sliced）
    turk_annotations    人工标注，anns[0].task_desc 是给模型的任务描述
    plan.high_pddl      gold 高层动作序列，每项含 discrete_action{action, args}
    scene.object_poses  场景里的物体

前三样是给模型的输入，第四样是标准答案，第五样是场景。

================================================================================
三个坑（都真实踩过）
================================================================================

坑一：任务类型名。官方文档写 pick_clean_then_place，真实数据是
      pick_clean_then_place_in_recep。匹配不上会静默落到默认难度，
      而且因为默认恰好是对的，光看结果还发现不了。

坑二：场景字段。官方写 objectId/objectType，真实数据是 objectName，
      格式还是「类型名_哈希」（Mug_2a940808）。gold 参数却用纯类型名
      小写（mug）。不对齐的话全部样本判成「物体不在场景中」。

坑三：固定器具缺失。微波炉、水槽、落地灯不在 object_poses 里，
      但 gold 动作会引用。不补全的话几乎每条都卡在「容器不在场景中」。

这三个坑加起来，第一版跑出来是 140 条 0% 通过。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .simulator import normalize

# ALFRED 的物体命名是「类型名_哈希」，如：
#     Mug_2a940808
#     AlarmClock_1a2b3c4d
_HASH_SUFFIX = re.compile(r"_[0-9a-z]{6,}$", re.IGNORECASE)


def extract_object_type(name: str) -> str:
    """从 objectName 提取类型名。

        Mug_2a940808  -> Mug
        CounterTop_1  -> CounterTop
        Apple         -> Apple

    为什么需要：真实数据里场景物体名带哈希后缀，而 gold 动作的参数
    是纯类型名小写。两边不对齐，每条样本都会判成「物体不在场景中」。
    """
    # ① 空值直接返回空，别让正则去啃 None
    if not name:
        return ""
    # ② 用正则把 "_哈希" 那一段删掉。哈希是 6 位以上十六进制，
    #    所以 "CounterTop_1" 这种短数字后缀不会被误删，还是 CounterTop
    return _HASH_SUFFIX.sub("", name) or name


# ============================================================================
# 7 种官方任务类型天然就是难度阶梯，不需要另外设计分层
# ============================================================================
#
# 真实数据里 clean / heat / cool 三种带 _in_recep 后缀，
# 和官方 README 写的名字不一致。两种写法都收，否则会静默落到默认难度。
DIFFICULTY_BY_TASK_TYPE = {
    "pick_and_place_simple": "L1",
    "look_at_obj_in_light": "L1",
    "pick_clean_then_place": "L2",
    "pick_clean_then_place_in_recep": "L2",
    "pick_heat_then_place": "L2",
    "pick_heat_then_place_in_recep": "L2",
    "pick_cool_then_place": "L2",
    "pick_cool_then_place_in_recep": "L2",
    "pick_two_obj_and_place": "L3",
    "pick_and_place_with_movable_recep": "L3",
}

DIFFICULTY_ORDER = ["L1", "L2", "L3"]


@dataclass
class PlanCase:
    """一条评测样本。

    字段分成三组：
        给模型的输入   task_desc、scene_objects
        标准答案       gold_plan、pddl_params
        分层标签       task_type、difficulty、dimensions
    """
    id: str
    task_desc: str
    task_type: str
    difficulty: str
    gold_plan: list[str]           # 形如 ["PickupObject(Apple)", ...]
    scene_objects: list[dict]
    pddl_params: dict
    split: str = ""
    dimensions: list[str] = field(default_factory=list)


def _augment_scene(objects: list[dict], gold_actions: list[tuple[str, list]]) -> list[dict]:
    """补全固定家具与器具（坑三的解法）。

    ALFRED Lite 的 scene.object_poses 只登记可移动物体（苹果、杯子），
    微波炉、冰箱、水槽、落地灯这些固定器具不在里面。但 gold 动作会引用
    它们 —— 比如 PutObject(egg, microwave)，microwave 就不在列表里。

    不补的话几乎每条样本都卡在「容器不在场景中」。第一版跑出来
    违反原因统计里这一项有 119 条，就是这么来的。

    补全依据是 gold 动作的参数：专家演示是在完整场景里录的，
    gold 引用过的名字必然真实存在，用它反推场景是合法的。

    会不会放水？不会。放水指模型引用不存在的物体时放过它，
    而这里补的是环境本来就有的东西，只是 Lite 版没登记。
    模型引用了 gold 都没提过的物体，照样判失败。
    """
    # ① 先把场景里已有的类型名收成集合，用来判断「这名字是不是新的」
    known = {normalize(o.get("objectType") or "") for o in objects}
    # ② 待补清单。用 dict 而不是 list 是为了自动去重，
    #    同一个微波炉被 gold 引用十次也只补一条
    added: dict[str, dict] = {}
    # ③ 扫 gold 动作的每一个参数。依据是：专家是在完整场景里录的演示，
    #    他引用过的名字必然真实存在，用它反推场景是合法的
    for _act, args in gold_actions:
        for a in args:
            n = normalize(a)
            # ④ 场景里没有、也还没补过 → 补一条进去
            if n and n not in known and n not in added:
                added[n] = {"objectId": a, "objectType": a, "parentReceptacles": []}
    # ⑤ 原场景 + 补出来的固定器具，一起返回
    return objects + list(added.values())


def _infer_dimensions(task_type: str, params: dict, high_descs: list[str]) -> list[str]:
    """推断能力维度标签。

    难度（L1/L2/L3）回答「有多难」，维度回答「考什么能力」。
    两个轴独立，报告里才能说清模型到底哪项能力不行。

    比如统计出「状态改变-温度」这个维度全崩，那就是模型不理解
    「加热要先走到微波炉前」这类物理流程，改进方向非常明确。
    """
    # ① 每种任务都考「动作序列生成」，先无条件打上
    dims = ["动作序列生成"]
    # ② 下面按任务类型名里的关键词打标签。
    #    关键词匹配而不是精确匹配，因为真实数据带 _in_recep 这类后缀
    if "clean" in task_type:
        dims.append("状态改变-清洁")
    if "heat" in task_type or "cool" in task_type:
        dims.append("状态改变-温度")
    if "two_obj" in task_type:
        dims.append("多物体状态管理")
    if "movable_recep" in task_type:
        dims.append("可移动容器规划")
    if "light" in task_type:
        dims.append("开关操作")
    # ③ 切片不是任务类型，是完成条件里的一项，要单独看
    if params.get("object_sliced"):
        dims.append("切片前置依赖")
    # ④ 人工标注的自然语言步骤数 ≥ 4 就算长程
    if len(high_descs) >= 4:
        dims.append("长程多步")
    return dims


def load_traj(path: Path) -> PlanCase:
    """读一条轨迹并转成 PlanCase。

    读取顺序有讲究：先解析 gold 拿到动作参数，再构造场景物体。
    因为补全场景需要用到 gold 的参数（见 _augment_scene）。
    """
    # ① 整个原始文件读进来，就是一个大字典
    j = json.loads(Path(path).read_text(encoding="utf-8"))

    # ---------------------------------------------------------- 任务描述
    # ② anns 里有多份人工标注，取第一份就行 —— 它们描述的是同一个任务。
    #    用 or [] 兜底：没有标注时不能崩，后面靠空字符串识别出来
    anns = (j.get("turk_annotations") or {}).get("anns") or []
    task_desc = anns[0]["task_desc"] if anns else ""
    # ③ high_descs 是人工写的分步描述，用来推断「长程多步」这个维度
    high_descs = anns[0].get("high_descs", []) if anns else []

    # ---------------------------------------------------------- gold 序列
    # ④ high_pddl 每项形如
    #     {"discrete_action": {"action": "PutObject", "args": ["bread", "microwave"]}}
    #    遍历它，同时攒两份：给模拟器用的字符串、给补场景用的原始参数
    gold = []
    gold_actions: list[tuple[str, list]] = []
    for step in (j.get("plan") or {}).get("high_pddl") or []:
        da = step.get("discrete_action") or {}
        act = da.get("action")
        args = da.get("args") or []
        # ⑤ 没动作名的跳过。一条数据坏了不该中断整个导入流程
        if not act:
            continue
        gold_actions.append((act, [str(a) for a in args]))
        # ⑥ 统一转成 "PutObject(bread, microwave)" 这种字符串。
        #    模拟器接收的就是这个格式，也方便跟模型输出直接比
        gold.append(f"{act}({', '.join(str(a) for a in args)})" if args else f"{act}()")

    # ---------------------------------------------------------- 场景物体
    scene = j.get("scene") or {}
    objects = []
    for o in scene.get("object_poses") or []:
        # ⑦ 坑二：真实字段是 objectName，不是官方文档写的 objectId / objectType
        raw = o.get("objectName") or o.get("objectId") or ""
        objects.append(
            {
                "objectId": raw,
                # ⑧ 从 "Mug_2a940808" 剥出 "Mug"。
                #    因为 gold 参数用的是纯类型名，不剥就永远匹配不上
                "objectType": extract_object_type(raw),
                "parentReceptacles": o.get("parentReceptacles") or [],
            }
        )
    # ⑨ 坑三：补上固定器具（微波炉、冰箱、水槽、落地灯），
    #    不补的话后面几乎每条都卡在「容器不在场景中」
    objects = _augment_scene(objects, gold_actions)

    # ---------------------------------------------------------- 所属子集
    # ⑩ 从文件路径里认子集。ALFRED 分 train / valid_seen / valid_unseen /
    #    tests_seen / tests_unseen，其中 valid_unseen 的场景训练时没出现过，
    #    最适合做泛化测试
    parts = Path(path).parts
    split = ""
    for p in parts:
        if p in ("train", "valid_seen", "valid_unseen", "tests_seen", "tests_unseen"):
            split = p
            break

    # ⑪ 坑一：任务类型名带 _in_recep 后缀，跟官方 README 不一致。
    #    查表查不到时兜底给 L2，不是 L1 —— 兜底错了会让难度分布歪掉
    task_type = j.get("task_type") or ""
    return PlanCase(
        id=j.get("task_id") or Path(path).parent.name,
        task_desc=task_desc,
        task_type=task_type,
        difficulty=DIFFICULTY_BY_TASK_TYPE.get(task_type, "L2"),
        gold_plan=gold,
        scene_objects=objects,
        pddl_params=j.get("pddl_params") or {},
        split=split,
        dimensions=_infer_dimensions(task_type, j.get("pddl_params") or {}, high_descs),
    )


def iter_traj_files(root: Path, limit: int | None = None):
    """遍历轨迹文件。

    加 limit 是因为 ALFRED 有近 8000 个文件，全量扫描在慢机器上会卡。
    抽样时给个上限，够用就停。
    """
    # ① rglob 递归找所有 traj_data.json。用 yield 而不是一次性返回列表，
    #    因为全量近 8000 个文件，先攒成列表会很占内存
    n = 0
    for p in Path(root).rglob("traj_data.json"):
        yield p
        n += 1
        # ② 到上限就收手。调试时给个小的 limit，几秒钟就能跑通
        if limit and n >= limit:
            return


def _scene_objects(case: PlanCase | dict) -> list[dict]:
    """取出场景物体列表，兼容两种来源。

    load_traj() 产出的是 PlanCase，而 dataset/cases.jsonl 反序列化出来
    是 dict。整条评测链路（run_eval / verify_gold）用的是 dict，
    所以两种都得接，不然一跑就 AttributeError。
    """
    # ① 从 dataset/cases.jsonl 读出来的是 dict，走这条
    if isinstance(case, dict):
        return case.get("scene_objects") or []
    # ② 从 load_traj() 直接拿到的是 PlanCase 对象，走这条。
    #    两种都得接，不然一跑就 AttributeError
    return list(getattr(case, "scene_objects", []) or [])


def describe_scene_for_prompt(case: PlanCase | dict, max_objects: int = 40) -> dict:
    """生成给模型看的场景摘要。

    关键约束：只给这条样本真实存在的物体。

    为什么强调：如果场景里凭空多出物体，模型可能去操作根本不存在的东西，
    测出来的就不是规划能力而是想象力。反过来漏了物体，模型又会被冤枉。

    max_objects 是防止场景太大把 prompt 撑爆。
    """
    # ① 取场景物体列表（dict 和 PlanCase 两种来源都兼容）
    scene_objects = _scene_objects(case)
    # ② 抽类型名并排序去重。排序是为了让同一条样本每次生成同样的 prompt，
    #    不然缓存键会飘，重复调用就白花钱了
    types = sorted({o.get("objectType") or o.get("objectId") for o in scene_objects if o})
    # ③ 容器名单独抽一份。物体可能挂在多个容器下，这里展平去重
    receptacles = sorted(
        {r for o in scene_objects for r in (o.get("parentReceptacles") or [])}
    )
    # ④ 两边都按 max_objects 截断，防止场景太大把 prompt 撑爆。
    #    注意只给这两样 —— gold_plan 和 pddl_params 绝不出现在给模型的内容里
    return {
        "objects": [t for t in types if t][:max_objects],
        "receptacles": receptacles[:max_objects],
    }
