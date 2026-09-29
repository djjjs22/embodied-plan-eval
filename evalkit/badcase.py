"""bad case 归因体系。

================================================================================
为什么需要三层
================================================================================

只报「失败 143 条、手上已有物体 23 条」是没用的。这个数字不能驱动任何动作：
你知道了 23 条这类错误，但你不知道该改什么。

归因要能回答三个层层递进的问题：

    第一层 性质：这是哪一类问题？（物理常识 / 场景理解 / 格式 / 任务理解）
    第二层 根因：具体在第几步、违反了哪条约束？（能定位到动作）
    第三层 回归：怎么防止它复发？（沉淀成可复跑的用例集）

第三层是闭环的关键。没有它，评测就只是个打分器，不能驱动改进 ——
改完 prompt 之后你不知道同类错误有没有减少，只能凭感觉。

================================================================================
四类败因的区分依据
================================================================================

按「改进方向」分，不按「错误现象」分。因为归因的目的是指导下一步做什么，
所以同一类里的问题应该用同一种手段解决。

    物理常识   模型不知道物理约束      改 prompt 里的约束描述
    场景理解   模型不知道场景有什么    改 prompt 里的场景呈现方式
    输出格式   模型不遵守输出要求      改 prompt 里的格式说明，或加后处理
    任务理解   动作都能做但目标不对    改 prompt 里的目标描述

这四类的修复手段完全不同，混在一起就没法行动了。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .simulator import FailReason

# ============================================================================
# 归因表：每一类失败对应的性质、根因描述、修复方向
# ============================================================================
#
# 这张表是给人看的，不是给程序看的。写的时候想着「如果面试官问这类错误
# 该怎么修，我能答上来吗」。答不上来的条目说明还没想清楚，不如不写。

TAXONOMY: dict[str, dict] = {
    FailReason.HANDS_FULL.value: {
        "nature": "物理常识",
        "root_cause": "未先放下手中物体就尝试拿取下一个",
        "fix": "在 prompt 里强化「一次只能手持一个物体」的约束，或要求模型显式记录当前手持状态",
    },
    FailReason.NOT_HOLDING.value: {
        "nature": "物理常识",
        "root_cause": "未持有目标物体就尝试对它执行操作（清洗/加热/放下）",
        "fix": "在 prompt 中强调「操作前必须已经在手上」，并给出正确示例",
    },
    FailReason.WRONG_LOCATION.value: {
        "nature": "物理常识",
        "root_cause": "未先移动到对应器具前就执行清洗/加热/冷却",
        "fix": "在 prompt 里把「器具前置条件」单列成一条硬性规则，并示例先移动再操作",
    },
    FailReason.NEED_KNIFE.value: {
        "nature": "物理常识",
        "root_cause": "切片前未先拿起刀",
        "fix": "补充工具依赖说明：切片必须手持刀，切完要放下刀才能拿被切物体",
    },
    FailReason.NOT_TOGGLEABLE.value: {
        "nature": "物理常识",
        "root_cause": "对不可开关的物体执行了开关动作",
        "fix": "在场景描述里标注哪些物体可开关，避免模型凭常识猜测",
    },
    FailReason.OBJ_NOT_FOUND.value: {
        "nature": "场景理解",
        "root_cause": "引用了场景中不存在的物体",
        "fix": "强化「只使用给定物体清单」的约束；检查场景物体列表是否给全",
    },
    FailReason.RECEP_NOT_FOUND.value: {
        "nature": "场景理解",
        "root_cause": "引用了场景中不存在的容器或器具",
        "fix": "把容器与器具清单完整给出，并禁止模型创造新容器",
    },
    FailReason.UNKNOWN_ACTION.value: {
        "nature": "输出格式",
        "root_cause": "使用了动作白名单之外的动作名",
        "fix": "在 prompt 末尾重申可用动作列表，或加后处理把近义动作映射回白名单",
    },
    FailReason.UNPARSEABLE.value: {
        "nature": "输出格式",
        "root_cause": "输出无法解析成动作（格式错误或混入解释文字）",
        "fix": "加严输出格式要求；考虑用 response_format 强制 JSON",
    },
    "目标未达成（动作均可执行）": {
        "nature": "任务理解",
        "root_cause": "每一步动作都能执行，但最终状态不满足目标条件",
        "fix": "优化 prompt 里的目标描述，把放置位置、物体状态说清楚",
    },
}

# 兜底：没登记的失败原因归到这里，提醒补表
FALLBACK = {
    "nature": "未分类",
    "root_cause": "该失败原因尚未登记归因",
    "fix": "补充到 evalkit/badcase.py 的 TAXONOMY 表里",
}


@dataclass
class Attribution:
    """一条 bad case 的归因结果。"""
    case_id: str
    nature: str            # 第一层：性质
    reason: str            # 状态机给出的失败原因
    root_cause: str        # 第二层：根因
    fix: str               # 修复方向
    step_index: int = -1   # 第几步出的错，-1 表示不是执行失败而是目标未达成
    step_text: str = ""    # 出错那一步的原文

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "nature": self.nature,
            "reason": self.reason,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "step_index": self.step_index,
            "step_text": self.step_text,
        }


def attribute(case_id: str, run_result) -> Attribution:
    """给一条失败的评测结果做归因。

    run_result 是 simulator.run_plan 的返回值，里面带着每一步的执行结果。
    为什么要传入完整结果而不是只传失败原因字符串：
    第二层归因需要定位到「第几步、那一步原文是什么」，
    只给一个原因字符串是定位不了的。
    """
    # ① 找第一个没跑通的步骤。只取第一个是因为
    #    run_plan 默认遇错就停，后面根本没执行，谈不上归因
    bad_step = next((s for s in run_result.steps if not s.ok), None)

    if bad_step is not None:
        # ② 有失败步骤：原因是状态机给的枚举，定位到第几步、原文是什么
        reason = bad_step.reason.value
        idx, text = bad_step.index, bad_step.raw
    else:
        # ③ 没有失败步骤，说明每一步都合法但最后没达成目标。
        #    这是另一类问题 —— 不是「做不到」，是「做错了方向」
        reason = "目标未达成（动作均可执行）"
        idx, text = -1, ""

    # ④ 查归因表。查不到就用兜底条目，兜底里明写着「去补表」，
    #    这样漏登记的原因不会静默消失
    entry = TAXONOMY.get(reason, FALLBACK)
    return Attribution(
        case_id=case_id,
        nature=entry["nature"],
        reason=reason,
        root_cause=entry["root_cause"],
        fix=entry["fix"],
        step_index=idx,
        step_text=text,
    )


def summarize(attrs: list[Attribution]) -> dict:
    """汇总归因结果，输出三层结构。

    第一层按性质聚合，第二层列出每类原因的详情，第三层给修复方向。
    """
    # ① 第一层：按性质（物理常识 / 场景理解 / 输出格式 / 任务理解）分组
    by_nature: dict[str, list[Attribution]] = {}
    for a in attrs:
        by_nature.setdefault(a.nature, []).append(a)

    # ② 总数和占比。除零保护：一条 bad case 都没有时占比算 0
    total = len(attrs)
    out = {"total": total, "natures": {}, "reasons": {}}
    for nature, items in by_nature.items():
        out["natures"][nature] = {
            "count": len(items),
            "share": len(items) / total if total else 0,
        }

    # ③ 第二层：按具体失败原因分组
    reason_counter: dict[str, list[Attribution]] = {}
    for a in attrs:
        reason_counter.setdefault(a.reason, []).append(a)
    # ④ 按条数从多到少排。排最前的就是最该先修的那类
    for reason, items in sorted(reason_counter.items(), key=lambda kv: -len(kv[1])):
        # ⑤ 每条原因都带上归因表里的三项：性质、根因、修复方向。
        #    第三层就在这里 —— 报告里能直接给出「下一步该做什么」
        entry = TAXONOMY.get(reason, FALLBACK)
        out["reasons"][reason] = {
            "count": len(items),
            "share": len(items) / total if total else 0,
            "nature": entry["nature"],
            "root_cause": entry["root_cause"],
            "fix": entry["fix"],
            # ⑥ 附三个样例 id，方便直接翻原文看
            "samples": [i.case_id for i in items[:3]],
        }
    return out


def regression_record(case: dict, score_row: dict, attr: Attribution) -> dict:
    """把一条 bad case 转成回归集记录。

    回归集的作用：改动 prompt 或状态机之后重跑这批用例，
    看同类错误有没有减少。所以必须保留「预期行为」——
    也就是这条用例本应该达成目标，失败就是回归。
    """
    # ① 样本标识 + 当时用的模型，用来复现
    # ② 分层标签，回归时能按难度和任务类型分别看
    # ③ expectation 固定写「达成目标」。
    #    这是回归集的锚点：改完东西重跑，没达成就是回归了
    # ④ 归因结果
    # ⑤ 模型当时给的动作 + gold 动作，两边留着方便人工对比
    return {
        "case_id": case["id"],
        "model": score_row.get("model", ""),
        "difficulty": case.get("difficulty", ""),
        "task_type": case.get("task_type", ""),
        "task_desc": case.get("task_desc", ""),
        "expectation": "达成目标",
        "attribution": attr.as_dict(),
        "predicted_plan": score_row.get("plan", []),
        "gold_plan": case.get("gold_plan", []),
    }
