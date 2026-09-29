#!/usr/bin/env python3
"""路径多样性分析。

================================================================================
要回答的问题
================================================================================

报告里有两个数字摆在一起很刺眼：

    任务成功率        54.6%
    序列精确匹配       6.3%

既然超过一半的样本都达成目标了，为什么路径和专家标注几乎完全不一样？

两种可能：
    甲：指标失灵 —— 判定或比对有问题，把对的判成了不同
    乙：解法本来就不唯一 —— 模型走了另一条同样合法的路

这两种情况的结论完全相反。甲说明评测不可信，乙说明"精确匹配"这个指标
本身就不该当主指标。**必须查清楚。**

================================================================================
方法
================================================================================

用现有评测结果就能查，不需要重新调模型：

1. 取出所有「模型达成目标」的样本（这些的路径已经过状态机验证，是合法解）
2. 和 gold 比对，看有多少条走的是不同的路
3. 比长度：模型更短 / 一样 / 更长

如果「成功但路径不同」占绝大多数，乙就成立。

================================================================================
一个必须警惕的陷阱
================================================================================

如果发现大量「模型路径比 gold 短」，先别急着下「模型找到了捷径」的结论。

要检查是不是模型**省略了 GotoLocation**。因为当前状态机里 GotoLocation
不做任何校验（只记位置），而只有 Clean/Heat/Cool 才要求站在对应器具前。
所以模型完全可以不写「走到微波炉前」就直接加热——只要它手持物体且
location 恰好对得上就行；而 Pickup/Put 根本不看 location。

那样的话「路径更短」不是找到捷径，是**状态机放水**。

所以脚本会把「省略移动」单独统计出来，避免得出假结论。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.metrics import normalize_plan
from evalkit.report import ReportBuilder

ROOT = Path(__file__).resolve().parent.parent


def load_all_results() -> dict:
    files = sorted((ROOT / "reports").glob("plan_*.json"), key=lambda p: p.stat().st_mtime)
    merged: dict = {}
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                merged[k] = v
    return merged


def load_cases() -> dict:
    out = {}
    for line in (ROOT / "dataset" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            out[c["id"]] = c
    return out


def action_name(step: str) -> str:
    return step.split("(")[0].strip().lower()


def count_action(plan: list[str], name: str) -> int:
    return sum(1 for a in plan if action_name(a) == name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="只分析某个模型，默认全部")
    args = ap.parse_args()

    all_results = load_all_results()
    cases = load_cases()
    targets = [args.model] if args.model else list(all_results)

    rb = ReportBuilder(
        "路径多样性分析",
        "回答：成功率高但精确匹配低，是指标失灵还是解法不唯一",
    )

    for tgt in targets:
        rows = [r for r in all_results[tgt] if not r.get("error")]
        succeeded = [r for r in rows if r["goal_achieved"]]
        if not succeeded:
            continue

        n_ok = len(succeeded)
        same = diff = 0
        shorter = longer = equal = 0
        fewer_move = 0          # 更短且移动步数比 gold 少

        for r in succeeded:
            case = cases.get(r["case_id"])
            if not case:
                continue
            p = normalize_plan(r["plan"])
            g = normalize_plan(case["gold_plan"])
            if p == g:
                same += 1
                continue
            diff += 1
            if len(p) < len(g):
                shorter += 1
                # 判断是不是靠减少移动动作变短的。
                #
                # 注意判据用「移动步数变少」而不是「完全不含移动」——
                # 后者只能抓到极端情况（实测只有 4 条），
                # 真实的「部分省略」有 80 条，用那个判据会漏掉，从而得出
                # 「模型找到捷径」的假结论。这是个差点踩进去的坑。
                if count_action(p, "gotolocation") < count_action(g, "gotolocation"):
                    fewer_move += 1
            elif len(p) > len(g):
                longer += 1
            else:
                equal += 1

        print(f"\n=== {tgt} ===")
        print(f"达成目标 {n_ok} 条")
        print(f"  与 gold 路径完全相同 {same} 条 ({same/n_ok:.1%})")
        print(f"  走了不同的路         {diff} 条 ({diff/n_ok:.1%})")
        print(f"    其中更短 {shorter} / 等长 {equal} / 更长 {longer}")
        print(f"    更短的那些里，移动步数比 gold 少的 {fewer_move} 条 ({fewer_move/max(shorter,1):.0%})")

        rb.h2(f"{tgt}：成功样本走了什么样的路")
        rb.cards([
            ("达成目标", str(n_ok), "已过状态机验证，都是合法解"),
            ("路径与 gold 相同", f"{same/n_ok:.1%}", f"{same} 条"),
            ("路径与 gold 不同", f"{diff/n_ok:.1%}", f"{diff} 条"),
            ("不同路径里更短的", f"{shorter/max(diff,1):.1%}", f"{shorter} 条"),
        ])

        rb.table(
            ["情况", "数量", "占成功样本"],
            [
                ["与 gold 完全相同", str(same), f"{same/n_ok:.1%}"],
                ["不同但等长", str(equal), f"{equal/n_ok:.1%}"],
                ["不同且更短", str(shorter), f"{shorter/n_ok:.1%}"],
                ["不同且更长", str(longer), f"{longer/n_ok:.1%}"],
            ],
            numeric_cols={1, 2},
        )

        # ---------------------------------------------------------- 判读
        rb.h2("判读")
        if diff / n_ok > 0.8:
            rb.note(
                f"成功样本里有 {diff/n_ok:.0%} 走的是和 gold 不同的路径。"
                "这说明**达成同一目标存在多条等价合法路径**，「序列精确匹配」低"
                "不是指标失灵，而是如实反映了「解法本来就不唯一」。"
                "所以它只能当辅助指标，不能用它判断模型好坏。",
                kind="ok",
            )
        else:
            rb.note(
                f"成功样本里只有 {diff/n_ok:.0%} 路径不同，"
                "多数成功案例和 gold 一致。那么精确匹配低主要来自失败样本，"
                "需要另外排查比对逻辑。"
            )

        if shorter > 0 and fewer_move / shorter > 0.8:
            rb.note(
                f"**关键发现**：更短的路径里有 {fewer_move}/{shorter} "
                f"({fewer_move/shorter:.0%}) 是靠**减少 GotoLocation** 变短的。"
                "当前状态机对移动动作不做校验（只有 Clean/Heat/Cool 要求站在"
                "对应器具前，Pickup/Put 完全不看位置），所以模型不写「走到某处」"
                "也能判成功。**这不是模型找到捷径，是状态机放水。**"
                "<br><br>"
                "已尝试给 Pickup/Put 补上位置校验，结果 gold 通过率从 98.7% 掉到 "
                "67.9%，被门槛拦住 —— 因为 ALFRED 的 GotoLocation 目标可以是"
                "**物体本身**（走到面包前）而不只是容器，按容器校验过严。"
                "正确修法需要先建对「物体 → 所在位置」的映射。"
                "已记为待改进项，见 docs/DATA_CARD.md。"
            )
        elif shorter > 0:
            rb.note(
                f"更短的路径有 {shorter} 条，且多数不是靠省略移动动作，"
                "说明模型确实找到了比专家标注更短的解法。"
                "这也侧面说明 **ALFRED 的 gold 不是最短路径**，"
                "以 gold 长度为基准的「路径效率」指标会失真。"
            )

    rb.h2("结论怎么用")
    rb.note(
        "如果「成功但路径不同」占多数，那么报告里的精确匹配就应该明确标注为"
        "参考项，主指标只用「目标达成」。同时这给 rubric 的「路径效率」维度"
        "提供了一个修正依据：以 gold 长度为基准会失真，"
        "更合理的基准是「成功样本的最短路径」而不是 gold。"
    )

    out = ROOT / "reports" / "path_diversity.html"
    rb.save(out)
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
