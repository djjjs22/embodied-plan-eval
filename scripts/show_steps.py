#!/usr/bin/env python3
"""一步一步看状态机怎么推演一条动作序列。

verify_gold.py 只告诉你「这条挂了」，脚本标题里的原因也只有一个词。
要看懂到底哪儿对不上，得把每一步执行前后的「记账本」翻出来对比。

这个脚本干的就是这件事：打印开局状态、每步之后哪一项变了、
最后拿记账本去对 pddl_params 的要求逐项划勾。

用法:
    python scripts/show_steps.py trial_T20190908_233657_231915
    python scripts/show_steps.py --failed 1      # 第 1 条 gold 跑不通的
    python scripts/show_steps.py --failed 2      # 第 2 条
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.simulator import (  # noqa: E402
    apply_action,
    build_initial_state,
    check_goal,
    parse_action_line,
    run_plan,
)

DATA = Path(__file__).resolve().parent.parent / "dataset" / "cases.jsonl"

NA = "(没记录)"
FIELD_NAMES = ["位置", "已清洁", "已加热", "已冷却", "已切片", "已开关"]


def load_cases():
    return [
        json.loads(l)
        for l in DATA.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def failed_ids():
    out = []
    for c in load_cases():
        res = run_plan(
            lines=c["gold_plan"],
            scene_objects=c["scene_objects"],
            params=c["pddl_params"],
        )
        if not (res.goal_achieved and res.violations == 0):
            out.append(c["id"])
    return out


def fmt(v):
    return NA if v is None else str(v)


def snake(state):
    """给当前状态拍张快照。"""
    return {
        "location": state.location,
        "inventory": state.inventory,
        "objs": {
            k: (v.parent, v.cleaned, v.heated, v.cooled, v.sliced, v.toggled)
            for k, v in state.objects.items()
        },
    }


def print_diff(before, after, state):
    """只打印发生变化的那些字段。"""
    if before["location"] != after["location"]:
        print(f"    机器人位置 : {fmt(before['location'])}  ->  {fmt(after['location'])}")
    if before["inventory"] != after["inventory"]:
        print(f"    手上拿着   : {fmt(before['inventory'])}  ->  {fmt(after['inventory'])}")

    for key, old in before["objs"].items():
        new = after["objs"].get(key)
        if new == old:
            continue
        obj = state.objects[key]
        for idx, (o, n) in enumerate(zip(old, new)):
            if o != n:
                if idx == 0:
                    print(f"    {obj.obj_type} 的{FIELD_NAMES[0]}: {fmt(o)}  ->  {fmt(n)}")
                else:
                    print(f"    {obj.obj_type} 的{FIELD_NAMES[idx]}: {o}  ->  {n}")


def show(case):
    gold = case["gold_plan"]
    params = case["pddl_params"]
    state = build_initial_state(case["scene_objects"])

    print("=" * 66)
    print(f"样本  {case['id']}")
    print(f"任务  {case['task_desc'].strip()}")
    print(f"类型  {case['task_type']}")
    print("=" * 66)

    print("\n【开局】状态机的记账本 —— 机器人和物体是两本独立的账")
    print(f"  机器人位置 : {NA}    <- 只有 GotoLocation 会改这一行")
    print(f"  手上拿着   : {NA}    <- 只有 PickupObject 会改这一行")
    print(f"  各物体位置 : 全是 {NA}  <- 只有 PutObject 会改这一行")
    print("  物体清单:")
    for o in list(state.objects.values())[:20]:
        print(f"    {o.obj_type:<18} 位置 = {fmt(o.parent)}")

    stopped = False
    for i, line in enumerate(gold):
        before = snake(state)
        action, args = parse_action_line(line)
        if action is None:
            print(f"\n第{i}步 {line}")
            print("    ✗ 解析不出来，中断")
            stopped = True
            break
        ok, reason, detail = apply_action(state, action, args)
        mark = "✓" if ok else "✗"
        print(f"\n第{i}步 {line}")
        print(f"    {mark} {detail if ok else reason.value}")
        if not ok:
            stopped = True
            break
        print_diff(before, snake(state), state)

    achieved, detail = check_goal(state, params)

    print("\n" + "=" * 66)
    print("【最后查账】任务到底做完了没有")
    print("=" * 66)
    print("\n要求来自 pddl_params（这份东西模型是看不到的）:")
    for k in ("object_target", "parent_target", "mrecep_target", "toggle_target", "object_sliced"):
        v = params.get(k)
        if v not in (None, "", False):
            print(f"    {k:<16} = {v}")

    print("\n逐项对照:")
    obj = state.find(params.get("object_target") or "")
    for k, v in detail.items():
        tag = "✓" if v else "✗"
        extra = ""
        if not v and k.startswith("放置在") and obj is not None:
            want = k[len("放置在"):]
            extra = f"（现在的位置是 {fmt(obj.parent)}，要求放在 {want}）"
        elif not v and k == "目标物体存在":
            extra = f"（场景里没有 {params.get('object_target')}）"
        print(f"    [{tag}] {k}{extra}")

    n_bad = sum(1 for v in detail.values() if not v)
    print()
    if stopped:
        print(f"结论: 动作在第 {len([s for s in gold])} 步之前就断了，后面的步骤根本没执行。")
        print(f"      上面账本里挂的 {n_bad} 项，是被中断连累的，不是独立的问题。")
    elif achieved:
        print("结论: 通过。全部要求都满足了。")
    else:
        print(f"结论: 没通过。{len(detail)} 项里挂了 {n_bad} 项，而且所有动作都执行成功了。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="?", help="样本 id")
    ap.add_argument("--failed", type=int, default=0, help="看第 N 条 gold 跑不通的样本")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"找不到 {DATA}")
        return 1

    cid = args.case_id
    if not cid:
        ids = failed_ids()
        if args.failed:
            if args.failed > len(ids):
                print(f"只有 {len(ids)} 条 gold 跑不通")
                return 1
            cid = ids[args.failed - 1]
        else:
            print("用法: python scripts/show_steps.py <样本id>")
            print("\ngold 跑不通的样本 id:")
            for n, i in enumerate(ids, 1):
                print(f"  --failed {n}   {i}")
            return 0

    for c in load_cases():
        if c["id"] == cid:
            show(c)
            return 0
    print(f"找不到样本 {cid}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
