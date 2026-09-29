#!/usr/bin/env python3
"""把一条样本从头到尾走一遍，看它在每一站长什么样。

看不懂整个项目的时候，先别读代码。跑这个脚本，
看同一条数据在七个地方分别是长什么样的，心里就有地图了。

用法:
    python scripts/trace_pipeline.py
    python scripts/trace_pipeline.py trial_T20190908_233657_231915
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import describe_scene_for_prompt, load_traj  # noqa: E402
from evalkit.simulator import build_initial_state, run_plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "dataset" / "cases.jsonl"
ALFRED_ROOT = ROOT.parent / "data" / "alfred" / "json_2.1.0"

DEFAULT_ID = "trial_T20190908_233657_231915"


def station(n, title, where):
    print()
    print(f"【第 {n} 站】{title}")
    print(f"    {where}")


def find_raw(case_id):
    """回原始目录里找这条样本的那个文件。"""
    for p in ALFRED_ROOT.rglob("traj_data.json"):
        if p.parent.name == case_id:
            return p
    return None


def main():
    case_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID

    case = None
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            if c["id"] == case_id:
                case = c
                break
    if case is None:
        print(f"cases.jsonl 里没有 {case_id}")
        return 1

    raw_path = find_raw(case_id)

    print("=" * 68)
    print(f"跟着一条样本走完全程：{case_id}")
    print("=" * 68)
    print("\n上半场：这条数据是怎么造出来的（只做一次）")
    print("下半场：这条数据是怎么变成判定结果的（每跑一次做一遍）")

    # ---------------------------------------------------------- 第 1 站
    station(1, "ALFRED 的原始数据，躺在硬盘上", "data/alfred/json_2.1.0/.../traj_data.json")
    if raw_path is None:
        print("    (原始文件没找到，跳过)")
    else:
        j = json.loads(raw_path.read_text(encoding="utf-8"))
        print(f"    文件: .../train/{raw_path.parent.parent.name}/{raw_path.parent.name}/")
        print(f"    这是个大字典，有 {len(j)} 个键，我们只用到 5 个:")
        print(f"      task_type          = {j.get('task_type')}")
        anns = (j.get("turk_annotations") or {}).get("anns") or []
        print(f"      task_desc          = {(anns[0]['task_desc'] if anns else '').strip()}")
        print(f"      plan.high_pddl     = {len((j.get('plan') or {}).get('high_pddl') or [])} 个动作")
        print(f"      scene.object_poses = {len((j.get('scene') or {}).get('object_poses') or [])} 条物体记录")
        print(f"      pddl_params        = {j.get('pddl_params')}")

    # ---------------------------------------------------------- 第 2 站
    station(2, "load_traj() 把它翻成程序里的一个对象", "evalkit/alfred.py 第 177 行")
    if raw_path is not None:
        pc = load_traj(raw_path)
        print(f"    输出一个 PlanCase，里面装着:")
        print(f"      id           = {pc.id}")
        print(f"      task_desc    = {pc.task_desc.strip()}")
        print(f"      task_type    = {pc.task_type}   difficulty = {pc.difficulty}")
        print(f"      gold_plan    = {pc.gold_plan}")
        print(f"      pddl_params  = {pc.pddl_params}")
        print(f"      scene_objects= {len(pc.scene_objects)} 个，第一个长这样:")
        print(f"                     {pc.scene_objects[0]}")

    # ---------------------------------------------------------- 第 3 站
    station(3, "抽样 315 条，写成一行一个 JSON 的文件", "scripts/import_alfred.py -> dataset/cases.jsonl")
    total = sum(1 for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"    {CASES.name} 一共 {total} 行，每行一条样本")
    print(f"    这一行的键: {list(case.keys())}")
    print(f"    前 120 个字符:")
    print(f"      {json.dumps(case, ensure_ascii=False)[:120]}...")

    print()
    print("-" * 68)
    print("下半场开始")

    # ---------------------------------------------------------- 第 4 站
    station(4, "状态机把场景读进来，变成一份「记账本」", "evalkit/simulator.py build_initial_state()")
    state = build_initial_state(case["scene_objects"])
    print(f"    输入: scene_objects（{len(case['scene_objects'])} 个物体）")
    print(f"    输出: 一个 WorldState，长这样:")
    print(f"      机器人位置 = {state.location}")
    print(f"      手上拿着   = {state.inventory}")
    print(f"      物体字典   = {len(state.objects)} 项，取两个看:")
    for o in list(state.objects.values())[:2]:
        print(f"        {o.obj_type:<16} 位置={o.parent}  已切片={o.sliced}")

    # ---------------------------------------------------------- 第 5 站
    station(5, "拿一串动作去跑，看记账本被改成什么样", "evalkit/simulator.py run_plan()")
    res = run_plan(
        lines=case["gold_plan"],
        scene_objects=case["scene_objects"],
        params=case["pddl_params"],
    )
    print(f"    输入: gold_plan（{len(case['gold_plan'])} 个动作）+ 完成条件")
    print(f"    输出: 一个 RunResult")
    print(f"      执行了     = {res.executed} 个动作")
    print(f"      违反前置   = {res.violations} 次")
    print(f"      目标达成   = {res.goal_achieved}")
    print(f"      逐项对照   = {res.goal_detail}")

    # ---------------------------------------------------------- 第 6 站
    station(6, "汇总成指标，写成报告", "evalkit/metrics.py + evalkit/report.py")
    print(f"    这一步的输入是「315 条样本各自的 RunResult」")
    print(f"    输出: 成功率、分层得分、bad case 列表 -> reports/ 下的 html")

    # ---------------------------------------------------------- 第 7 站
    station(7, "顺带看看：喂给模型的那份东西长什么样", "evalkit/alfred.py describe_scene_for_prompt()")
    scene = describe_scene_for_prompt(case)
    print(f"    任务描述: {case['task_desc'].strip()}")
    print(f"    场景物体: {scene['objects'][:10]} ...(共 {len(scene['objects'])} 个)")
    print(f"    ↑ 就这两样。gold_plan 和 pddl_params 不在这里面，模型看不到。")

    print()
    print("=" * 68)
    print("走完了。现在回去看代码，每一个文件对应哪一站，心里有数了。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
