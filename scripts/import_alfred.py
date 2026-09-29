#!/usr/bin/env python3
"""从 ALFRED 抽取评测样本。

================================================================================
这一步在做什么
================================================================================

ALFRED 有近 8000 条轨迹，全用太慢也太贵。这个文件负责按任务类型
分层抽样，抽出一个规模合适、分布合理的评测集。

================================================================================
为什么要按任务类型分层，而不是随机抽
================================================================================

ALFRED 的 7 种任务类型天然难度不同：
    pick_and_place_simple           拿起放下，最简单
    pick_clean_then_place_in_recep  多一步清洗
    pick_two_obj_and_place          两个物体，要管理状态
如果随机抽，样本会集中在数量多的类型上，难度分布就歪了。
最后算出来的总分会被简单题稀释，看不出模型在难题上到底行不行。

分层抽样保证每种类型都有代表，这样报告里才能按类型分别出分。

================================================================================
用法
================================================================================

    python scripts/import_alfred.py --root ../data/alfred/json_2.1.0 --per-type 45
    python scripts/import_alfred.py --root ../data/alfred/json_2.1.0 --split valid_unseen

--split valid_unseen 只抽「场景在训练中没出现过」的子集，
适合做泛化测试：模型在见过的场景上表现好、在没见过的场景上崩，
这个落差本身就是个值得报告的发现。
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import DIFFICULTY_BY_TASK_TYPE, iter_traj_files, load_traj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ALFRED 解压目录，内含 traj_data.json")
    ap.add_argument("--per-type", type=int, default=50, help="每种任务类型采多少条")
    ap.add_argument("--split", default="", help="只要某个子集，如 valid_unseen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-scan", type=int, default=0, help="最多扫多少条，调试用")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"目录不存在: {root}")
        return 1

    # ------------------------------------------------------------ 扫描分组
    # 按 task_type 分组存放，为后面的分层采样做准备。
    # 解析失败的样本直接跳过：一条数据坏了不该中断整个流程，
    # 但如果坏得太多就要留意（可能是数据版本不对）。
    # ① 按任务类型分组存放，为后面的分层采样做准备。
    #    用 defaultdict 省掉「键不存在先建空 list」的样板代码
    by_type: dict[str, list] = defaultdict(list)
    scanned = 0
    for p in iter_traj_files(root, limit=args.limit_scan or None):
        scanned += 1
        try:
            # ② 解析单条轨迹。可能失败（文件损坏、版本不对）
            case = load_traj(p)
        except Exception:  # noqa: BLE001
            # ③ 解析失败就跳过，一条数据坏了不该中断整个流程。
            #    但如果坏得太多就要留意 —— 可能是数据版本不对
            continue
        # ④ 指定了子集就只要这一个（如 valid_unseen）
        if args.split and case.split != args.split:
            continue
        # ⑤ 三条过滤条件，缺任何一样这条样本都没法用：
        #   没有 gold 序列 -> 没有标准答案
        #   没有场景物体   -> 状态机没法初始化
        #   没有任务描述   -> 没法给模型出题
        if not case.gold_plan:
            continue
        if not case.scene_objects:
            continue
        if not case.task_desc:
            continue
        # ⑥ 按任务类型归类
        by_type[case.task_type].append((p, case))

    print(f"扫描 {scanned} 条，可用 {sum(len(v) for v in by_type.values())} 条")
    print("\n各类型可用数量:")
    for t in sorted(by_type):
        print(f"  {t:<38}{len(by_type[t]):>6}  难度 {DIFFICULTY_BY_TASK_TYPE.get(t, '?')}")

    # ------------------------------------------------------------ 分层采样
    # 每种类型独立随机抽 per_type 条。
    # 固定 seed 是为了可复现：同一条命令两次跑出同一个评测集，
    # 否则换一批样本分数就变，没法判断是模型变了还是数据变了。
    # ① 固定 seed 的随机数生成器。seed 固定是为了可复现：
    #    同一条命令两次跑出同一个评测集，
    #    否则换一批样本分数就变，没法判断是模型变了还是数据变了
    rng = random.Random(args.seed)
    picked = []
    # ② 每种类型独立随机抽 per_type 条。
    #    按类型排序再遍历，保证抽样顺序也是确定的
    for t in sorted(by_type):
        pool = by_type[t][:]
        rng.shuffle(pool)
        picked.extend(pool[: args.per_type])

    # ③ 再整体打乱一遍，避免同类型的样本都排在一起
    #   （评测不依赖顺序，但抽样看样本时更方便）
    rng.shuffle(picked)
    print(f"\n采样 {len(picked)} 条")

    # ------------------------------------------------------------ 写盘
    # ① 输出路径。dataset/ 不存在就建
    out = Path("dataset/cases.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # ② 一行一个 JSON（jsonl 格式）。
    #    用 w 而不是 a —— 每次导入都是完整重来，不追加，
    #    否则跑两次就会出现 630 条重复样本
    with out.open("w", encoding="utf-8") as f:
        for _, case in picked:
            # ③ 九个字段全写进去。
            #    顺序和字段集要固定，后面所有脚本都按这个结构读
            f.write(
                json.dumps(
                    {
                        "id": case.id,
                        "task_desc": case.task_desc,
                        "task_type": case.task_type,
                        "difficulty": case.difficulty,
                        "split": case.split,
                        "dimensions": case.dimensions,
                        "gold_plan": case.gold_plan,
                        "pddl_params": case.pddl_params,
                        "scene_objects": case.scene_objects,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"写入 {out}")
    # ④ 顺手报一下难度分布，让人当场就能看出分层是不是歪了
    levels: dict[str, int] = defaultdict(int)
    for _, c in picked:
        levels[c.difficulty] += 1
    for lv in sorted(levels):
        print(f"  {lv}: {levels[lv]}")

    # 提醒下一步。这个顺序不能省：状态机没验证过就跑模型，
    # 出来的分数没有意义。
    print("\n下一步（必做）: python scripts/verify_gold.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
