#!/usr/bin/env python3
"""数据集统计，用于生成数据卡。

================================================================================
这个脚本在整条流水线里排第一
================================================================================

    导入数据 → **统计数据集（本脚本）** → 跑评测 → 归因 → 回归验证
                ↑
         先说清「我测的是什么」，后面的分数才有意义

数据卡不能用「覆盖了多种任务类型」这种空话，要用数字说清楚
覆盖了什么、没覆盖什么、有哪些已知偏差。

面试官问「你这个数据集凭什么能测出规划能力」时，
答案就在这份统计里：难度怎么分的、有没有长序列、目标条件有几种。

================================================================================
输出分六块
================================================================================

    难度分层        L1/L2/L3 各多少，避免被简单题拉高平均分
    任务类型        覆盖哪些 ALFRED 任务
    能力维度        每条样本打了哪些维度标签
    场景覆盖        有多少种物体/容器，每个场景几个物体
    序列特征        gold 平均几步，含 NoOp 的有多少
    已知偏差        物体名带哈希后缀的比例（这是本项目最大的数据坑）

最后打印一份 sha256，用来证明「评测集没被偷偷改过」。

================================================================================
两个函数
================================================================================

    load_cases()   读 dataset/cases.jsonl，一行一条
    main()         全部统计逻辑，纯打印不落盘

用法:
    python scripts/dataset_stats.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def load_cases() -> list[dict]:
    """读评测集，返回用例列表。

    进来：无参数，路径写死在 dataset/cases.jsonl。
    出去：用例字典的列表，顺序和文件里一致。

    这里返回列表而不是字典，和另外两个脚本不一样 ——
    因为统计只关心「一共多少条、各维度多少个」，
    不需要按 id 反查，列表就够用。
    """
    path = ROOT / "dataset" / "cases.jsonl"
    # ① 一行一条样本，所以按行 json.loads 而不是 json.load。
    #    if l.strip() 过滤掉末尾换行切出来的空行
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    """把数据集的全部统计打印一遍，供抄进数据卡。

    进来：无参数。出去：0。

    为什么只打印不落盘：这份统计是给人抄进数据卡的，不是给机器读的。
    要存档直接重定向到文件即可，没必要再引一层输出格式。
    """
    cases = load_cases()
    # ① 总条数是后面所有百分比的分母，先算出来。
    #    除零保护也在这里：空数据集的话后面的占比全都会炸
    n = len(cases)
    print(f"总条数 {n}\n")

    # ---------------------------------------------------------------- 分层
    # ② 难度分层。这是最重要的一个数 —— 没有它，
    #    「成功率 53%」说不清是题太难还是模型太差
    print("=== 难度分层 ===")
    # ① 按 difficulty 字段计数。用 sorted 而不是按数量排，
    #    因为 L1/L2/L3 本身有顺序，按数量排会打乱这个顺序
    lv = Counter(c["difficulty"] for c in cases)
    for k in sorted(lv):
        print(f"  {k}: {lv[k]:>4}  {lv[k]/n:>5.1%}")

    # ③ 任务类型：看覆盖了多少种 ALFRED 任务
    print("\n=== 任务类型 ===")
    # ① 用 most_common() 按数量从多到少排。
    #    类型名是无序的，按多少排最能看出「是不是集中在某几种」
    tt = Counter(c["task_type"] for c in cases)
    for k, v in tt.most_common():
        print(f"  {k:<40}{v:>4}")

    # ④ 能力维度：一条样本可以有多个标签，所以是扁平化后再计数。
    #    各项加起来会大于总条数，这是正常的
    print("\n=== 能力维度 ===")
    # ① 双重 for 把每条的维度列表摊平。
    #    用 c.get("dimensions", []) 兜底，老样本可能没打标签
    dim = Counter(d for c in cases for d in c.get("dimensions", []))
    for k, v in dim.most_common():
        print(f"  {k:<24}{v:>4}")

    # ⑤ 子集划分（train/val/test 之类），看看有没有划分、划得匀不匀
    print("\n=== 子集 ===")
    # ① 用 get + 默认空串：没标 split 的样本统一归到「(未标注)」，
    #    不能让它 KeyError，也不能让它悄悄消失
    sp = Counter(c.get("split", "") for c in cases)
    for k, v in sp.most_common():
        print(f"  {k or '(未标注)'}: {v:>4}")

    # ---------------------------------------------------------------- 覆盖度
    # ⑥ 场景里到底有多少种东西。三个计数器：
    #    obj_types  每种物体出现过多少次
    #    receps     每个容器被引用多少次
    #    obj_counts 每个场景有几个不同物体（用来看场景复杂度）
    obj_types: Counter = Counter()
    receps: Counter = Counter()
    obj_counts = []
    for c in cases:
        # ① seen 用来给「这个场景有几种不同物体」去重。
        #    同一个场景可能有 3 个 Apple，但类型只算 1 种
        seen = set()
        for o in c.get("scene_objects", []):
            # ② 优先取 objectType（Apple），没有就退回 objectId（Apple_1）。
            #    统计「种类」必须用类型名，用 id 的话每个苹果都算一种
            t = o.get("objectType") or o.get("objectId")
            if t:
                seen.add(t)
                obj_types[t] += 1
            # ③ 顺手记一下这个物体待在哪些容器里。
            #    注意这里大概率是空的，原因见下面那段说明
            for r in o.get("parentReceptacles") or []:
                receps[r] += 1
        # ④ 这个场景的物体种类数，记进列表后面算平均/极值
        obj_counts.append(len(seen))

    print(f"\n=== 场景覆盖 ===")
    print(f"  不同物体类型 {len(obj_types)} 种")
    print(f"  出现最多的 8 种: {[t for t, _ in obj_types.most_common(8)]}")
    # ① 空数据集时 obj_counts 是空列表，min/max 会抛异常，必须先判
    if obj_counts:
        print(f"  每个场景物体数 平均 {sum(obj_counts)/len(obj_counts):.1f}, "
              f"最少 {min(obj_counts)}, 最多 {max(obj_counts)}")

    # 容器与器具改从 gold 动作里统计。
    #
    # 为什么不用 parentReceptacles：ALFRED 的 object_poses 里根本没有这个字段，
    # 容器信息是靠从 gold 动作参数反推补全的（见 alfred.py 的 _augment_scene）。
    # 直接统计那个字段会得到 0 —— 我第一版就是这么写的，白跑一遍。
    import re as _re

    used_places: Counter = Counter()
    # ① 换个思路：gold 动作里出现过的位置/容器，才是评测真正要用到的。
    #    「场景里有什么」不重要，「gold 会用到什么」才决定模型该知道什么
    for c in cases:
        for a in c["gold_plan"]:
            # ② 只认两种带位置/容器参数的动作。
            #    GotoLocation(x) 的 x 是位置，PutObject(obj, x) 的第二个是容器
            m = _re.match(r"(GotoLocation|PutObject)\((.*)\)", a)
            if not m:
                continue
            # ③ 括号里按逗号切成参数，去掉空格和空串
            args = [x.strip() for x in m.group(2).split(",") if x.strip()]
            if m.group(1) == "GotoLocation" and args:
                used_places[args[0]] += 1
            elif m.group(1) == "PutObject" and len(args) > 1:
                used_places[args[1]] += 1
    print(f"\n  实际被 gold 引用的位置与容器 {len(used_places)} 种")
    print(f"  最高频的 10 种: {[p for p, _ in used_places.most_common(10)]}")

    # ---------------------------------------------------------------- 序列特征
    # ⑦ gold 序列有多长。太长模型容易忘前文，太短测不出规划能力
    lens = [len(c["gold_plan"]) for c in cases]
    # ① NoOp 是 ALFRED gold 里的占位动作。
    #    统计它有多少，是因为归一化时要把它们剔掉再比对，
    #    不然编辑距离会被白白拉大
    noop = sum(1 for c in cases if any("noop" in a.lower() for a in c["gold_plan"]))
    print(f"\n=== gold 序列特征 ===")
    print(f"  平均长度 {sum(lens)/len(lens):.1f} 步（最短 {min(lens)}，最长 {max(lens)}）")
    print(f"  含 NoOp 占位符的 {noop} 条  {noop/n:.1%}")

    # 目标条件类型
    # ⑧ 目标条件有几种，决定了状态机要判哪几类终局。
    #    同一条样本可能同时满足多个条件，所以下面用 if 而不是 elif
    goal_kinds = Counter()
    for c in cases:
        # ① 用 get 兜底：没有 pddl_params 的样本当成空字典，别炸
        p = c.get("pddl_params") or {}
        if p.get("object_sliced"):
            goal_kinds["需要切片"] += 1
        if p.get("toggle_target"):
            goal_kinds["需要开关"] += 1
        if p.get("mrecep_target"):
            goal_kinds["经由可移动容器"] += 1
        # ② 「直接放置」要排除掉经由可移动容器的情况，
        #    否则一条样本会被算两次，各项占比加起来超过 100%
        if p.get("parent_target") and not p.get("mrecep_target"):
            goal_kinds["直接放置"] += 1
    print(f"\n=== 目标条件构成 ===")
    for k, v in goal_kinds.most_common():
        print(f"  {k:<20}{v:>4}  {v/n:>5.1%}")

    # ---------------------------------------------------------------- 已知偏差
    # ⑨ 这一块是数据卡里最该写的：主动交代数据集的毛病
    print(f"\n=== 已知偏差与数据质量 ===")
    # 物体名带哈希后缀的比例
    hashed = 0
    for c in cases:
        for o in c.get("scene_objects", []):
            # ① 判断规则：名字里有下划线，且最后一段长度 >= 6，
            #    那就是 ALFRED 的哈希后缀（Cloth_2476119a 里的 2476119a）
            oid = o.get("objectId") or ""
            if "_" in oid and len(oid.split("_")[-1]) >= 6:
                hashed += 1
                # ② 一个场景只要有一个这样的名字就算一条。
                #    break 是为了按「样本」计数而不是按「物体」计数
                break
    print(f"  场景物体名带哈希后缀的样本 {hashed} 条  {hashed/n:.1%}")
    print(f"  （这类名字需要剥离后缀才能和 gold 参数匹配）")

    # ⑩ 打印评测集指纹。写进数据卡之后，别人就能验证
    #    「你跑评测时用的集子和这份数据卡是同一份」
    print(f"\n=== 评测集 hash（防污染记录用）===")
    import hashlib
    # ① 读原始字节再算哈希。用 read_bytes 而不是 read_text，
    #    编码差异会改变字节、从而改变哈希值
    raw = (ROOT / "dataset" / "cases.jsonl").read_bytes()
    # ② 只取前 32 位：够用了，全 64 位印在终端里又长又没必要
    print(f"  sha256: {hashlib.sha256(raw).hexdigest()[:32]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
