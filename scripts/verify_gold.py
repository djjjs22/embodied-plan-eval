#!/usr/bin/env python3
"""Day 2 硬门槛：用 gold 动作序列验证符号状态机。

这一步过不去就不要往下走。gold 是专家标注的正确解法，
如果它都跑不通，说明 simulator.py 里的前置条件规则写错了，
后面所有模型评测分数全是废数据。

判据：gold 序列的成功率必须接近 100%。
低于 95% 就要回去修规则，不要急着跑模型。

用法:
    python scripts/verify_gold.py
    python scripts/verify_gold.py --show 10     # 打印失败样例细节
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import DIFFICULTY_ORDER
from evalkit.simulator import FailReason, run_plan


def load_cases(path: Path) -> list[dict]:
    # ① 一行一个 JSON。空行跳过，不然 json.loads 会抛异常中断整个验证
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path("dataset/cases.jsonl")))
    ap.add_argument("--show", type=int, default=5, help="打印多少条失败样例")
    args = ap.parse_args()

    # ① 数据集不存在就直接提示该跑哪条命令，别甩一堆看不懂的异常
    path = Path(args.dataset)
    if not path.exists():
        print(f"找不到 {path}，先跑 python scripts/import_alfred.py")
        return 1

    # ② 一次性读全部样本进来
    cases = load_cases(path)
    print(f"验证 {len(cases)} 条 gold 序列\n")

    # ③ 四个统计容器，各管一摊：
    #    ok             跑通的条数
    #    failed         失败的 (样本, 结果)，留着后面打印细节
    #    reason_counter 失败原因分布，这是修规则的线索
    #    per_type       每种任务类型的 [总数, 通过数]
    ok = 0
    failed: list[tuple[dict, object]] = []
    reason_counter: Counter = Counter()
    per_type: dict[str, list[int]] = {}

    # ④ 逐条把 gold 序列喂进状态机
    for c in cases:
        res = run_plan(
            lines=c["gold_plan"],
            scene_objects=c["scene_objects"],
            params=c["pddl_params"],
        )
        # ⑤ 登记到分类型统计里
        tt = c.get("task_type", "?")
        per_type.setdefault(tt, [0, 0])
        per_type[tt][0] += 1

        # ⑥ 通过的标准是两个条件同时满足：目标达成 且 零违反。
        #    只看前一个的话，一条「违规但碰巧达成」的序列会被放过；
        #    只看后一个的话，一条「没违规但也没做完」的序列会被放过
        if res.goal_achieved and res.violations == 0:
            ok += 1
            per_type[tt][1] += 1
        else:
            failed.append((c, res))
            # ⑦ 统计失败原因。只有真正跑到过、且不 ok 的那一步才有 reason
            for s in res.steps:
                if not s.ok:
                    reason_counter[s.reason.value] += 1

    # ⑧ 算总通过率。除零保护：样本数为 0 时rate 算 0，不崩
    total = len(cases)
    rate = ok / total if total else 0
    print(f"gold 完全跑通: {ok}/{total} = {rate:.1%}")

    # ⑨ 失败原因分布。这一栏是修规则的线索：
    #    哪一项多，就说明状态机的哪条规则该回炉
    if reason_counter:
        print("\n违反原因分布（这批是修规则的线索）:")
        for r, n in reason_counter.most_common():
            print(f"  {r:<20}{n:>6}")

    # ⑩ 按通过率从低到高排，最差的类型排最前，一眼看到问题在哪
    print("\n各任务类型通过情况:")
    for tt in sorted(per_type, key=lambda k: per_type[k][1] / max(per_type[k][0], 1)):
        tot, good = per_type[tt]
        print(f"  {tt:<38}{good:>4}/{tot:<4}{good/max(tot,1):>7.0%}")

    # ⑪ 打印失败样例细节。
    #    注意这里只打印第一个失败的步骤就 break ——
    #    所以「没有 ✗ 行」意味着所有动作都执行成功了，
    #    失败发生在最后的目标比对上。这是判读失败样例的第一刀
    if failed and args.show:
        print(f"\n失败样例（前 {min(args.show, len(failed))} 条）:")
        for c, res in failed[: args.show]:
            print(f"\n  [{c['id']}] {c['task_type']}")
            print(f"  任务: {c['task_desc'][:60]}")
            print(f"  gold: {c['gold_plan']}")
            for s in res.steps:
                if not s.ok:
                    print(f"  ✗ 第{s.index}步 {s.raw} -> {s.reason.value}: {s.detail}")
                    break
            print(f"  目标达成情况: {res.goal_detail}")

    print()
    # ⑫ 95% 是硬门槛。过不了不许往下走 ——
    #    状态机没验过就跑模型，出来的分数全是废数据
    if rate >= 0.95:
        print("门槛通过。状态机规则可用，可以开始跑模型评测。")
        return 0
    print(f"门槛未过（{rate:.1%} < 95%）。先修 simulator.py 里的前置条件规则，")
    print("修完重跑本脚本，别急着往下走。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
