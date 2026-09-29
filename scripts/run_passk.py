#!/usr/bin/env python3
"""pass^k 可靠性实验：同一个任务跑 k 次，全都成功才算通过。

================================================================================
为什么要这个指标
================================================================================

常规的成功率（pass@1）有个盲区：它只看「跑一次能不能对」。

但真实使用场景里用户只会问一次，没有重试机会。一个模型跑一次对、
跑三次只对一次，平均成功率还有 33%，可实际上你有三分之二的概率拿到错的规划。
**这种不稳定，pass@1 完全看不出来。**

pass^k 要求连续 k 次全部成功。pass@1 和 pass^k 的差距就是**不稳定性**：

| 情况 | pass@1 | pass^3 | 解读 |
| --- | --- | --- | --- |
| 稳定且强 | 80% | 78% | 差距小，可靠 |
| 靠运气 | 60% | 22% | 差距大，不可靠 |

τ-bench 用这个指标发现前沿 Agent 单次看着还行，连续八次掉到 25% 以下。

================================================================================
为什么要用 temperature > 0
================================================================================

temperature=0 时模型输出基本确定，跑 3 次得到 3 个相同答案，pass^k 等于 pass@1，
测不出任何东西。

所以这个脚本默认用 temperature=0.7 采样。这是和常规评测的关键区别：
**常规评测要的是模型的确定性表现，可靠性评测要的是它的分布。**

用法:
    python scripts/run_passk.py --k 3 --limit 50         # 先试通
    python scripts/run_passk.py --k 3                     # 全量（315×3 = 945 次调用）
    python scripts/run_passk.py --k 3 --model deepseek-v4-flash --temperature 1.0

================================================================================
在流水线里站在哪一环
================================================================================

    run_eval.py（常规评测，temperature=0，测确定性表现 → 出 pass@1）
        ↓ 同一个任务换个温度重复采样
    run_passk.py（本文件：测稳定性 → 出 pass^k）
        ↓
    run_rubric_check.py（主观评分与客观真值对账）

和 run_eval.py 的区别不在指标，而在采样方式：那边要的是模型
「最典型的一次表现」，这边要的是它的**输出分布**。
两边共用同一套 prompt 和同一个状态机，所以 pass@1 和 pass^k 可以直接相减。
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import describe_scene_for_prompt
from evalkit.config import load_models
from evalkit.metrics import wilson_interval
from evalkit.models import LLMClient, PROMPT_SYSTEM, PROMPT_USER, extract_plan
from evalkit.report import ReportBuilder
from evalkit.simulator import run_plan

ROOT = Path(__file__).resolve().parent.parent


def load_cases(limit: int = 0) -> list[dict]:
    """读 cases.jsonl。limit > 0 时只取前 limit 条。

    进来：limit，0 表示不截断
    出去：用例列表

    为什么留这个口子：本脚本的调用量是「用例数 × k」，全量上千次。
    先用小样本把流程和限流都试顺了再放开，能省下大量冤枉钱。
    """
    path = ROOT / "dataset" / "cases.jsonl"
    # ① 一行一条用例（jsonl），整读整切。几百条的量级没必要搞流式读取
    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # ② limit=0 表示「不截断」，这个约定全项目统一，别写成 cases[:0] 那种空列表
    return cases[:limit] if limit else cases


def build_prompt(case: dict) -> str:
    """把一条用例拼成给模型看的 prompt。

    进来：一条用例
    出去：填好的 prompt 字符串

    为什么复用 PROMPT_USER 而不另写一套：pass^k 必须和常规评测口径一致，
    否则两个 pass@1 对不上，「差距」就失去意义了。
    这里只填 task_desc / 物体 / 容器三项，不额外给任何提示。
    """
    # ① 把场景压成两串名字。原始 scene 结构太啰嗦，全塞进去会稀释任务描述
    scene = describe_scene_for_prompt(case)
    # ② 套模板。物体或容器为空时用「（无）」占位，
    #    留空白的话模型会自己脑补场景里的东西
    return PROMPT_USER.format(
        task_desc=case["task_desc"],
        objects="、".join(scene["objects"]) or "（无）",
        receptacles="、".join(scene["receptacles"]) or "（无）",
    )


def main() -> int:
    """主流程：每条用例采样 k 次 → 用状态机判每次成败 → 算 pass@1 与 pass^k → 出报告。"""
    # ① 命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MiniMax-M3")
    ap.add_argument("--k", type=int, default=3, help="跑几次，全成功才算通过")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="必须大于 0，否则每次输出一样，测不出稳定性")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="并发别开太大，容易触发限流")
    args = ap.parse_args()

    # ② 自保式拦截：temperature<=0 时模型输出基本确定，k 次采样拿到 k 份一样的答案，
    #    pass^k 会恒等于 pass@1。花了几百次调用却什么都没测出来，不如一开始就拦下
    if args.temperature <= 0:
        print("temperature 必须大于 0，否则每次输出相同，pass^k 恒等于 pass@1")
        return 1

    # ③ 模型不在配置里就退出，别等发请求才失败
    specs = load_models()
    if args.model not in specs:
        print(f"配置里没有 {args.model}，现有：{list(specs)}")
        return 1

    # ④ 加载用例，--limit 可以在这里就截断
    cases = load_cases(args.limit)
    # ⑤ 先把总调用量打出来。这是真实花钱的数字，让用户在跑之前心里有数
    print(f"模型 {args.model} | k={args.k} | temperature={args.temperature}")
    print(f"用例 {len(cases)} 条，总调用 {len(cases) * args.k} 次")

    client = LLMClient(specs[args.model])

    # 每条样本跑 k 次，记录每次是否达成
    # ⑥ outcomes: 用例 id → 这一条每次采样的成败列表。
    #    用 defaultdict 是因为并发返回顺序是乱的，没法按下标顺序回填
    outcomes: dict[str, list[bool]] = defaultdict(list)
    errors = 0

    # ⑦ 一次采样的完整流程：问一次 → 抠出动作序列 → 扔进状态机判是否达成。
    #    判定逻辑和常规评测完全一致，唯一的区别是带了 temperature
    def one(case: dict, run_idx: int):
        # ① 采样。temperature 从这里传进去，多次采样之间的差异就来自它
        res = client.complete(
            system=PROMPT_SYSTEM,
            user=build_prompt(case),
            temperature=args.temperature,
        )
        # ② 调用出错用 None 标记，绝不能记成 False ——
        #    网络失败不等于模型做错，混进去会把成功率白白算低
        if res.error:
            return case["id"], None
        # ③ 从回答里抠出动作序列，再用符号状态机跑一遍拿「是否达成目标」
        plan = extract_plan(res.text)
        run = run_plan(plan, case["scene_objects"], case["pddl_params"])
        return case["id"], run.goal_achieved

    # ⑧ 笛卡尔积展开成「用例 × k」的扁平任务表。
    #    摊平之后同一条用例的 k 次采样会分散到不同时刻，
    #    一次限流不至于把整条用例的所有采样全废掉
    tasks = [(c, i) for c in cases for i in range(args.k)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, c, i) for c, i in tasks]
        # ① as_completed：谁先跑完处理谁，不等提交顺序
        for fut in as_completed(futs):
            cid, ok = fut.result()
            # ② 调用失败的只累加错误数，不写进 outcomes ——
            #    少一次这条用例就凑不满 k，后面会被整体过滤掉
            if ok is None:
                # ① 单独计数是为了在报告里如实交代剔了多少，
                #    否则读者会以为这些样本压根没跑
                errors += 1
                continue
            # ③ 按用例 id 归堆。顺序无所谓，最后只看次数够不够 k
            outcomes[cid].append(ok)

    # 只保留跑满 k 次的样本，否则 pass^k 没法算
    # ⑨ 过滤掉没跑满 k 次的用例。pass^k 的定义是「k 次全成功」，
    #    少一次就无法判断这条到底算通过还是不通过，硬算只会把结论带偏
    complete = {cid: v for cid, v in outcomes.items() if len(v) == args.k}
    # ⑩ 一条都跑不满说明调用基本全挂了，后面的统计没有意义
    if not complete:
        print(f"没有跑满 {args.k} 次的样本（调用失败 {errors} 次）")
        return 1

    n = len(complete)
    # pass@1：所有单次结果的平均成功率
    # ⑪ 把 k 次采样摊平成「随机抽一次能对」的概率。
    #    这样算出来的口径和常规评测的 pass@1 一致，两者才可以直接相减
    flat = [ok for v in complete.values() for ok in v]
    pass1 = wilson_interval(sum(flat), len(flat))
    # pass^k：k 次全成功的样本比例
    # ⑫ 分母是样本数 n，不是调用次数 k×n —— 统计单位是「用例」不是「采样」
    all_ok = sum(1 for v in complete.values() if all(v))
    passk = wilson_interval(all_ok, n)
    # 不稳定性 = 差距
    # ⑬ 这个差值就是本实验要的答案：越大说明模型越靠运气
    gap = pass1.point - passk.point

    print(f"\n有效样本 {n} 条（调用失败 {errors} 次已剔除）")
    print(f"pass@1  {pass1}")
    print(f"pass^{args.k}  {passk}")
    print(f"不稳定性（差距）{gap:+.1%}")

    # 按难度分层看稳定性
    # ⑭ 分层的原因：稳定性问题往往集中在难题上。只看整体平均，
    #    「简单题很稳、难题全崩」这种模式会被平均掉，看不出来
    by_lv: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for c in cases:
        # ① 没跑满 k 次的用例在分层里也一并跳过，保持和整体同一口径
        if c["id"] not in complete:
            continue
        # ② 两层索引：难度 → 用例 id → 成败列表
        by_lv[c["difficulty"]][c["id"]] = complete[c["id"]]

    # ---------------------------------------------------------------- 报告
    # ⑮ 报告容器。副标题必须带上 temperature 和 k ——
    #    这两个参数一变结论就不可比，写在报告上防止日后误读
    rb = ReportBuilder(
        f"pass^{args.k} 可靠性报告",
        f"模型 {args.model} · temperature {args.temperature} · k={args.k}",
    )
    # ⑯ 四个核心数字并列成卡片。用 Wilson 置信区间而不是裸百分比，
    #    是因为样本量只有几百条，只报点估计会让人误以为这个数很准
    rb.cards([
        ("pass@1", f"{pass1.point:.1%}", f"95% CI {pass1.low:.0%}-{pass1.high:.0%}"),
        (f"pass^{args.k}", f"{passk.point:.1%}", f"95% CI {passk.low:.0%}-{passk.high:.0%}"),
        ("不稳定性", f"{gap:+.1%}", "两者差距，越大越不可靠"),
        ("有效样本", str(n), f"每条跑 {args.k} 次，失败 {errors} 次已剔除"),
    ])

    rb.h2("怎么读这两个数")
    rb.note(
        f"pass@1 是「跑一次能对」的概率，pass^{args.k} 是「连续 {args.k} 次都对」的概率。"
        "真实使用场景里用户只问一次，没有重试机会，所以 pass^k 更接近实际体验。"
        f"两者差 {gap:.1%}，这个差值就是模型的不稳定程度。"
    )
    # ⑰ 阈值判读：差距超过 20 个百分点就认定不可靠。
    #    这个阈值是经验值，用来把结论说成一句人话，别只丢个数字给读者
    if gap > 0.2:
        # ① 差距大意味着用户有一定概率直接拿到错的规划，
        #    所以建议落在「加重试或自检」这种可执行的兜底手段上
        rb.note(
            f"差距 {gap:.1%} 偏大：模型表现不稳定，同一任务换个采样就可能失败。"
            "这类模型在生产环境需要加重试或自检机制。"
        )
    else:
        # ① kind="ok" 标绿：稳定性过得去
        rb.note(f"差距 {gap:.1%}，稳定性尚可。", kind="ok")

    rb.h2("分层稳定性")
    rows = []
    # ⑱ 难度按 L1→L2→L3 固定顺序输出。用固定列表而不是遍历字典，
    #    是为了报表顺序稳定，不随数据里出现过的难度而变
    for lv in ["L1", "L2", "L3"]:
        # ① 这一层一条样本都没有就跳过（--limit 截断或该层全失败时会发生）
        if lv not in by_lv:
            continue
        d = by_lv[lv]
        # ② 算法和整体完全一致，只是换成这一层的子集，
        #    这样同一层的 pass@1 才能和顶部的整体 pass@1 对上
        flat_lv = [ok for v in d.values() for ok in v]
        p1 = wilson_interval(sum(flat_lv), len(flat_lv)) if flat_lv else None
        ak = sum(1 for v in d.values() if all(v))
        pk = wilson_interval(ak, len(d)) if d else None
        # ③ 两个区间都算得出来才写进表里，缺一个就整行不显示
        if p1 and pk:
            rows.append([
                lv, str(len(d)),
                f"{p1.point:.1%}", f"{pk.point:.1%}", f"{p1.point - pk.point:+.1%}",
            ])
    rb.table(["难度", "样本", "pass@1", f"pass^{args.k}", "差距"], rows, numeric_cols={1, 2, 3, 4})

    # ⑲ 时间戳命名，历史报告留着做前后对比 ——
    #    换了模型或调了 temperature 之后要能跟这次直接比
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"passk_{stamp}.html"
    rb.save(out)
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
