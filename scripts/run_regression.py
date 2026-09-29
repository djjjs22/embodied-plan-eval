#!/usr/bin/env python3
"""回归验证：改动之后，那些已知的 bad case 修好了没有。

================================================================================
这个脚本在闭环里的位置
================================================================================

    跑评测 → 归因导出回归集 → 改 prompt 或改状态机 → **跑回归验证** → 再跑评测

补上最后一环，评测才能驱动改进。否则你改完 prompt 只能凭感觉说「好像好点了」。

================================================================================
两种模式，成本差很多
================================================================================

replay（默认，零成本）
    用回归集里**已存的模型输出**重跑状态机。
    验证的是「状态机或归因逻辑的改动」，比如你把某条前置条件改松了/改严了。
    不调 API，几秒钟跑完。

rerun（需要 API key）
    重新调模型生成规划。
    验证的是「prompt 改动」，这才是真正的改进验证。
    315 条失败样本，成本大概几毛钱。

先跑 replay 确认管道通，再跑 rerun 验证真实改进。

================================================================================
判读标准
================================================================================

    修复率 = 从失败变成功的比例。越高说明改动有效。
    仍失败 = 没修好的。按原因分类，看是不是集中在你没处理的那一类上。

**注意一个陷阱**：如果你改的是状态机规则（不是 prompt），修复率变高可能是
你把判定改松了，而不是模型变好了。这种情况要回头核对 gold 通过率有没有掉。
判定一松，gold 通过率会立刻升高 —— 那是个警号，不是好消息。

================================================================================
文件里四个函数，按调用顺序读
================================================================================

    load_cases()        读评测集，按 id 建成字典，供随时反查
    load_regression()   读回归集，可按模型过滤
    build_prompt()      只有 rerun 模式用得上：重新组装 prompt 去问模型
    main()              串起上面三个，跑完出报告并决定退出码

replay 模式只用到前两个和 main，全程不联网。

用法:
    python scripts/run_regression.py                       # replay，零成本
    python scripts/run_regression.py --mode rerun --limit 50
    python scripts/run_regression.py --model MiniMax-M3
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
from evalkit.config import PROMPT_VERSION, load_models
from evalkit.models import LLMClient, PROMPT_SYSTEM, PROMPT_USER, extract_plan
from evalkit.report import ReportBuilder
from evalkit.simulator import run_plan

ROOT = Path(__file__).resolve().parent.parent


def load_cases() -> dict:
    """读评测集，建成 id -> 用例 的字典。

    进来：无参数，路径写死在 dataset/cases.jsonl。
    出去：{case_id: 用例字典}。

    为什么返回字典而不是列表：回归集里每条记录只存了 case_id，
    跑的时候要拿着 id 回评测集里找原始场景。用字典一次就能查到，
    用列表就得每次从头扫一遍 —— 300 多条差别不大，但字典写起来也短。
    """
    # ① 整文件读进来按行切。cases.jsonl 是一行一条样本，
    #    所以不能用 json.load（那要求整个文件是一个 JSON 数组）
    out = {}
    for line in (ROOT / "dataset" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        # ② 空行跳过。文件末尾那个换行会被 splitlines 切出一个空串
        if line.strip():
            c = json.loads(line)
            # ③ 以 id 当键存进去。后面全靠这个键反查
            out[c["id"]] = c
    return out


def load_regression(model: str = "") -> list[dict]:
    """读回归集，可按模型过滤。

    进来：model 为空表示全要，给了名字就只留这个模型的记录。
    出去：回归记录列表，每条含 case_id / predicted_plan / attribution。

    为什么过滤放在这里而不是调用处：--model 的语义是「我只关心这个模型的
    bad case」，越早过滤后面所有统计（成功率、原因分布）就越干净，
    不用在每处统计里再判一次模型名。
    """
    path = ROOT / "dataset" / "regression.jsonl"
    # ① 没有回归集就硬报错，并直接告诉他该跑哪个脚本。
    #    不能静默返回空列表 —— 空列表会让后面算出「0/0 全部修复」这种假好消息
    if not path.exists():
        raise FileNotFoundError("没有回归集，先跑 python scripts/build_regression.py")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            # ② 没传 --model 就全收；传了就只收这一个模型的。
            #    用 r.get("model") 而不是 r["model"]，老记录可能没这个字段
            if not model or r.get("model") == model:
                out.append(r)
    return out


def build_prompt(case: dict) -> str:
    """重新组装给模型的输入，只有 rerun 模式用得上。

    进来：一条评测用例。出去：填好的 prompt 字符串。

    这个函数和 run_eval.py 里的 build_prompt 一模一样，必须保持一致：
    rerun 的目的是复现「同样的输入下模型现在会不会答对」，
    只要 prompt 差一个字，修复率就不能再归因到 prompt 改动上了。
    """
    # ① 取场景摘要。只给真实存在的物体，不给 gold，也不给完成条件
    scene = describe_scene_for_prompt(case)
    # ② 往模板里填三个槽位。列表为空时填「（无）」而不是留空，
    #    空字符串会让模型以为我们漏给了信息
    return PROMPT_USER.format(
        task_desc=case["task_desc"],
        objects="、".join(scene["objects"]) or "（无）",
        receptacles="、".join(scene["receptacles"]) or "（无）",
    )


def main() -> int:
    """跑一轮回归验证，出报告，返回退出码。

    进来：命令行参数（--mode / --model / --limit / --workers / --gate）。
    出去：0 表示正常结束，1 表示有问题或门禁没过。

    为什么退出码默认不阻断见文件末尾的说明，那里是整个脚本最容易被误解的地方。
    """
    # ① 定义命令行参数。默认 replay（零成本），
    #    想验证 prompt 改动必须显式加 --mode rerun
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "rerun"], default="replay")
    ap.add_argument("--model", default="", help="只验证某个模型的回归集")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model-for-rerun", default="")
    ap.add_argument(
        "--gate", action="store_true",
        help="门禁模式：仍有失败用例时返回非零退出码，用于 CI 阻断",
    )
    args = ap.parse_args()

    # ② 先把两份数据读进来：评测集（按 id 查）和回归集（要跑的用例）
    cases = load_cases()
    records = load_regression(args.model)
    # ③ --limit 是调试用的闸：先跑前几十条看看通不通。
    #    没有它，rerun 模式下一次配错就要浪费全部调用额度
    if args.limit:
        records = records[: args.limit]
    # ④ 空集直接退出。不往下走是因为后面到处要除以总数，会炸
    if not records:
        print("回归集是空的")
        return 1

    # ⑤ 抬头打印这一轮的关键信息。带上 prompt 版本，
    #    否则过几天回看报告根本不知道这份结果对应哪版 prompt
    print(f"回归用例 {len(records)} 条 | 模式 {args.mode} | prompt {PROMPT_VERSION}")

    # ⑥ 结果统一攒成 (记录, 是否修复, 当前失败原因) 三元组。
    #    两种模式产出同样的结构，后面的统计代码才能共用一套
    results: list[tuple[dict, bool, str]] = []   # (记录, 是否修复, 当前失败原因)

    # ================================================================ replay
    if args.mode == "replay":
        # 用存的输出重跑状态机，零成本
        for rec in records:
            # ① 拿着 case_id 回评测集找原始用例。
            #    找不到就跳过 —— 评测集改过 id 是常事，不能因此中断整轮
            case = cases.get(rec["case_id"])
            if not case:
                continue
            # ② 用回归集里**已存的**模型输出重跑状态机。
            #    注意这里不再调模型，所以改动只能体现在状态机和归因逻辑上
            run = run_plan(rec["predicted_plan"], case["scene_objects"], case["pddl_params"])
            # ③ 取第一个失败步骤的原因当作「当前为什么还失败」。
            #    全 ok 却没达成目标时，说明是任务理解问题，
            #    用一句话兜底，不能让它显示成「ok」混进成功里
            reason = next((s.reason.value for s in run.steps if not s.ok), "目标未达成（动作均可执行）")
            # ④ goal_achieved 就是「修复了没」。原来这批全是失败样本
            results.append((rec, run.goal_achieved, reason))
    # ================================================================= rerun
    else:
        # 重新调模型，验证 prompt 改动
        # ① 没指定就用回归集第一条记录的模型名，
        #    这样默认复现的是当初产生这批 bad case 的那个模型
        target = args.model_for_rerun or (records[0].get("model") or "MiniMax-M3")
        specs = load_models()
        # ② 配置里没有这个模型就停。不然后面每条都失败，
        #    跑完才发现白花了一轮钱
        if target not in specs:
            print(f"配置里没有模型 {target}，现有：{list(specs)}")
            return 1
        client = LLMClient(specs[target])

        # ③ 单条处理函数，会被丢进线程池并发跑。
        #    返回 None 表示这条没拿到结果（用例缺失或 API 失败），
        #    和「跑出来还是失败」是两回事，必须分开
        def one(rec):
            # ① 回评测集找原始用例，找不到返回 None 让外层跳过
            case = cases.get(rec["case_id"])
            if not case:
                return rec, None, ""
            # ② 真正调模型。用和 run_eval 里完全一致的 system + user
            res = client.complete(system=PROMPT_SYSTEM, user=build_prompt(case))
            # ③ 调用失败（限流/网络/key 错）原样返回错误串，不参与统计。
            #    把 API 故障算成「模型没修好」是最常见的误判
            if res.error:
                return rec, None, res.error
            # ④ 从返回文本里抠出动作序列，再丢进状态机判定
            plan = extract_plan(res.text)
            run = run_plan(plan, case["scene_objects"], case["pddl_params"])
            reason = next((s.reason.value for s in run.steps if not s.ok), "目标未达成（动作均可执行）")
            return rec, run.goal_achieved, reason

        # ④ 并发跑。用 as_completed 而不是按提交顺序取：
        #    哪条先返回就先处理，一条卡住不会拖住后面已完成的
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(one, r) for r in records]
            for fut in as_completed(futs):
                rec, fixed, reason = fut.result()
                # ⑤ fixed 是 None 的那批直接丢掉。它们既不算修复也不算失败，
                #    算进去会同时污染分子和分母
                if fixed is None:
                    continue
                results.append((rec, fixed, reason))

    # ---------------------------------------------------------------- 统计
    # ⑦ 分母是「真正拿到结果的条数」。
    #    API 失败的已经在上面被丢掉了，不进分母 —— 否则网络抖动会拉低修复率
    total = len(results)
    if total == 0:
        print("没有可用结果（可能都调用失败了）")
        return 1

    # ⑧ 一刀切成两堆：修好的和还失败的。
    #    回归集原来全是失败样本，所以「成功」就等于「这条修好了」
    fixed = [(r, rs) for r, ok, rs in results if ok]
    still = [(r, rs) for r, ok, rs in results if not ok]
    print(f"\n修复 {len(fixed)}/{total} = {len(fixed)/total:.1%}")
    print(f"仍失败 {len(still)}")

    # ⑨ 仍失败的按**当前**原因分堆：看还剩下的错误是不是集中在某一类。
    #    集中说明改动没打到点上，分散说明是若干独立问题
    by_reason = defaultdict(int)
    for _, rs in still:
        by_reason[rs] += 1
    if by_reason:
        print("\n仍失败的原因分布:")
        for rs, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {rs:<26}{n:>5}")

    # ⑩ 再按**当初**的归因性质分堆，每堆记 [总数, 已修复数]。
    #    这是本脚本最有价值的一张表：它回答「哪类问题改好了、哪类没动」，
    #    只看总修复率的话，很可能是一类全修好、另一类原地踏步
    by_nature = defaultdict(lambda: [0, 0])   # nature -> [total, fixed]
    for rec, ok, _ in results:
        # ⑪ nature 是当初 build_regression 打上的标签（幻觉/顺序错/…）。
        #    老记录可能没有这个字段，兜底成「未分类」，不能让它 KeyError
        nat = rec.get("attribution", {}).get("nature", "未分类")
        by_nature[nat][0] += 1
        if ok:
            by_nature[nat][1] += 1

    # ---------------------------------------------------------------- 报告
    # ⑫ 报告副标题带上模式和 prompt 版本。
    #    少这两个信息，过一周回看这份 HTML 就不知道它验证的是哪次改动
    rb = ReportBuilder(
        "回归验证报告",
        f"模式 {args.mode} · prompt {PROMPT_VERSION} · 回归集 {len(records)} 条",
    )
    # ⑬ 四张概览卡片。刻意把「模式」也列出来，
    #    因为 replay 和 rerun 的修复率含义完全不同，混着看会得出错误结论
    rb.cards([
        ("回归用例", str(total), "已知的 bad case"),
        ("已修复", f"{len(fixed)}", f"{len(fixed)/total:.1%} 从失败变成功"),
        ("仍失败", str(len(still)), "需要继续处理"),
        ("模式", args.mode, "replay 零成本 / rerun 验证 prompt 改动"),
    ])

    # ⑭ 主表：按当初的归因性质看修复率。
    #    按用例数从多到少排，样本最多的那类通常就是要优先解决的
    rb.h2("按原始败因看修复情况")
    rb.table(
        ["原始性质", "用例数", "已修复", "修复率"],
        [[k, str(v[0]), str(v[1]), f"{v[1]/v[0]:.0%}"]
         for k, v in sorted(by_nature.items(), key=lambda kv: -kv[1][0])],
        numeric_cols={1, 2, 3},
    )

    # ⑮ 副表：还剩下的错误按原因分布。全部修好时这张表不会出现
    if by_reason:
        rb.h2("仍失败的原因分布")
        rb.table(
            ["失败原因", "数量"],
            [[k, str(v)] for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1])],
            numeric_cols={1},
        )

    # ⑯ 判读提示。两种模式说的不是一回事，所以分开写：
    rb.h2("判读提示")
    if args.mode == "replay":
        # ① replay 用的是旧输出，模型没变。
        #    所以修复率一变，只能是你改了判定规则 ——
        #    这时候必须回头核对 gold 通过率，防止是「尺子变松」造成的假上涨
        rb.note(
            "这一轮用的是已存的模型输出，验证的是状态机与归因逻辑。"
            "如果修复率有变化，说明判定规则改了 —— 这时候务必回去跑一遍 "
            "python scripts/verify_gold.py，确认 gold 通过率没有异常升高。"
            "判定放松会让通过率虚高，那是警号不是好消息。"
        )
    else:
        # ② rerun 才是真的重新问了模型，修复率可以归因到 prompt 改动上。
        #    提醒读者横向比较各类修复率，别只盯着总数
        rb.note(
            f"这一轮重新调用了模型，验证的是 prompt {PROMPT_VERSION} 的效果。"
            "如果某一类错误的修复率明显低于其他类，说明 prompt 的改动没覆盖到那类问题，"
            "回头对照 bad case 报告里的「修复方向」逐条检查。"
        )
    if still:
        # ③ 点名还剩下的最大一类，直接告诉读者下一步该去改什么
        top = max(by_reason.items(), key=lambda kv: kv[1])
        rb.note(
            f"仍失败最多的是「{top[0]}」，共 {top[1]} 条。"
            f"这一类需要重点处理，它的修复方向见 bad case 归因报告。"
        )
    else:
        rb.note("全部回归用例已通过。", kind="ok")

    # ⑰ 文件名带时间戳而不是固定名字：
    #    每次回归都留一份，方便前后对比，不会被下一次覆盖掉
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"regression_{stamp}.html"
    rb.save(out)
    print(f"\n报告: {out}")

    # 退出码逻辑。
    #
    # 默认永远返回 0：回归集里全是已知失败用例，replay 模式下必然全部失败，
    # 每次都返回非零的话根本没法接 CI，跑一次红一次就成了狼来了。
    #
    # 只有显式加 --gate 才用退出码阻断，那时候的语义是
    # 「我期望这批已经修好，没修好就拦下来」。
    if args.gate and still:
        print(f"\n门禁未通过：仍有 {len(still)} 条已知用例未修复")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
