#!/usr/bin/env python3
"""生成 bad case 归因报告与回归集。

================================================================================
这个脚本补的是闭环的最后一环
================================================================================

评测跑完只告诉你「错了 143 条」。这个脚本回答的是：
**错在哪一类、根因是什么、下一步该改什么、改完怎么验证。**

所以它产出两份东西：

    归因报告   给人看的，三层结构（性质 → 根因 → 修复方向）+ 典型样例
    回归集     给机器跑的，把失败样本沉淀下来，改动之后重跑验证

没有回归集，评测就只是个打分器：改完 prompt 你不知道变好了还是变差了。

================================================================================
为什么要把「第几步出错」单独输出
================================================================================

「手上已有物体」这个说法太笼统。定位到「第 3 步 PickupObject(Bread_1) ——
手上已有 Apple_1」才能指导行动：原来它没放下就想去拿下一个。

状态机本来就记录了每一步的结果（StepResult.index / raw / reason），
只是之前没往外输出。这个脚本把它利用起来。

================================================================================
文件里四个函数，按调用顺序读
================================================================================

    load_all_results()  把历次评测的 JSON 合并成一份，同名模型取最新的
    load_cases()        读评测集，按 id 建成字典
    analyze_model()     核心：对失败样本逐条重跑状态机并打归因标签
    main()              串起来，出报告 + 导出 dataset/regression.jsonl

导出的回归集由 scripts/run_regression.py 消费，两个脚本是一对。

用法:
    python scripts/build_regression.py                     # 全部模型
    python scripts/build_regression.py --model MiniMax-M3
    python scripts/build_regression.py --min-nature 3      # 只输出样本>=3类的
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.badcase import attribute, regression_record, summarize
from evalkit.report import ReportBuilder
from evalkit.simulator import run_plan

ROOT = Path(__file__).resolve().parent.parent


def load_all_results(path: Path | None) -> tuple[dict, Path]:
    """合并所有评测结果，同名模型以最新的为准。

    进来：path 为 None 表示自动找 reports/plan_*.json；
          给了具体路径就只读那一份。
    出去：(结果字典, 实际使用的那个文件路径)。

    为什么要合并而不是只读最新一份：全量跑太贵，实际经常是分批跑的
    （今天跑 A 模型、明天加个 B 模型），每批落一个 JSON。
    只取最新一份会把之前批次的模型整个丢掉，报告就不全了。

    为什么要返回用的那个路径：报告副标题要写明「结果来源 xxx.json」。
    读者得知道这份归因是基于哪次跑的，不然数字没法复现。
    """
    # ① 显式指定了结果文件就直接读它，不用管 reports 目录里还有别的
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8")), path
    # ② 按修改时间从旧到新排。后读的覆盖先读的，
    #    所以同名模型最终留下的是最新那次跑的结果
    files = sorted((ROOT / "reports").glob("plan_*.json"), key=lambda p: p.stat().st_mtime)
    # ③ 一份都没有就报错，并告诉他该跑哪个脚本。
    #    不静默返回空 —— 空结果会生成一份「没有失败样本」的假报告
    if not files:
        raise FileNotFoundError("没有找到评测结果，先跑 python scripts/run_eval.py")
    merged: dict = {}
    for f in files:
        # ④ 单份文件坏了（比如跑到一半被中断写坏了）就跳过。
        #    一份坏文件不该让整轮归因失败，剩下的还能用
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        # ⑤ 只认「模型名 -> 结果列表」这种结构。
        #    别的文件（比如回归报告）也放在 reports 下，形状对不上就别混进来
        if isinstance(d, dict):
            for k, v in d.items():
                merged[k] = v
    # ⑥ 第二个返回值用最后一份文件（最新），报告里要显示它
    return merged, files[-1]


def load_cases() -> dict:
    """读评测集，建成 id -> 用例 的字典。

    进来：无参数。出去：{case_id: 用例字典}。

    为什么要字典：下面 analyze_model 拿到的是结果记录，里面只有 case_id，
    要查原用例的场景物体和 pddl_params 才能重跑状态机。
    字典一次查到，不用每次遍历。
    """
    out = {}
    # ① 一行一条样本，所以按行读而不是 json.load
    for line in (ROOT / "dataset" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        # ② 空行跳过。末尾换行会切出一个空串
        if line.strip():
            c = json.loads(line)
            # ③ 以 id 当键存进去，供后面按 case_id 反查
            out[c["id"]] = c
    return out


def analyze_model(rows: list[dict], cases: dict) -> tuple[list, list]:
    """对一个模型的失败样本做归因。

    返回 (归因列表, 回归集记录列表)。

    注意这里要重跑状态机，因为评测结果里只存了第一个失败原因字符串，
    而第二层归因需要「第几步、那一步原文」这些细节。
    重跑不花钱（纯本地），只花一点时间。
    """
    attrs = []
    records = []
    for r in rows:
        # ① 只要失败样本。API 调用失败的（有 error）不算模型的错，
        #    已经成功的（goal_achieved）也不需要归因 ——
        #    回归集里混进成功样本，run_regression 就永远显示「已修复」
        if r.get("error") or r.get("goal_achieved"):
            continue
        # ② 拿着 case_id 回评测集找原用例。找不到就跳过，
        #    评测集改过 id 是常事，不该让一条对不上就中断整轮
        case = cases.get(r["case_id"])
        if not case:
            continue
        # 用模型输出的动作序列重跑一遍，拿到每一步的执行细节
        # ③ 重跑状态机。评测结果里只存了「第一个失败原因」这一个字符串，
        #    而我们要的是「第几步、那一步原文」，只能靠重跑拿
        run = run_plan(r["plan"], case["scene_objects"], case["pddl_params"])
        # ④ 打归因标签：这一步错在哪一层（性质/根因/修复方向）
        a = attribute(r["case_id"], run)
        attrs.append(a)
        # ⑤ 沉淀成回归集记录：把「预期行为 + 归因标签 + 模型输出 + 专家答案」
        #    打包存一条，将来 run_regression 就是拿这批去重跑
        records.append(regression_record(case, r, a))
    return attrs, records


def main() -> int:
    """逐个模型做归因，出报告，并把失败样本导出成回归集。

    进来：命令行参数（--results / --model / --min-samples）。
    出去：0 正常结束，1 表示没有可分析的模型。

    为什么把「分析」和「导出」放在一个脚本里：归因和回归集用的是同一次
    重跑状态机的结果。分成两个脚本就得重跑两遍，还可能因为版本不同
    导致报告和回归集对不上。
    """
    # ① 定义命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="", help="评测结果 JSON，默认合并全部")
    ap.add_argument("--model", default="", help="只分析指定模型")
    ap.add_argument("--min-samples", type=int, default=1, help="原因类别少于此样本数不写进报告")
    args = ap.parse_args()

    # ② 读评测结果 + 评测集。
    #    没传 --results 就合并 reports 下全部，传了就只认那一份
    all_results, results_path = load_all_results(Path(args.results) if args.results else None)
    cases = load_cases()

    # ③ 确定要分析哪些模型。给了 --model 就只跑它，否则全部都跑
    targets = [args.model] if args.model else list(all_results)
    # ④ 过滤掉结果里根本不存在的名字。
    #    手敲模型名打错是常事，留着会在下面 KeyError
    targets = [t for t in targets if t in all_results]
    if not targets:
        print(f"没有可分析的模型，现有：{list(all_results)}")
        return 1

    # ⑤ 建报告。副标题写明结果来源文件，
    #    否则过几天回看这份 HTML 不知道归因的是哪次跑的
    rb = ReportBuilder(
        "Bad case 归因报告",
        f"三层归因：性质 → 根因 → 修复方向 · 结果来源 {results_path.name}",
    )
    # ⑥ 所有模型的回归记录都攒到这里，最后一次性导出
    all_records = []

    # ⑦ 每个模型单独一节。模型的错误构成差别很大，
    #    混在一起统计会被 dominant 的那个模型掩盖掉
    for tgt in targets:
        rows = all_results[tgt]
        # ① 先剔掉 API 调用失败的样本。没有输出的样本没法归因，
        #    算进来只会让「失败构成」里多出一堆说不清的东西
        valid = [r for r in rows if not r.get("error")]
        # ② 核心一步：逐条重跑状态机 + 打归因标签，同时产出回归记录
        attrs, records = analyze_model(valid, cases)
        all_records.extend(records)

        # ③ 一个失败样本都没有（全对，或全是调用失败）→ 记一句好话就过。
        #    不能因为没失败就报错退出，那本身就是最好的结果
        if not attrs:
            rb.h2(f"{tgt}")
            rb.note("没有失败样本。", kind="ok")
            continue

        # ④ 汇总成性质分布和原因分布两张表
        s = summarize(attrs)
        print(f"\n=== {tgt} ===")
        print(f"失败 {s['total']} 条")
        for nature, info in sorted(s["natures"].items(), key=lambda kv: -kv[1]["count"]):
            print(f"  {nature:<10} {info['count']:>4} 条  {info['share']:>5.1%}")

        # -------- 第一层：性质分布 --------
        # ⑤ 按样本数从多到少排。排在最前的那类通常就是优先要改的
        rb.h2(f"{tgt}：失败构成")
        rb.table(
            ["性质", "数量", "占失败比例"],
            [[k, str(v["count"]), f"{v['share']:.1%}"]
             for k, v in sorted(s["natures"].items(), key=lambda kv: -kv[1]["count"])],
            numeric_cols={1, 2},
        )

        # 第二层：逐原因的根因与修复方向
        rb.h2(f"{tgt}：根因与修复方向")
        rows_out = []
        for reason, info in sorted(s["reasons"].items(), key=lambda kv: -kv[1]["count"]):
            # ⑥ 样本太少的类别不写进报告。
            #    一两条的类别往往是噪声，写进去会误导读者去追一个偶发问题
            if info["count"] < args.min_samples:
                continue
            rows_out.append([
                info["nature"],
                reason,
                str(info["count"]),
                info["root_cause"],
                info["fix"],
            ])
        rb.table(["性质", "失败原因", "数量", "根因", "修复方向"], rows_out)
        print("  （详细根因与修复方向见报告）")

        # 典型样例：给出错在第几步、那一步原文
        rb.h2(f"{tgt}：典型样例")
        shown = 0
        for a in attrs:
            # ⑦ step_index < 0 表示「动作都能执行，就是目标没达成」，
            #    没有具体的出错步骤可展示，跳过
            if a.step_index < 0:
                continue
            # ⑧ 一个样例一张两列表。
            #    把「第几步 + 那一步原文」印出来，读者才能对上号 ——
            #    光说「手上已有物体」是没法指导改动的
            rb.table(
                ["项", "内容"],
                [
                    ["用例", a.case_id],
                    ["出错步骤", f"第 {a.step_index + 1} 步：{a.step_text}"],
                    ["失败原因", a.reason],
                    ["根因", a.root_cause],
                    ["修复方向", a.fix],
                ],
            )
            shown += 1
            # ⑨ 最多展示 5 个。再多报告就翻不动了，
            #    要看全部的话去 regression.jsonl 里翻
            if shown >= 5:
                break
        # ⑩ 一个样例都没展示 = 全是「目标未达成」型失败，
        #    这句话要说清楚，否则读者会以为脚本坏了
        if shown == 0:
            rb.note("所有失败都是「动作可执行但目标未达成」，没有执行层面的错误。")

        # 第三层：闭环说明
        rb.h2(f"{tgt}：如何验证改进")
        rb.note(
            f"本次共沉淀 {len(records)} 条回归用例到 dataset/regression.jsonl。"
            "修改 prompt 或状态机之后，跑 python scripts/run_regression.py 复跑这批用例，"
            "看同类错误有没有减少。没有这一步，评测就只能打分，不能驱动改进。"
        )

    # ---------------------------------------------------------------- 导出回归集
    # ⑧ 回归集是给机器跑的，只有它有落盘价值 —— 报告只是给人看的一页 HTML
    if all_records:
        # ① 覆盖写而不是追加。
        #    追加会让上一轮的旧记录赖着不走，回归集会越跑越脏，
        #    run_regression 的修复率也就永远算不准了
        out = ROOT / "dataset" / "regression.jsonl"
        with out.open("w", encoding="utf-8") as f:
            # ② 一行一条 JSON。ensure_ascii=False 保住中文，
            #    不然归因标签全变成 \uXXXX，人就没法直接看了
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n回归集: {out}（{len(all_records)} 条）")
        rb.h2("回归集")
        rb.note(
            f"已导出 {len(all_records)} 条到 dataset/regression.jsonl。"
            "每条记录了：预期行为、归因标签、模型输出、专家标准答案。"
            f"其中归因分布已在上方按模型给出。"
        )

    # ⑨ 报告文件名带时间戳：每次归因都留一份，方便和改完之后的那次对比
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = ROOT / "reports" / f"badcase_{stamp}.html"
    rb.save(out_html)
    print(f"报告: {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
