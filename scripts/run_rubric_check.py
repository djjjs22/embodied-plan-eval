#!/usr/bin/env python3
"""Rubric 校验：主观评分到底有没有用。

================================================================================
这一步在回答什么问题
================================================================================

rubric（评分量表）是评测体系里补主观维度用的。但它有个根本问题没人问过：
**打出来的分，和真实质量到底有没有关系？**

本项目有客观真值（状态机判定是否达成目标），所以能回答这个问题。

================================================================================
核心洞察：重合是冗余，增量才是价值
================================================================================

如果一个 rubric 维度和「目标是否达成」高度重合，那它是**冗余**的——
有了客观真值，这个维度根本不需要。

如果一个维度能反映客观判定**看不到**的东西（比如路径绕不绕、有没有冗余步骤），
那它才是**增量**，在没有客观真值的场景里才值得用。

所以本脚本不只看总分相关性，**逐维度看**：

| 维度 | 与客观判定的关系 | 判断 |
| --- | --- | --- |
| 目标覆盖 | 大概率重合 | 冗余（有真值时不需要） |
| 步骤必要性 | 客观判定看不到 | 增量 |
| 路径效率 | 部分能用编辑距离替代 | 半增量 |
| 可读性 | 客观判定完全看不到 | 增量 |

这个分析才是 rubric 迭代的正确依据：
不能用的维度删掉，有增量的维度保留并优化锚点。

================================================================================
用法
================================================================================

    python scripts/run_rubric_check.py --limit 30        # 先试通
    python scripts/run_rubric_check.py                    # 全量
    python scripts/run_rubric_check.py --model deepseek-v4-flash

================================================================================
在流水线里站在哪一环
================================================================================

    run_eval.py（客观判定，产出 reports/plan_*.json，里面有 goal_achieved）
        ↓ 拿它的 goal_achieved 当客观真值
    run_rubric_check.py（本文件：让 Judge 按 rubric 主观打分，再和真值对账）
        ↓ 输出"哪些维度冗余、哪些有增量"
    rubric/plan_quality_v1.md（按结论删维度、改锚点，并升 RUBRIC_VERSION）

本文件自己不生成规划、也不跑状态机，只消费上一环的结果，
是整条流水线里唯一一个「拿主观分数去对账客观真值」的环节。

"""

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import describe_scene_for_prompt
from evalkit.config import RUBRIC_VERSION, load_models, load_prompt
from evalkit.models import LLMClient
from evalkit.report import ReportBuilder

ROOT = Path(__file__).resolve().parent.parent

# 四个维度，与 rubric/plan_quality_v1.md 保持一致。
# 改这里必须同步改那份文档，并升 RUBRIC_VERSION。
DIMENSIONS = [
    ("steps_necessity", "步骤必要性"),
    ("path_efficiency", "路径效率"),
    ("goal_coverage", "目标覆盖"),
    ("readability", "可读性"),
]

# prompt 内容在 prompts/ 目录，四个维度与 rubric/plan_quality_v1.md 保持一致
RUBRIC_SYSTEM = load_prompt("rubric_system")
RUBRIC_USER = load_prompt("rubric_user")


def goal_text(params: dict) -> str:
    """把 pddl_params 里散落的目标字段拼成一句人话，交给 Judge 看。

    进来：一条用例的 pddl_params（一堆可能为空的字段）
    出去：一句中文目标描述，比如「操作对象：Apple_1；最终放置在：Fridge_1」

    为什么需要它：Judge 是个外部模型，看不懂结构化的 pddl_params。
    不把目标讲清楚，它打「目标覆盖」这一维全靠猜，分数自然不可信。
    """
    # ① 一句一句攒起来最后再拼。直接把 dict 打印给 Judge 的话，
    #    一堆 None 值和英文键名会混在里面，反而干扰它判断
    parts = []
    # ② 主目标物体，绝大多数任务都有；没有就不写这一句
    if params.get("object_target"):
        parts.append(f"操作对象：{params['object_target']}")
    # ③ 最终放置位置。ALFRED 数据里这个字段有两种叫法（mrecep / parent），
    #    两个都试一遍、谁有值用谁，不然会漏掉一整类任务的落点信息
    parent = params.get("mrecep_target") or params.get("parent_target")
    if parent:
        parts.append(f"最终放置在：{parent}")
    # ④ 开关类任务（开灯、关微波炉）才有，纯摆放任务没有这一项
    if params.get("toggle_target"):
        parts.append(f"需要开关：{params['toggle_target']}")
    # ⑤ 切片是硬约束：忘了切状态机直接判不达成。必须明写出来提醒 Judge
    if params.get("object_sliced"):
        parts.append("该物体需要被切片")
    # ⑥ 一个字段都没捞到时给个占位，避免 prompt 里留一段空白让 Judge 自由发挥
    return "；".join(parts) or "（未给出明确目标）"


def load_all_results(path: Path | None) -> tuple[dict, Path]:
    """合并所有评测结果。同名模型以最新的为准。

    进来：可选的一个结果文件路径；不给就去 reports/ 里捞全部 plan_*.json
    出去：(合并后的 {模型名: 样本列表}, 数据来自哪个文件)

    为什么要「以最新为准」：run_eval 每次跑都会新生成一个带时间戳的
    plan_*.json，同一个模型在多个文件里都可能出现。取旧的会把已被修掉的
    老问题又算进来，所以必须让时间最新的那份覆盖前面所有同名结果。
    """
    # ① 显式指定了文件就直接用它。想只针对某一次评测做校验时用这个口子
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8")), path
    # ② 没指定就把 reports/ 下所有 plan_*.json 按修改时间从旧到新排队。
    #    排序是「最新生效」的前提 —— 后面靠覆盖实现，顺序错了结论就反了
    files = sorted((ROOT / "reports").glob("plan_*.json"), key=lambda p: p.stat().st_mtime)
    # ③ 一份结果都没有就没法对账，直接把下一步该跑什么告诉用户，别让他猜
    if not files:
        raise FileNotFoundError("没有找到评测结果，先跑 python scripts/run_eval.py")
    merged = {}
    # ④ 逐个读。单个文件坏掉只跳过它，不能让一次损坏毁掉整个校验
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        # ⑤ 结果文件结构是 {模型名: [样本...]}，按模型名往里塞。
        #    后读到的同 key 会覆盖先读到的 —— 这就是「最新为准」的落地方式
        if isinstance(d, dict):
            for k, v in d.items():
                merged[k] = v
    # ⑥ 顺手把最后读的那个文件也交回去，报告里要标明数据出处
    return merged, files[-1]


def load_cases() -> dict:
    """把 cases.jsonl 读成 {用例 id: 用例}。

    为什么建索引而不留成列表：打分时手上只有结果里的 case_id，
    要靠它反查原始任务描述和场景。留成列表就得每次线性扫一遍，
    几百条 × 几百次调用，白烧时间。
    """
    # ① 一行一条用例（jsonl）。用例量只有几百条，整读整切就够了
    out = {}
    for line in (ROOT / "dataset" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        # ② 跳过空行，否则 json.loads 会抛异常，把整个读取打断
        if line.strip():
            c = json.loads(line)
            # ③ 用 id 当键：后面 ask_score 拿着 case_id 来查，O(1) 取回原始用例
            out[c["id"]] = c
    return out


def ask_score(client: LLMClient, case: dict, plan: list[str]) -> dict | None:
    """让 Judge 按 rubric 打分。解析失败返回 None。

    进来：Judge 客户端、原始用例、被测模型生成的动作序列
    出去：{维度 key: 1-5 的分数}，任何一环不对就返回 None

    为什么一律用 None 而不是抛异常或给默认分：一条打分坏了就坏了，
    剔除它只是少一个样本；但要是塞个 0 分或 3 分进去，
    相关性会被这些假数据带偏，最后得出错误结论还发现不了。
    """
    # ① 先把场景压成两串名字。Judge 得知道屋里有什么，
    #    才判断得了「路径绕不绕」「这一步有没有必要」
    scene = describe_scene_for_prompt(case)
    # ② 套 rubric_user 模板。plan 为空也要填「（空序列）」占位，
    #    留空白的话 Judge 会自己脑补一个规划再打分
    user = RUBRIC_USER.format(
        task_desc=case["task_desc"],
        objects="、".join(scene["objects"]) or "（无）",
        receptacles="、".join(scene["receptacles"]) or "（无）",
        goal=goal_text(case["pddl_params"]),
        plan="\n".join(plan) if plan else "（空序列）",
    )
    # ③ 调 Judge。response_json=True 是要求它直接吐 JSON，
    #    省掉从一大段解释文字里抠字段的麻烦和误差
    res = client.complete(system=RUBRIC_SYSTEM, user=user, response_json=True)
    # ④ 网络错/限流错：返回 None，让调用方计数剔除，不让它混进统计
    if res.error:
        return None
    # ⑤ 模型经常在 JSON 外面包一层 ```json 或前后加两句客套话，
    #    解析不出来同样作废这一条
    try:
        d = json.loads(res.text)
    except Exception:  # noqa: BLE001
        return None
    # 每个维度必须在 1-5 之间，否则这条判定不可用
    # ⑥ 兜底校验：四个维度必须齐全、且都落在量表的 1-5 内。
    #    少一个维度、或者给了 0 分/6 分这种越界值，说明 Judge 没按量表来，
    #    这种分数算进相关性会把结论带偏，宁可整条丢掉
    for key, _cn in DIMENSIONS:
        v = d.get(key)
        if not isinstance(v, (int, float)) or not (1 <= v <= 5):
            return None
    return d


def point_biserial(scores: list[float], flags: list[bool]) -> float:
    """点二列相关：连续分数与二分类变量的相关性。

    这里用来算 rubric 分数和「是否达成目标」的相关程度。

    进来：两个等长的列表 —— 连续分数、以及对应的 True/False 标记
    出去：-1 到 1 的相关系数，越接近 1 说明「分高的就是达成的」

    为什么用它而不是普通 pearson：其中一个变量是二分类的
    （达成 / 未达成），点二列相关就是为这种情况准备的专用公式。
    """
    # ① 只有一个点算不出相关性。给 0 而不是硬算，
    #    「看不出来」总好过编一个看起来很漂亮的数
    if len(scores) < 2:
        return 0.0
    # ② 按达成与否把分数劈成两组，后面比的就是这两组的均值差
    s1 = [s for s, f in zip(scores, flags) if f]
    s0 = [s for s, f in zip(scores, flags) if not f]
    # ③ 有一组是空的（全都达成、或全都失败），相关系数在数学上无定义，返回 0
    if not s1 or not s0:
        return 0.0
    m1, m0 = statistics.mean(s1), statistics.mean(s0)
    # ④ 用总体标准差 pstdev（不是样本标准差 stdev）：这批样本就是全部，
    #    不是从更大的总体里抽样出来的，不该再去做无偏修正
    sd = statistics.pstdev(scores)
    # ⑤ 所有人分数一模一样 → 分母为 0，同样无定义，返回 0
    if sd == 0:
        return 0.0
    # ⑥ p / q 是两组各占多少比例，用它把「均值差」换算成标准的相关系数尺度，
    #    这样不同达成率的模型之间才可比
    p = len(s1) / len(scores)
    q = 1 - p
    return (m1 - m0) / sd * math.sqrt(p * q)


def cohens_d(a: list[float], b: list[float]) -> float:
    """效应量：两组均值差除以合并标准差。比只看差值更可靠。

    进来：两组分数（达成组 / 未达成组）
    出去：d 值，0.5 算中等、0.8 算大，负号表示方向相反

    为什么不能只看均值差：同样是差 1 分，组内波动很小和组内波动很大，
    含义完全不同。除以标准差就是把这个「波动」归一化掉，
    这才是「这两组到底分得开不开」的真正度量。
    """
    # ① 少于两个点算不出方差，直接给 0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    # ② 合并标准差：两组样本量不一样（达成组通常比失败组多），
    #    不能把两个方差简单平均，得按自由度 (n-1) 加权，这才是标准的 pooled SD
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    # ③ pooled 为 0 说明两组内部毫无波动，除零会炸，返回 0 表示区分不出来
    return (ma - mb) / pooled if pooled else 0.0


def main() -> int:
    """主流程：加载客观真值 → 让 Judge 逐条打分 → 算相关性 → 判定冗余/增量 → 出报告。"""
    # ① 命令行参数。--limit 和 --workers 的存在是为了「先试通再全量」：
    #    打分要真实花钱，直接全量跑错一次代价不小
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="deepseek-v4-flash", help="打分用的模型")
    ap.add_argument("--model", default="", help="给哪个被测模型的输出打分，默认全部")
    ap.add_argument("--results", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # ② 读模型配置。注意 Judge 和被测模型是分开的两个角色：
    #    被测模型出规划，Judge 只负责打分，不能让模型给自己打分
    specs = load_models()
    # ③ Judge 不在配置里就立刻退出，别等发了几十次请求才报错
    if args.judge not in specs:
        print(f"配置里没有 {args.judge}，现有：{list(specs)}")
        return 1

    # ④ 客观真值和原始用例一起加载。本脚本不生成规划、也不跑状态机，
    #    只消费上一环 run_eval 已经判好的 goal_achieved
    all_results, results_path = load_all_results(Path(args.results) if args.results else None)
    cases = load_cases()
    # ⑤ --model 支持逗号分隔指定多个被测模型；一个都不给就把结果里所有模型都过一遍
    targets = [t.strip() for t in args.model.split(",") if t.strip()] or list(all_results)

    # ⑥ 建 Judge 客户端
    judge = LLMClient(specs[args.judge])
    print(f"打分模型 {args.judge} | rubric 版本 {RUBRIC_VERSION} | 被测 {targets}")

    # ⑦ 报告容器。副标题特意写明真值来自符号状态机，
    #    免得看报告的人以为这些分数是模型自己判的
    rb = ReportBuilder(
        "Rubric 校验：主观评分与客观真值的关系",
        f"打分模型 {args.judge} · rubric {RUBRIC_VERSION} · 客观真值来自符号状态机",
    )

    # ⑧ 每个被测模型独立出一套分析，模型之间互不干扰
    for tgt in targets:
        # ① 只留没出错的样本：error 的那条根本没有 plan，无从打分
        rows = [r for r in all_results[tgt] if not r.get("error")]
        # ② --limit 截断，试通用
        if args.limit:
            rows = rows[: args.limit]
        # ③ 一条可用样本都没有就跳过，别在报告里留一张空表
        if not rows:
            continue

        # ④ scored 存 (样本, 分数) 配对，failed 记作废条数。
        #    必须成对存 —— 后面算相关性时要拿分数去对这条样本的真值，
        #    一旦错位，所有结论都是错的
        scored: list[tuple[dict, dict]] = []
        failed = 0

        # ⑤ 把「一条样本 → (样本, 分数)」包成闭包，方便直接丢进线程池
        def one(r):
            return r, ask_score(judge, cases[r["case_id"]], r["plan"])

        # ⑥ 并发打分。一次打分就是一次独立的 HTTP 请求、彼此无依赖，
        #    串行跑耗时不可接受；并发度交给 --workers 控制以防限流
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(one, r) for r in rows]
            # ① as_completed：谁先返回就先处理谁，不等提交顺序
            for fut in as_completed(futs):
                r, sc = fut.result()
                # ② 打分作废的只累加计数，绝不塞进 scored ——
                #    它连分数都没有，进去了只会污染相关性
                if sc is None:
                    # ① 失败数单独记着，报告里要写出来，
                    #    否则读者不知道这些样本是被剔掉了还是压根没跑
                    failed += 1
                else:
                    scored.append((r, sc))

        # ⑦ 全部作废就没得算了，跳过这个模型
        n = len(scored)
        if n == 0:
            print(f"  {tgt}: 没有可打分的样本")
            continue

        # ⑧ flags 是客观真值（达成/未达成），totals 是四个维度的平均分。
        #    两条列表下标一一对应，后面所有相关性和效应量都建立在这个对应关系上
        flags = [bool(r["goal_achieved"]) for r, _ in scored]
        totals = [sum(sc[k] for k, _ in DIMENSIONS) / len(DIMENSIONS) for _, sc in scored]

        # ⑨ 总分与真值的相关性：这个数越接近 1，说明 rubric 总分≈客观判定的复刻
        r_total = point_biserial(totals, flags)
        # ⑩ 拆出达成组/未达成组的总分，既用来算效应量，也直接展示给读者看
        ok_n, bad_n = sum(flags), len(flags) - sum(flags)
        ok_t = [t for t, f in zip(totals, flags) if f]
        bad_t = [t for t, f in zip(totals, flags) if not f]

        print(f"\n{tgt}: {n} 条（打分失败 {failed}）")
        print(f"  总分相关性 r={r_total:.3f}  达成组均分 {statistics.mean(ok_t):.2f} / "
              f"未达成组 {statistics.mean(bad_t):.2f}")

        rb.h2(f"被测模型：{tgt}")
        rb.cards([
            ("样本", str(n), f"打分失败 {failed} 条已剔除"),
            ("总分相关", f"{r_total:.3f}", "rubric 总分与是否达成目标的相关"),
            ("达成组均分", f"{statistics.mean(ok_t):.2f}", f"n={ok_n}"),
            ("未达成组均分", f"{statistics.mean(bad_t):.2f}", f"n={bad_n}"),
        ])

        # 逐维度分析：这个是重点。
        # 先把每个维度的统计量算出来存好，后面判读直接复用，
        # 不要在列表推导里重算一遍 —— 那种写法又慢又容易错。
        rows_out = []
        dim_stats = []
        # ⑪ 逐维度跑一遍。同一个模型下四个维度的表现可以完全不同，
        #    只有拆开看才知道该删哪个、该留哪个
        for key, cn in DIMENSIONS:
            # ① 取出这个维度在所有样本上的分数
            vals = [sc[key] for _, sc in scored]
            # ② 它与客观判定的相关性：高就说明它只是在复述判定结果
            r = point_biserial(vals, flags)
            # ③ 按达成/未达成分组，算效应量看这两组分得开不开
            ok_v = [v for v, f in zip(vals, flags) if f]
            bad_v = [v for v, f in zip(vals, flags) if not f]
            d = cohens_d(ok_v, bad_v)
            # ④ 存一份，最后汇总「哪些冗余 / 哪些有增量」时还要再用一次
            dim_stats.append({"cn": cn, "r": r, "d": d})

            # 判读：与判定高度相关 = 重合（冗余）；
            # 相关性低但效应量大 = 客观判定看不到的增量信息
            # ⑤ 两个指标交叉判读，是本脚本最核心的一段逻辑
            if r >= 0.5:
                # ① 和客观判定高度重合：既然已经有真值了，这个维度就是多余的
                judge_txt = "与客观判定重合，冗余"
            elif abs(d) >= 0.5:
                # ① 相关性不高、但达成组和未达成组确实分得开 ——
                #    说明它量的是真值看不见的东西（绕路、废话步骤），这才是增量
                judge_txt = "能区分好坏，且不与判定重合 → 增量"
            else:
                # ① 既重合不上又分不开：维度本身可能没问题，是锚点写得太糊
                judge_txt = "区分力弱，需要改锚点"
            rows_out.append([
                cn,
                f"{statistics.mean(vals):.2f}",
                f"{statistics.mean(ok_v):.2f} / {statistics.mean(bad_v):.2f}",
                f"{r:.3f}",
                f"{d:+.2f}",
                judge_txt,
            ])

        rb.h2("逐维度判别力")
        rb.table(
            ["维度", "全体均分", "达成/未达成", "与判定相关", "效应量 d", "判断"],
            rows_out,
        )

        # ⑫ 汇总判定：相关性高 = 冗余；相关性不高但效应量大 = 增量
        redundant = [s["cn"] for s in dim_stats if s["r"] >= 0.5]
        incremental = [s["cn"] for s in dim_stats if s["r"] < 0.5 and abs(s["d"]) >= 0.5]

        # ⑬ 有冗余维度就点名
        if redundant:
            # ① 冗余不等于没用，而是「在有真值时没必要花这个钱」，
            #    所以措辞落在「不需要引入主观评分」上
            rb.note(
                "这些维度与客观判定高度重合：" + "、".join(redundant) + "。"
                "有客观真值时它们是冗余的，不需要引入主观评分。"
            )
        # ⑭ 有增量维度就点名，这条才是 rubric 活下来的理由
        if incremental:
            # ① kind="ok" 标绿：这是好消息，说明至少有一部分是白赚的信息
            rb.note(
                "这些维度提供了客观判定看不到的信息：" + "、".join(incremental) + "。"
                "这才是 rubric 的价值所在——在没有客观真值的场景里可以保留使用。",
                kind="ok",
            )
        # ⑮ 一个增量都没有：整套 rubric 在有真值的前提下就是多余的，得重做
        if not incremental:
            # ① 这个结论很硬，但正是它才让「要不要继续投 rubric」有据可依
            rb.note(
                "没有维度表现出增量价值。说明这套 rubric 在已有客观真值的前提下是多余的，"
                "需要重新设计维度，或者把锚点写得更具体。"
            )

    # ⑨ 结论段：把「这份校验到底该怎么用」讲清楚，
    #    免得别人只看到一堆相关系数却不知道下一步该干什么
    rb.h2("结论")
    rb.note(
        "这份校验回答的是：主观评分能不能替代客观判定。"
        "如果总分相关性高，说明 rubric 能在没有真值时凑合用；"
        "如果只有部分维度有增量，说明应该只保留那几个维度，其余删掉。"
        "rubric 的迭代依据是这里的数据，不是拍脑袋加维度。"
    )

    # ⑩ 时间戳命名，不覆盖历史报告 ——
    #    改完 rubric 再跑一次要能跟这次的结论对比，才知道改得对不对
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"rubric_check_{stamp}.html"
    rb.save(out)
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
