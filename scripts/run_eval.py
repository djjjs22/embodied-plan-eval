#!/usr/bin/env python3
"""跑一轮具身任务规划评测并出报告。

================================================================================
这是整个项目的主流程，把前几个部件串起来
================================================================================

    评测集(cases.jsonl)
        -> 组装 prompt（问题 + 场景物体）
        -> 调模型 API
        -> 从返回文本里抠出动作序列
        -> 丢进符号状态机执行
        -> 得到每条样本的评价
        -> 汇总成指标 + 分层报告

对应方案里的「自动化 Pipeline」这一环。

================================================================================
用法
================================================================================

    python scripts/run_eval.py --limit 10 --workers 2     # 先跑通链路
    python scripts/run_eval.py --models MiniMax-M3        # 全量

强烈建议先用 --limit 10 跑通。全量 315 条如果配置有问题（比如 key 填错、
端点不对），会一口气浪费掉全部调用额度，而 10 条能立刻暴露问题。

================================================================================
两个设计要点
================================================================================

1. 缓存。每条样本的调用结果按 (模型, prompt 版本, 样本 id) 缓存。
   中断后重跑不会重复花钱，改 prompt 后必须升 PROMPT_VERSION 才会失效。
   这是可复现和成本控制的基础。

2. 并发。用线程池而不是逐个跑。315 条串行动辄十几分钟，
   并发后一两分钟。workers 别开太大，容易触发限流。
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import describe_scene_for_prompt
from evalkit.config import CACHE_DIR, PROMPT_VERSION, load_models
from evalkit.metrics import (
    Interval,
    normalize_plan,
    normalized_edit_distance,
    plan_exact_equal,
    wilson_interval,
)
from evalkit.models import LLMClient, PROMPT_SYSTEM, PROMPT_USER, extract_plan
from evalkit.report import ReportBuilder
from evalkit.simulator import FailReason, run_plan

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PlanScore:
    """一条样本的完整评价结果。

    字段分成四组，对应指标体系的四层：
        判定      goal_achieved、violations、first_fail_reason
        相似度    exact_match、edit_distance
        分层      task_type、difficulty
        效率      latency_ms、tokens

    保留 raw_output 是为了排查：分数不对的时候能回头看模型到底输出了什么。
    截断到 400 字符是防止结果文件过大。
    """
    case_id: str
    model: str
    task_type: str
    difficulty: str
    raw_output: str
    plan: list[str]
    goal_achieved: bool
    violations: int
    first_fail_reason: str
    edit_distance: float
    exact_match: bool
    latency_ms: int
    tokens: int
    error: str | None = None


def build_prompt(case: dict) -> str:
    """组装给模型的输入。

    三段内容缺一不可：
        任务描述       模型要知道做什么
        场景中的物体   模型要知道有什么可用
        可用容器器具   模型要知道能往哪放

    第三块很容易被忽略。不给容器列表的话，模型只能靠常识猜场景里
    有没有微波炉，猜错了不是它规划能力差，是我们没给足信息。
    """
    # ① 取出场景摘要。只给真实存在的物体，不给 gold，也不给完成条件
    scene = describe_scene_for_prompt(case)
    # ② 往 prompt 模板里填三个槽位。
    #    列表为空时填「（无）」而不是留空 —— 空字符串会让模型以为漏给了
    return PROMPT_USER.format(
        task_desc=case["task_desc"],
        objects="、".join(scene["objects"]) or "（无）",
        receptacles="、".join(scene["receptacles"]) or "（无）",
    )


class Runner:
    """单个模型的评测执行器，负责并发调用与结果缓存。"""

    def __init__(self, spec, max_workers: int, use_cache: bool):
        # ① 记下模型配置、并发数、是否用缓存
        self.spec = spec
        self.max_workers = max_workers
        self.use_cache = use_cache
        # ② 缓存目录不存在就建
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # ③ 缓存文件名里混入 prompt 版本的哈希。
        #    这样升了 PROMPT_VERSION，文件名就变，旧缓存自然读不到 ——
        #    不需要额外的失效逻辑，也不会出现「改了 prompt 却还用旧结果」
        self._cache_path = CACHE_DIR / (
            # ④ 模型名里的斜杠换成下划线，不然会当成路径分隔符
            spec.name.replace("/", "_") + "_"
            + hashlib.md5(f"{spec.name}|{PROMPT_VERSION}".encode()).hexdigest()[:10]
            + ".jsonl"
        )
        # ⑤ 启动时一次性把已有缓存读进内存
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        # ① 关了缓存、或者缓存文件还不存在，就返回空字典
        if not (self.use_cache and self._cache_path.exists()):
            return {}
        # ② 一行一个 JSON，用 case_id 当键。空行跳过
        out = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["case_id"]] = rec
        return out

    def _append(self, rec: dict) -> None:
        """追加一条缓存。只缓存成功的调用，失败的不缓存，
        否则一次网络抖动会被永久记住。"""
        # ① 以追加模式打开，写完一行就关。
        #    用 append 而不是整体重写，是为了中断后已跑的不丢
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run(self, cases: list[dict]) -> tuple[list[PlanScore], int, int]:
        """跑一批样本，返回 (评价列表, 缓存命中数, API 失败数)。"""
        # ① 建一个模型客户端
        client = LLMClient(self.spec)
        # ② 结果字典，键是 case_id
        results: dict[str, dict] = {}

        # ③ 单条样本的处理函数，会被丢进线程池并发执行
        def one(case: dict) -> dict:
            # ④ 命中缓存就直接返回，不再花钱。
            #    注意 copy 一份，不然 _hit 标记会写回缓存本体
            if case["id"] in self._cache:
                r = dict(self._cache[case["id"]])
                r["_hit"] = True
                return r
            # ⑤ 真正调 API
            res = client.complete(system=PROMPT_SYSTEM, user=build_prompt(case))
            rec = {
                "case_id": case["id"],
                "raw_output": res.text,
                "latency_ms": res.latency_ms,
                "tokens": res.prompt_tokens + res.completion_tokens,
                "error": res.error,
            }
            # ⑥ 只缓存成功的。失败的不缓存，
            #    否则一次网络抖动会被永久记住，以后永远跑不出真实结果
            if not res.error:
                self._append(rec)
            rec["_hit"] = False
            return rec

        # -------- 并发调用 --------
        # ⑦ 用 as_completed 而不是按顺序取结果：哪条先返回就先处理，
        #    一条卡住不会拖住后面已经完成的。
        #    workers 别开太大，容易触发限流
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(one, c): c for c in cases}
            for fut in as_completed(futs):
                rec = fut.result()
                results[rec["case_id"]] = rec

        # ⑧ 统计缓存命中数和 API 失败数。
        #    失败数高说明配置有问题，得停下来查而不是硬跑完
        hits = sum(1 for r in results.values() if r.get("_hit"))
        errors = sum(1 for r in results.values() if r.get("error"))

        # -------- 逐条判定 --------
        # ⑨ 这一步不调 API，纯本地执行，所以很快。
        #    这也是「判定和生成分离」的好处：改判定逻辑不用重新花钱跑模型
        scores = []
        for c in cases:
            rec = results[c["id"]]
            # ⑩ 从模型输出里抠出动作序列
            plan = extract_plan(rec.get("raw_output", ""))
            # ⑪ 丢进状态机跑，得到客观判定
            run = run_plan(plan, c["scene_objects"], c["pddl_params"])
            # ⑫ 记第一个失败原因，用于后面的归因分布。
            #    全 ok 时记 OK，表示「不是执行失败而是目标没达成」
            first_reason = next(
                (s.reason.value for s in run.steps if not s.ok), FailReason.OK.value
            )
            scores.append(
                PlanScore(
                    case_id=c["id"],
                    model=self.spec.name,
                    task_type=c.get("task_type", ""),
                    difficulty=c.get("difficulty", ""),
                    raw_output=rec.get("raw_output", "")[:400],
                    plan=plan,
                    goal_achieved=run.goal_achieved,
                    violations=run.violations,
                    first_fail_reason=first_reason,
                    # 编辑距离也在归一化后的序列上算，
                    # 否则 gold 里的 NoOp 会平白多算一步，把差异放大
                    edit_distance=round(
                        normalized_edit_distance(
                            normalize_plan(plan), normalize_plan(c["gold_plan"])
                        ),
                        3,
                    ),
                    # 归一化后再比，否则大小写和 NoOp 会让匹配率恒为 0
                    exact_match=plan_exact_equal(plan, c["gold_plan"]),
                    latency_ms=rec.get("latency_ms", 0),
                    tokens=rec.get("tokens", 0),
                    error=rec.get("error"),
                )
            )
        return scores, hits, errors


def _rate(scores: list[PlanScore], pred) -> Interval:
    """按条件统计比例，返回带置信区间的 Interval。

    所有百分比都走这个函数，保证没有一个数字是光秃秃的点估计。
    """
    n = len(scores)
    return wilson_interval(sum(1 for s in scores if pred(s)), n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset" / "cases.jsonl"))
    ap.add_argument("--models", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"找不到 {path}，先跑 python scripts/import_alfred.py")
        return 1

    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]

    specs = load_models()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()] or list(specs)
    missing = [m for m in model_names if m not in specs]
    if missing:
        print(f"配置里没有: {missing}")
        return 1

    print(f"用例 {len(cases)} 条，模型 {len(model_names)} 个，prompt {PROMPT_VERSION}")

    # ------------------------------------------------------------ 跑评测
    all_scores: dict[str, list[PlanScore]] = {}
    meta = {}
    for name in model_names:
        runner = Runner(specs[name], args.workers, not args.no_cache)
        scores, hits, errors = runner.run(cases)
        all_scores[name] = scores
        # 有效样本 = 真的拿到模型输出的那些。
        #
        # 为什么必须剔除 API 失败：失败的样本没有输出，不是模型的错。
        # 把它算进分母等于让网络问题拉低模型分数。
        # 这是实测踩到的坑：79 条限流失败没剔除，成功率被压到 40%，
        # 剔除后是 53.4%，差了 13 个百分点 —— 面试官一定会问这个。
        valid = [s for s in scores if not s.error]
        meta[name] = {
            "cache_hits": hits,
            "api_errors": errors,
            "valid": len(valid),
            "total": len(scores),
        }
        succ = _rate(valid, lambda s: s.goal_achieved)
        viol = _rate(valid, lambda s: s.violations > 0)
        print(f"  {name:<16} 有效 {len(valid)}/{len(scores)} | 成功率 {succ} | 违反率 {viol}")

    # ------------------------------------------------------------ 存原始结果
    # 存 JSON 而不是只存 HTML 报告：报告是给人看的，
    # 原始结果是给后续分析用的（归因、对照、重跑对比都要读它）。
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (ROOT / "reports").mkdir(exist_ok=True)
    out_json = ROOT / "reports" / f"plan_{stamp}.json"
    out_json.write_text(
        json.dumps({m: [asdict(s) for s in ss] for m, ss in all_scores.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------ 生成报告
    rb = ReportBuilder(
        "具身任务规划评测报告",
        f"符号状态机执行作客观真值 · prompt {PROMPT_VERSION}",
    )

    # 概览卡片：四个核心数字，每个都带置信区间。
    # 分母统一用有效样本，不是全量样本。
    valid_scores = {m: [s for s in rows if not s.error] for m, rows in all_scores.items()}
    for name in model_names:
        sc = valid_scores[name]
        succ = _rate(sc, lambda s: s.goal_achieved)
        viol = _rate(sc, lambda s: s.violations > 0)
        em = _rate(sc, lambda s: s.exact_match)
        avg_ed = sum(s.edit_distance for s in sc) / max(len(sc), 1)
        rb.cards([
            ("任务成功率", f"{succ.point:.1%}", f"95% CI {succ.low:.0%}-{succ.high:.0%}"),
            ("前置条件违反率", f"{viol.point:.1%}", "动作物理上做不到，量化规划幻觉"),
            ("序列精确匹配", f"{em.point:.1%}", "与 gold 完全一致，偏严格"),
            ("平均编辑距离", f"{avg_ed:.2f}", "归一化到 0-1，越小越接近"),
        ])

    # 分层成功率：只看总分会被简单题稀释，必须分层
    rb.h2("分层成功率")
    rows = []
    for name in model_names:
        sc = valid_scores[name]
        by_lv: dict[str, list[PlanScore]] = defaultdict(list)
        for s in sc:
            by_lv[s.difficulty].append(s)
        row = [name]
        for lv in ["L1", "L2", "L3"]:
            sub = by_lv.get(lv, [])
            iv = _rate(sub, lambda s: s.goal_achieved) if sub else None
            row.append(f"{iv.point:.0%} (n={iv.n})" if iv else "-")
        rows.append(row)
    rb.table(["模型", "L1 单步", "L2 含状态改变", "L3 多物体/容器"], rows)

    # 失败构成：这就是 bad case 归因的第一层输出
    rb.h2("失败原因分布")
    for name in model_names:
        sc = valid_scores[name]
        counter: dict[str, int] = defaultdict(int)
        for s in sc:
            if not s.goal_achieved:
                # first_fail_reason 为 "ok" 的含义是：所有动作都执行成功了，
                # 但最终目标条件没满足（比如东西放错了容器）。
                # 这不是「没问题」，是「任务理解不到位」，必须和达成目标区分开。
                # 实测踩到的坑：58 条这类失败被标成 ok，混进成功里，归因表直接失效。
                r = s.first_fail_reason
                counter["目标未达成（动作均可执行）" if r in ("", "ok") else r] += 1
        rb.h2(f"{name} 的失败构成")
        rb.table(
            ["原因", "数量", "占失败比例"],
            [[k, str(v), f"{v/max(sum(counter.values()),1):.0%}"] for k, v in sorted(counter.items(), key=lambda x: -x[1])],
            numeric_cols={1, 2},
        )

    # 运行统计：用来判断这次跑得可不可信
    # 缓存命中高说明大部分是复用的旧结果，API 失败高说明配置有问题
    rb.h2("运行统计")
    rb.table(
        ["模型", "有效/全量", "剔除的 API 失败", "缓存命中", "总 token"],
        [[m, f"{meta[m]['valid']}/{meta[m]['total']}", str(meta[m]["api_errors"]),
          str(meta[m]["cache_hits"]),
          f"{sum(s.tokens for s in all_scores[m]):,}"] for m in model_names],
        numeric_cols={1, 2, 3, 4},
    )
    # 把剔除率显式写出来，避免读者以为 315 条全部有效
    for m in model_names:
        if meta[m]["api_errors"]:
            rb.note(
                f"{m} 有 {meta[m]['api_errors']} 条样本未拿到模型输出"
                f"（{meta[m]['api_errors']/meta[m]['total']:.1%}），"
                "已从所有指标的分母中剔除。这批样本不会写进缓存，"
                "降低并发或稍后重跑可自动补齐。"
            )

    out_html = ROOT / "reports" / f"plan_{stamp}.html"
    rb.save(out_html)
    print(f"\nJSON: {out_json}\n报告: {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
