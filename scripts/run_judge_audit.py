#!/usr/bin/env python3
"""LLM-as-a-Judge 可靠性审计。

================================================================================
这一步在回答什么问题
================================================================================

业界默认用模型给模型打分。但没人验证过那把尺子准不准。

本项目有客观真值（符号状态机的执行结果），所以能做一件别人做不了的事：
**让 Judge 独立判定一批样本，拿状态机的判定当标准答案，量一量 Judge 差多远。**

注意审计对象是谁：不是被测模型，是**打分的那把尺子**。

================================================================================
三种实验
================================================================================

1. 绝对判定：Judge 判断「这条序列能不能达成目标」，和状态机对比
2. 位置偏差：成对比较，正反两序各问一次，看它改不改口
3. 自偏好：  同一个 Judge 分别评别人和自己，看给自家是不是更宽松

第 3 项需要 Judge 自己也跑过一次评测，否则没有「自家的答卷」可评。

================================================================================
用法
================================================================================

    python scripts/run_judge_audit.py --limit 30            # 先试通
    python scripts/run_judge_audit.py                        # 全量
    python scripts/run_judge_audit.py --judge MiniMax-M3     # 换个裁判反向验证

两个模型都跑过评测之后，脚本会自动算出跨族对比和自偏好差值。

================================================================================
在整条流水线里处于哪一环
================================================================================

    dataset/cases.jsonl（题目）
        ↓
    scripts/run_eval.py        被测模型写规划 → 状态机执行 → 客观真值
        ↓                      结果落在 reports/plan_*.json
    scripts/run_judge_audit.py ← 本文件
        ↓
    reports/judge_audit_*.html

顺序不能倒：本文件吃的是 run_eval.py 的产出，没有那一步就报
「没有找到评测结果」。它自己不产生任何新的规划，也不给被测模型打分，
唯一的产出是「Judge 这把尺子能不能用」这个结论。
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# 把项目根目录塞进 sys.path。不加这一行，下面 `from evalkit...` 会找不到包，
# 因为脚本在 scripts/ 子目录里，Python 默认只搜索脚本所在目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.alfred import describe_scene_for_prompt
from evalkit.config import load_models, load_prompt
from evalkit.metrics import (
    Confusion,
    cohens_kappa,
    confusion_from_pairs,
    wilson_interval,
)
from evalkit.models import LLMClient
from evalkit.report import ReportBuilder

# 绝对路径的项目根。后面所有 dataset/ reports/ 都挂在这下面，
# 这样不管你从哪个目录执行这个脚本，路径都不会找错
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Judge 的提示词
# ---------------------------------------------------------------------------
# 设计要点：
# 1. 明确告诉它物理约束，否则它不知道「一次只能拿一个」，判错了不是推理差，
#    是我们没给足信息 —— 和给被测模型的 prompt 同理。
# 2. 要求输出 JSON，方便解析。
# 3. 强调「不要重新规划，只判断」—— 要的是它的判定，不是让它另写一条。

# prompt 内容在 prompts/ 目录，改这里只需要改对应的 .md 文件
JUDGE_SYSTEM = load_prompt("judge_system")
JUDGE_USER = load_prompt("judge_user")

# 位置偏差实验：成对比较，问哪条更好
PAIRWISE_SYSTEM = load_prompt("pairwise_system")
PAIRWISE_USER = load_prompt("pairwise_user")


def load_eval_results(path: Path | None) -> tuple[dict, Path]:
    """加载最近一次评测的结果。返回 {模型名: [样本]}。

    为什么要返回全部而不是第一个：自偏好实验需要同一个 Judge
    分别评多个被测模型，其中包括它自己。只取第一个模型的话，
    「给自家打分」这件事永远测不到。
    """
    # ① 调用方用 --results 明确指定了文件：就只认这一份，不去 reports/ 里扫。
    #    指定路径却偷偷合并别的文件，会让人以为自己在复现某一次实验结果
    if path is not None:
        # ① 读文件、按 UTF-8 解析成 JSON
        data = json.loads(path.read_text(encoding="utf-8"))
        # ② 兜底校验：顶层必须是 {模型名: [样本]} 这种字典。
        #    不校验的话，后面 all_results[tgt] 会抛一个看不懂的错
        if not isinstance(data, dict):
            raise ValueError(f"结果文件格式不对：{path}")
        # ③ 连路径一起返回，报告要显示「结果来源」，方便别人复现
        return data, path

    # ② 没指定文件时，把 reports/ 下所有 plan_*.json 合并起来，按修改时间从旧到新排好。
    #
    #    为什么要合并而不是只取最新一份：每次 run_eval 会生成一个新文件，
    #    只含那次跑的模型。分两次跑 M3 和 DeepSeek 就会有两个文件各含一个模型，
    #    只取最新就看不到另一个，跨族对比和自偏好都没法算。
    #
    #    按时间排序是关键 —— 「同名模型以最新的为准（后跑的覆盖先跑的）」
    #    全靠这个顺序实现。
    files = sorted((ROOT / "reports").glob("plan_*.json"), key=lambda p: p.stat().st_mtime)
    # ③ 一份结果都没有就直接报错退出，
    #    别让它带着空数据往下跑 —— 那样最后只会得到一个没内容的报告
    if not files:
        raise FileNotFoundError("没有找到评测结果，先跑 python scripts/run_eval.py")
    merged: dict = {}
    # ④ 逐个文件往 merged 里并
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            # ① 单份文件坏了（跑到一半被中断、写坏了）就跳过它。
            #    不能因为一份坏文件让整个审计跑不起来
            continue
        if isinstance(d, dict):
            # ① 后读到的同名模型直接覆盖前面的 —— 这就是「同名以最新为准」
            for k, v in d.items():
                merged[k] = v
    # ⑤ 全被跳过了说明没有一份能用。这时报错比返回空字典好排查
    if not merged:
        raise ValueError("评测结果都是空的或格式不对")
    # ⑥ 返回最新那一份的路径当「结果来源」，只是显示用
    return merged, files[-1]


def load_cases() -> dict:
    """把 dataset/cases.jsonl 读成 {样本 id: 样本原文}，供按 id 反查题目。

    为什么要用字典而不是列表：评测结果里每条只带一个 case_id，
    要拿任务描述、场景、gold 序列就得按 id 查；用列表的话
    每查一条都得全量扫一遍，审计几百条就是几百次全量扫描。
    """
    # ① 建空字典，按 id 往里塞
    out = {}
    # ② cases.jsonl 一行一个样本，按行切开
    for line in (ROOT / "dataset" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        # ① 跳过空行。文件末尾常有一个换行，不跳过的话 json.loads 会炸
        if line.strip():
            # ① 解析成字典，拿 id 当键。数据集保证 id 唯一，
            #    所以这里不做重名检查
            c = json.loads(line)
            out[c["id"]] = c
    return out


def goal_text(params: dict) -> str:
    """把 pddl_params 转成给 Judge 看的自然语言目标描述。

    为什么不直接给 JSON：Judge 是语言模型，自然语言描述的目标
    比结构化字段更容易被正确理解，也避免它去猜字段含义。
    """
    # ① 一段一段攒进列表，最后用中文分号拼成一句话
    parts = []
    # ② 操作对象：任务要动的那个东西，几乎所有任务都有
    if params.get("object_target"):
        parts.append(f"操作对象：{params['object_target']}")
    # ③ 最终放置位置。两种字段都兼容：mrecep_target 是「放进某个容器」，
    #    parent_target 是「直接放在某处」。取先有值的那个就行 ——
    #    Judge 只需要知道放哪，不需要知道这层数据结构的区别
    parent = params.get("mrecep_target") or params.get("parent_target")
    if parent:
        parts.append(f"最终放置在：{parent}")
    # ④ 需要开关的器具（开灯、开微波炉这类任务）
    if params.get("toggle_target"):
        parts.append(f"需要开关：{params['toggle_target']}")
    # ⑤ 切片任务才带的标记，不加这句 Judge 不知道「要不要切」
    if params.get("object_sliced"):
        parts.append("该物体需要被切片")
    # ⑥ 一个字段都没给时给个占位。不能传空字符串 ——
    #    那样 Judge 分不清是「题目本身没目标」还是「我们漏填了」
    return "；".join(parts) or "（未给出明确目标）"


def ask_judge(client: LLMClient, case: dict, plan: list[str]) -> tuple[bool | None, str]:
    """让 Judge 判定一条序列能否达成目标。

    返回 (判定结果, 原始输出)。解析不出 JSON 时返回 None。
    """
    # ① 先把场景翻译成给模型看的文字（物体清单、容器清单）
    scene = describe_scene_for_prompt(case)
    # ② 往提示词模板里填变量。plan 为空时填占位文案，
    #    否则模板里是一片空白，Judge 可能自己脑补一条序列出来评
    user = JUDGE_USER.format(
        task_desc=case["task_desc"],
        objects="、".join(scene["objects"]) or "（无）",
        receptacles="、".join(scene["receptacles"]) or "（无）",
        goal=goal_text(case["pddl_params"]),
        plan="\n".join(plan) if plan else "（空序列）",
    )
    # ③ 打一次 Judge。response_json=True 是要求它直接吐 JSON，
    #    省得从一大段分析文字里抠结论
    res = client.complete(system=JUDGE_SYSTEM, user=user, response_json=True)
    # ④ 调用本身失败（网络、鉴权、限流）→ 返回 None。
    #    绝不能当成「判不能达成」，那是它没说话，不是它说了不行
    if res.error:
        return None, res.error
    try:
        # ① 解析 JSON 取 achievable 字段。外面套 bool() 是因为
        #    模型偶尔给 "true" 字符串或 1，不转一下后面统计会对不上类型
        data = json.loads(res.text)
        return bool(data["achievable"]), res.text
    except Exception:  # noqa: BLE001
        # ① 解析失败返回 None 而不是 False —— 「没答上来」和
        #    「答了不能达成」是两件事，混成一个会把放水率算歪
        return None, res.text


def audit_one_model(judge: LLMClient, rows: list[dict], cases: dict, workers: int) -> dict:
    """对单个被测模型做一轮审计。

    truth  是状态机的客观判定（标准答案）
    judged 是 Judge 的主观判定（被审的尺子）

    Judge 调用时看不到 goal_achieved 字段，是盲评。这是实验成立的前提。
    """
    # ① 三本账：真值列表、Judge 判定列表、Judge 没答上来的条数。
    #    前两个按下标一一对应，后面靠 zip 配成对
    truth: list[bool] = []
    judged: list[bool] = []
    judge_fail = 0

    # ② 包一层闭包：样本和它的判定结果要一起回传。
    #    直接 submit(ask_judge, ...) 只能拿回判定，不知道这条属于哪个样本
    def one(r: dict):
        # ① 返回 (样本, (判定, 原始输出))。原始输出在这里取名 _raw 丢掉不用，
        #    审计阶段只看判定对不对，不存档逐条回复
        return r, ask_judge(judge, cases[r["case_id"]], r["plan"])

    # ③ 并发打 Judge。每条判定之间互不依赖，串行跑几百条要等很久
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # ① 一次性把所有样本丢进线程池
        futs = [pool.submit(one, r) for r in rows]
        # ② 谁先回来就先处理谁（as_completed），不用被最慢的那条卡住
        for fut in as_completed(futs):
            r, (verdict, _raw) = fut.result()
            # ① Judge 没答上来 → 只累加失败数，不进统计。
            #    它没有给出判定，就凑不出「真值 vs 判定」这一对
            if verdict is None:
                judge_fail += 1
                continue
            # ② 只有 Judge 真给了判定，这两个列表才同时追加，
            #    保证两个列表长度永远相等、下标永远对齐
            truth.append(bool(r["goal_achieved"]))
            judged.append(verdict)

    # ④ 一条都没审成（全失败或本来就没样本）→ 返回空字典。
    #    调用方看到空就知道这个模型要跳过，而不是当成「0 条、一致率 0%」
    n = len(truth)
    if n == 0:
        return {}
    # ⑤ 把 (真值, 判定) 配成对算混淆矩阵，后面所有指标都从它派生
    cm: Confusion = confusion_from_pairs(list(zip(truth, judged)))
    return {
        "n": n,
        "cm": cm,
        # Kappa：扣掉「瞎猜也能猜对」的那部分之后的一致性，
        # 比裸一致率诚实 —— 真值里 90% 都是「能达成」时，
        # 一个永远回答「能达成」的 Judge 裸一致率 90%，Kappa 却是 0
        "kappa": cohens_kappa(truth, judged),
        # 一致率带 Wilson 置信区间，样本少的时候区间会很宽，别只看点估计
        "agree": wilson_interval(cm.tp + cm.tn, n),
        # 放水率：真值做不到却判成能。分母是「真值做不到」的那些
        "fp_rate": wilson_interval(cm.fp, max(cm.fp + cm.tn, 1)),
        "judge_fail": judge_fail,
    }


def run_position_bias(judge: LLMClient, rows: list[dict], cases: dict, workers: int,
                      max_pairs: int = 60) -> tuple[int, int]:
    """位置偏差实验。返回 (有效对数, 改口次数)。

    只对一批样本做，不按模型重复做 —— 调用量会翻倍，没必要。
    """
    # ① 挑出能凑成对的样本：模型自己得有序列，题目还得有 gold 序列，
    #    缺一个就没法比。再截前 max_pairs 条 —— 这条实验调用量是 2 倍
    pair_rows = [r for r in rows if r["plan"] and cases[r["case_id"]]["gold_plan"]][:max_pairs]
    # ② 一条都凑不出来就返回 (0, 0)，调用方看到 flip_total=0 会跳过这一节
    if not pair_rows:
        return 0, 0

    def ask_pair(r, swap: bool):
        """问一次「A 和 B 哪条更好」，返回 "A" / "B" / ""。

        swap=False 时 A 是模型序列、B 是 gold；swap=True 时反过来。
        同一条样本问两次，就是为了看它换了个位置会不会改口。
        """
        # ① 取题目原文和场景描述
        case = cases[r["case_id"]]
        scene = describe_scene_for_prompt(case)
        # ② 默认摆法：A = 模型序列，B = gold
        pa, pb = r["plan"], case["gold_plan"]
        # ③ swap=True 就把两个对调 —— 这就是「同一条，反着再问一次」
        if swap:
            pa, pb = pb, pa
        # ④ 填模板。注意这里没传目标描述：成对比较只问「哪条更好」，
        #    不给目标才能测出它是不是单纯偏爱写在前面的那条
        user = PAIRWISE_USER.format(
            task_desc=case["task_desc"],
            objects="、".join(scene["objects"]) or "（无）",
            plan_a="\n".join(pa),
            plan_b="\n".join(pb),
        )
        res = judge.complete(system=PAIRWISE_SYSTEM, user=user, response_json=True)
        try:
            # ① 取 choice 字段并转大写。模型可能写 "a"、"A" 甚至 "choice a"，
            #    统一成大写后面才好比对
            return str(json.loads(res.text).get("choice", "")).upper()
        except Exception:  # noqa: BLE001
            # ① 解析不出来就返回空串，后面靠 in ("A","B") 把它筛掉，
            #    不当成弃权、也不当成选了 A
            return ""

    # ③ 正反两批结果分开存，键统一用 case_id，方便最后按 id 配对
    fwd_res: dict[str, str] = {}
    rev_res: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # ① 正序一批、反序一批一起提交。这就是调用量翻倍的地方，
        #    所以前面要用 max_pairs 卡住样本数
        fwd_futs = {pool.submit(ask_pair, r, False): r["case_id"] for r in pair_rows}
        rev_futs = {pool.submit(ask_pair, r, True): r["case_id"] for r in pair_rows}
        # ② 结果回来时用 future→case_id 这张映射表回写，
        #    这样完全不用管谁先完成、顺序对不对
        for fut in as_completed(fwd_futs):
            fwd_res[fwd_futs[fut]] = fut.result()
        for fut in as_completed(rev_futs):
            rev_res[rev_futs[fut]] = fut.result()

    # ④ 逐条比对正反两次的回答
    flip_total = flip_hits = 0
    for cid in fwd_res:
        # ① 取同一条样本正反两次的回答；哪边缺失就是空串
        a, b = fwd_res.get(cid, ""), rev_res.get(cid, "")
        # ② 只有两次都明确选了 A 或 B，这一对才有效，计入分母。
        #    答非所问的样本直接丢掉，不然会把翻转率稀释
        if a in ("A", "B") and b in ("A", "B"):
            flip_total += 1
            # ① 正向时 A 是模型序列，反向时 A 是 gold —— 也就是说
            #    「正向选 A」和「反向选 B」才是同一个选择（都挑了模型序列）。
            #    两者不一致，就说明换个摆放位置它就改口了，计一次翻转
            if (a == "A") != (b == "B"):
                flip_hits += 1
    # ⑤ 返回 (有效对数, 改口次数)。只回两个数、不折算成率，
    #    是因为置信区间要在报告那一步按样本数现算，那里才看得见区间宽不宽
    return flip_total, flip_hits


def main() -> int:
    """审计主流程：读参数 → 装真值和题目 → 逐个被测模型盲评 → 出 HTML 报告。

    进来什么都不用传（命令行参数自己解析），出去只返回一个退出码：
    0 表示报告已生成，非 0 表示中途退出（模型名写错、没有可审样本）。
    报告写到 reports/judge_audit_<时间戳>.html，不覆盖旧报告 ——
    换 Judge 重跑时旧的还要留着对比。

    下面每一段前面有个带序号的分割线，编号在每段内部各自从 ① 重新开始。
    """
    # ==================================================================
    # ① 解析命令行参数
    # ==================================================================
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="deepseek-v4-flash", help="用作评判者的模型")
    ap.add_argument("--target", default="", help="审哪些被测模型，逗号分隔，默认全部")
    ap.add_argument("--results", default="", help="评测结果 JSON，默认取最新一份")
    ap.add_argument("--limit", type=int, default=0, help="只审前 N 条，调试用")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-bias", action="store_true", help="跳过位置偏差实验")
    args = ap.parse_args()

    # ==================================================================
    # ② 确认 Judge 这个模型在配置里存在
    # ==================================================================
    # ① 读 models 配置（API key、base_url、并发数这些）
    specs = load_models()
    # ② 名字写错就立刻退出。不在这里拦住的话，要等几十次调用全部失败
    #    才知道拼错了，白等一场
    if args.judge not in specs:
        print(f"配置里没有模型 {args.judge}，现有：{list(specs)}")
        return 1

    # ==================================================================
    # ③ 装「标准答案」和题目原文
    # ==================================================================
    # ① 评测结果 = 状态机跑出来的客观判定，本脚本的 ground truth
    all_results, results_path = load_eval_results(Path(args.results) if args.results else None)
    # ② 题目原文（任务描述、场景、gold），用来给 Judge 拼提示词
    cases = load_cases()

    # ==================================================================
    # ④ 确定审哪些被测模型
    # ==================================================================
    # ① --target 没给就默认审结果里所有模型
    targets = [t.strip() for t in args.target.split(",") if t.strip()] or list(all_results)
    # ② 写错名字或结果里没有的模型，列出来直接退出。
    #    静默跳过一个模型，最后报告就会少一块，看的人还以为是它没跑
    missing = [t for t in targets if t not in all_results]
    if missing:
        print(f"结果里没有这些模型：{missing}，现有：{list(all_results)}")
        return 1

    # ③ 先把这次实验的配置打印出来。报告是 HTML，但终端这三行是
    #    「谁评的、评了谁、真值从哪来」，复现实验时最先用得上
    print(f"Judge = {args.judge}")
    print(f"被测 = {targets}")
    print(f"结果来源 = {results_path.name}")

    # ④ 建 Judge 客户端。整个审计过程只用这一个，
    #    换 Judge 就是换这里，别的都不动
    judge = LLMClient(specs[args.judge])

    # ==================================================================
    # ⑤ 逐个被测模型审计
    # ==================================================================
    audits: dict[str, dict] = {}
    for tgt in targets:
        # ① 先剔掉跑挂了的样本（带 error 字段的没有 plan，审不了）
        rows = [r for r in all_results[tgt] if not r.get("error")]
        # ② --limit 是用来试通的，只取前 N 条，能省一大笔调用费
        if args.limit:
            rows = rows[: args.limit]
        # ③ 审一轮。Judge 在这条链路里看不到 goal_achieved，是盲评
        res = audit_one_model(judge, rows, cases, args.workers)
        # ④ 空字典 = 一条都没审成，跳过它但不要中断整个脚本
        if not res:
            print(f"  {tgt}: 没有可审计的样本，跳过")
            continue
        # ⑤ 打上「是不是在评自己」的标记。这是自偏好实验的分组依据，
        #    必须在这里记下来，后面报告要靠它把自评和他评拆开
        res["is_self"] = tgt == args.judge
        audits[tgt] = res
        tag = "自评" if res["is_self"] else "他评"
        print(
            f"  {tgt:<20}({tag}) 一致率 {res['agree'].point:.1%} | "
            f"Kappa {res['kappa']:.3f} | 放水率 {res['fp_rate'].point:.1%}"
        )

    # ⑥ 一个模型都没审成就没必要出报告了，直接失败退出
    if not audits:
        print("没有任何可审计的结果")
        return 1

    # ==================================================================
    # ⑦ 位置偏差（只对第一个模型做，避免调用翻倍）
    # ==================================================================
    # ① 先置零。--no-bias 或一条都没凑成时保持 0，
    #    报告那一节靠 flip_total 是否为 0 来决定写不写
    flip_total = flip_hits = 0
    if not args.no_bias:
        # ① 固定拿第一个被测模型做，不重复做 —— 这条实验调用量是 2 倍
        first = targets[0]
        rows = [r for r in all_results[first] if not r.get("error")]
        if args.limit:
            rows = rows[: args.limit]
        flip_total, flip_hits = run_position_bias(judge, rows, cases, args.workers)

    # ==================================================================
    # ⑧ 拼报告
    # ==================================================================
    # ① 建报告对象，标题下面那行小字写清真值来源 ——
    #    本审计的全部权威都来自「真值是状态机跑出来的」这件事
    rb = ReportBuilder(
        "LLM-as-a-Judge 可靠性审计",
        f"评判模型 {args.judge} · 客观真值来自符号状态机执行",
    )

    # ② 每个被测模型单独一节
    for tgt, a in audits.items():
        cm: Confusion = a["cm"]
        # ① 按 Kappa 给一句人话结论。0.6 / 0.4 是常用档位，
        #    先把数字翻译成「能不能用」，看报告的人不用自己去查表
        verdict_text = (
            "判定可信，可作客观真值的替代"
            if a["kappa"] >= 0.6
            else ("一致性中等，只能作辅助信号" if a["kappa"] >= 0.4 else "与真值偏离较大，不能用于决策")
        )
        # ② 标题里明说是自评还是他评，避免把「评自己」当成普通他评读过去
        title = f"{tgt}" + ("（自评：Judge 评自己的输出）" if a["is_self"] else "（他评）")
        rb.h2(f"审计对象：{title}")

        # ③ 四个卡片：一致率、Kappa、放水率、精确率/召回率。
        #    放水率单独拎出来，因为「把做不到的判成能」是最危险的方向
        rb.cards([
            ("与真值一致率", f"{a['agree'].point:.1%}", f"95% CI {a['agree'].low:.0%}-{a['agree'].high:.0%}"),
            ("Cohen's Kappa", f"{a['kappa']:.3f}", verdict_text),
            ("误判通过率", f"{a['fp_rate'].point:.1%}", "实际做不到却判成能，最危险"),
            ("精确率 / 召回率", f"{cm.precision:.2f} / {cm.recall:.2f}", "判能的准度 / 覆盖度"),
        ])

        # ④ 混淆矩阵，横着是 Judge 的判定、竖着是真值。
        #    四个格子里最该盯的是「真值不能达成 / Judge 判能」那一格
        rb.table(
            ["", "Judge 判能达成", "Judge 判不能达成"],
            [
                ["真值能达成", f"一致 {cm.tp}", f"漏杀 {cm.fn}"],
                ["真值不能达成", f"放水 {cm.fp}", f"一致 {cm.tn}"],
            ],
        )
        # ⑤ 明细表，把置信区间也列上 —— 样本少的时候区间很宽，
        #    只看点估计会高估结论的可靠性
        rb.table(
            ["指标", "数值"],
            [
                ["样本数", f"{a['n']}（Judge 解析失败 {a['judge_fail']} 条已剔除）"],
                ["一致率", f"{a['agree'].point:.1%}  [{a['agree'].low:.1%}, {a['agree'].high:.1%}]"],
                ["Cohen's Kappa", f"{a['kappa']:.3f}"],
                ["精确率 / 召回率", f"{cm.precision:.1%} / {cm.recall:.1%}"],
                ["放水率", f"{a['fp_rate'].point:.1%}"],
            ],
        )
        # ⑥ 判它偏乐观还是偏悲观。两个方向的错误代价不一样：
        #    放水是让不合格的规划混过去，漏杀只是低估能力，
        #    所以下面两种文案的语气一个是警告、一个是「还行」
        if cm.fp > cm.fn:
            rb.note(
                f"放水 {cm.fp} 条多于漏杀 {cm.fn} 条。Judge 偏向乐观，"
                "会把做不到的序列判成合格。这个方向的错误最危险，"
                "因为它让不合格的规划通过审核。"
            )
        else:
            rb.note(
                f"漏杀 {cm.fn} 条多于放水 {cm.fp} 条。Judge 偏向保守，"
                "会低估模型能力，但不会放过坏样本。",
                kind="ok",
            )

    # ------------------------------------------------------------------
    # ③ 自偏好对比（多模型审计才有的产出）
    # ------------------------------------------------------------------
    # ① 把审计结果拆成两组：Judge 评自己那一份、评别人那几份
    self_a = audits.get(args.judge)
    others = {k: v for k, v in audits.items() if k != args.judge}
    rb.h2("自偏好对比")
    # ② 两组都有才算得出来。缺任何一组就走 else 分支提示怎么补
    if self_a and others:
        rows = []
        # ① 逐个「评自己 vs 评某某」算差值。
        #    三个差值同向为正，才说明它确实对自己更宽松
        for k, v in others.items():
            kd = self_a["kappa"] - v["kappa"]
            fd = self_a["fp_rate"].point - v["fp_rate"].point
            rd = self_a["cm"].recall - v["cm"].recall
            rows.append([
                f"评自己 vs 评 {k}",
                f"{self_a['kappa']:.3f} / {v['kappa']:.3f}",
                f"{kd:+.3f}",
                f"{fd:+.1%}",
                f"{rd:+.1%}",
            ])
        rb.table(
            ["对比", "自评 Kappa / 他评 Kappa", "Kappa 差", "放水率差", "召回率差"],
            rows,
        )
        # ② 差值为正说明对自己更宽松。
        #    0.02 是「肉眼可见」的门槛，比这更小的差异可能只是样本波动
        if any(self_a["fp_rate"].point - v["fp_rate"].point > 0.02 for v in others.values()):
            rb.note(
                "自评的放水率高于他评，说明 Judge 对自己的输出更宽容。"
                "同族模型互相评审会系统性高估分数，"
                "所以评审模型必须和被测模型跨族。"
            )
        else:
            rb.note("自评与他评差异不明显，未观察到自偏好。", kind="ok")
    else:
        # ① 做不了就明说怎么补，别留一个空节让人猜。
        #    常见原因：Judge 那个模型自己还没跑过评测
        rb.note(
            f"当前评测结果里没有 {args.judge} 自己的输出，做不了自偏好对比。"
            f"补一步即可：python scripts/run_eval.py --models {args.judge}"
        )

    # ------------------------------------------------------------------
    # ④ 位置偏差
    # ------------------------------------------------------------------
    # ① 只有真的凑出了有效对、且没被 --no-bias 跳过，才写这一节
    if flip_total and not args.no_bias:
        rb.h2("位置偏差")
        # ① 翻转率 = 改口次数 / 有效对数
        rate = flip_hits / flip_total
        # ② 按 p=0.5 算标准误，再乘 1.96 得到 95% 置信区间。
        #    用 0.5 而不是用实测 rate，是因为要和「抛硬币」这个基线比
        se = (0.5 * 0.5 / flip_total) ** 0.5
        lo, hi = rate - 1.96 * se, rate + 1.96 * se
        rb.table(
            ["指标", "数值"],
            [
                ["成对比较样本", str(flip_total)],
                ["改口次数", str(flip_hits)],
                ["翻转率", f"{rate:.1%}  95% CI [{lo:.0%}, {hi:.0%}]"],
                ["随机基线", "50%（瞎选时正反序指向同一条的概率）"],
            ],
        )
        # ============================================================
        # 判读翻转率时，基线是 50%，不是 0
        # ============================================================
        # 我第一版拿 10% 当阈值，那是错的 —— 那等于假设随机情况不会翻转。
        # 正确做法：和 50% 比。区间覆盖 50% 就等于「和抛硬币没区别」。
        # ② 翻转率越低越稳。三档的结论分别是：
        #    ≥0.4「和抛硬币没区别」、≥0.15「不稳，要正反各跑一次」、其余「可用」
        if rate >= 0.4:
            rb.note(
                f"翻转率 {rate:.1%}，95% CI [{lo:.0%}, {hi:.0%}]，随机基线 50%。"
                "区间覆盖 50%，说明成对比较能力与抛硬币无显著差异。"
                "结论：Judge 的绝对判定可用，但不要用它做两两比较。"
            )
        elif rate >= 0.15:
            rb.note(
                f"翻转率 {rate:.1%}，95% CI [{lo:.0%}, {hi:.0%}]，随机基线 50%。"
                "判定不够稳定，做两两比较时应正反各跑一次，不一致判平局。"
            )
        else:
            rb.note(
                f"翻转率 {rate:.1%}，95% CI [{lo:.0%}, {hi:.0%}]，"
                "显著优于随机基线 50%，Judge 对呈现顺序不敏感。",
                kind="ok",
            )

    # ------------------------------------------------------------------
    # ⑤ 结论 + 落盘
    # ------------------------------------------------------------------
    rb.h2("结论")
    # ① 挑 Kappa 最高的那个当代表。取最高是故意的：
    #    连表现最好的一组都只有这个水平，就是 Judge 能力的上限
    best = max(audits.items(), key=lambda kv: kv[1]["kappa"])
    # ② 把结论翻译成一句「所以呢」：换成模型来判，可信度就是这个数
    rb.note(
        f"在 {sum(a['n'] for a in audits.values())} 条判定上，"
        f"Judge（{args.judge}）对被评模型的最高 Kappa 为 {best[1]['kappa']:.3f}（{best[0]}）。"
        "这个数字的含义是：如果项目里没有可执行的状态机，"
        "改用模型来判定规划是否合格，可信度就是这个水平。"
    )

    # ③ 文件名带时间戳，重跑不覆盖旧报告 ——
    #    换 Judge 反向验证时要留着两次结果对比
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"judge_audit_{stamp}.html"
    # ④ 写文件。目录必须已经存在，reports/ 由 run_eval.py 建好
    rb.save(out)
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
