"""统计口径。

================================================================================
为什么需要这个文件
================================================================================

评测报出去的每个数字都要能回答一个问题：**这个数有多可信**。

315 条样本上跑出「成功率 42%」，听起来是个确定的结论。但如果样本只有
30 条，真实值可能在 25% 到 60% 之间，这个区间宽到没有任何决策价值。
所以每个百分比都必须带置信区间，这也是面试官最容易追问的点：
「你这个数字怎么来的，样本多少，误差多大」。

================================================================================
这个文件里的三类东西
================================================================================

1. wilson_interval      给百分比配误差范围
2. cohens_kappa         两个判定者的一致性（Judge 审计用）
3. 编辑距离相关         动作序列「差多少步」的量化

Kappa 目前只在 Day 6 的 Judge 审计里会用到，提前放在这里。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class Interval:
    """一个带置信区间的比例。

    point 是点估计（就是通常说的那个百分比），
    low / high 是区间两端，n 是样本量。

    报告里必须三个一起出，只报 point 是不诚实的。
    """
    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}] n={self.n}"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Interval:
    # Wilson 区间是一种更聪明的算法。它不直接假设“成功率服从正态分布”，而是从二项分布出发，做了一个数学变换，保证区间永远落在 [0,1] 之间。
    # 你不需要手算，Python 里一行 statsmodels.stats.proportion.proportion_confint(count, nobs, method='wilson') 就能出结果。
    #
    # 它的直观效果：
    #
    # 样本量 315 左右时，95% 置信区间大约是 ±5 个百分点。
    #
    # 样本量 100 左右时，大约是 ±9 个百分点。
    #
    # 样本量越小，区间越宽；样本量越大，区间越窄。
    #
    # 注释里说的“315 条样本时区间约正负 5 个点，100 条时约正负 9 个点”，就是这个意思
    """Wilson 置信区间（默认 95%，对应 z=1.96）。

    为什么不用最直觉的正态近似（p ± 1.96*sqrt(p(1-p)/n)）：
    样本小或者比例接近 0/1 时，正态近似会算出负的下界或超过 1 的上界，
    显然不合理。Wilson 区间在这两种情况下仍然稳定。

    直观理解：315 条样本时区间约正负 5 个点，
    100 条时约正负 9 个点。两个模型的差距小于区间宽度，
    就不能说谁更好 —— 这个判断规则在报告里要用上。
    """
    # ① 除零保护。样本数为 0 时返回全零区间，别崩
    if total == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    # ② 点估计，就是平常报出去的那个百分比
    p = successes / total
    # ③ 下面三行是 Wilson 区间的标准公式，不用手推。
    #    只要记住它比正态近似强在哪：区间永远落在 [0,1] 里
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    # ④ 上下界用 max/min 夹在 [0,1] 内。
    #    正态近似在 p 接近 0 或 1 时会算出负的下界，这里不会
    return Interval(p, max(0.0, center - margin), min(1.0, center + margin), total)


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Cohen's Kappa：两个判定者之间的一致性，扣除随机猜中的部分。

    为什么不能直接用「一致率」：
    如果两个判定者都在瞎猜（比如 90% 的样本都是 True），
    他们闭着眼全猜 True 也能有 90% 的一致率。
    Kappa 会把这种「因为简单而一致」的部分扣掉，剩下的是真本事。

    判读标准：
        > 0.6   判定可信，可以用
        < 0.4   基本不能用于决策
        = 0     和随机猜一样

    用途：Day 6 做 Judge 审计时，用它衡量「LLM 判定」和「状态机判定」
    到底有多一致。这是整个项目方法论价值最高的一步。
    """
    n = len(labels_a)
    # ① 空列表直接给 0
    if n == 0:
        return 0.0
    # ② 两组长度必须一致。不一样说明数据配对错了，
    #    抛异常比静默算出个错数好
    if len(labels_b) != n:
        raise ValueError("两组标签长度不一致")

    # ③ po = 实际观察到的一致率，就是「两人判得一样」的比例
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # ④ 各自判 True 的比例
    a_true = sum(1 for a in labels_a if a) / n
    b_true = sum(1 for b in labels_b if b) / n
    # ⑤ pe = 「两人各自独立瞎猜，碰巧猜到一块」的期望比例。
    #    这是 Kappa 的精髓：先把靠运气一致的部分算出来，后面减掉
    pe = a_true * b_true + (1 - a_true) * (1 - b_true)

    # ⑥ pe 趋近 1 说明样本几乎全是同一类，这时分母为 0，公式失效。
    #    退化处理：全判对了就给 1，否则给 0
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    # ⑦ (实际一致 − 运气一致) / (1 − 运气一致)。
    #    扣掉运气之后剩下的，才是两个判定者真正的共识
    return (po - pe) / (1 - pe)


@dataclass
class Confusion:
    """混淆矩阵。用于 Judge 审计时分析误判的方向。

    tp  真值通过，判定者也判通过
    fp  真值不通过，判定者却判通过（放水）
    fn  真值通过，判定者判不通过（漏杀）
    tn  真值不通过，判定者也判不通过

    其中 fp 最危险：一条实际做不到的动作序列被判成合格，
    上线之后就是机器人真的去做一件做不到的事。
    """
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0


def confusion_from_pairs(pairs: list[tuple[bool, bool]]) -> Confusion:
    """从成对的标签算混淆矩阵。pairs 每项是 (真值, 判定者)。"""
    # pairs 每项是 (真值, 判定者)。四格挨个数一遍
    tp = sum(1 for t, p in pairs if t and p)          # ① 真值过、判定也过
    fp = sum(1 for t, p in pairs if not t and p)      # ② 真值不过、判定却过 → 放水，最危险
    fn = sum(1 for t, p in pairs if t and not p)      # ③ 真值过、判定没过 → 漏杀
    tn = sum(1 for t, p in pairs if not t and not p)  # ④ 两边都判不过
    return Confusion(tp, fp, fn, tn)


def edit_distance(a: list[str], b: list[str]) -> int:
    """动作序列的编辑距离：把 a 改成 b 最少要几步（增/删/改各算一步）。

    为什么需要它：成功率只说「对不对」，但一条差 1 步的和一条差 8 步的
    都被判成失败，信息全丢了。编辑距离保留了「差多少」这个粒度。

    局限性（要写进报告）：它对顺序敏感。存在多条等价路径时，
    语义正确的方案可能因为顺序不同而得高分（看起来更差）。
    所以它只作辅助指标，不能当主指标。
    """
    # ① 完全相等直接返回 0，省掉后面一整套动态规划
    if a == b:
        return 0
    m, n = len(a), len(b)
    # ② 一边为空时，另一边每一步都得补上，距离就是对方的长度
    if m == 0:
        return n
    if n == 0:
        return m
    # ③ 标准动态规划。只保留上一行，空间复杂度从 O(m*n) 压到 O(n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        # ④ 每行开头填 i，代表「把 a 的前 i 个全删掉」的代价
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            # ⑤ 这一步相同就不花钱，不同要花 1 步替换
            cost = 0 if a[i - 1] == b[j - 1] else 1
            # ⑥ 三种走法取最小：
            #    prev[j] + 1      删掉 a 的一个
            #    cur[j-1] + 1     插入 b 的一个
            #    prev[j-1] + cost 替换（或者相同，不花钱）
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def normalize_plan(plan: list[str]) -> list[tuple]:
    """把动作序列归一化成可比较的形式。

    ============================================================================
    为什么必须有这个函数
    ============================================================================

    直接把模型输出和 gold 做字符串比较，会得到 0% 的匹配率 ——
    但这不是模型差，是指标被两件与能力无关的事情干扰了：

    1. 大小写。gold 写 "pillow"，模型输出 "Pillow"。
    2. NoOp 占位符。86% 的 gold 序列末尾有 NoOp()，模型不会输出它。

    实测：原始字符串比较 0 条匹配，归一化后 20 条匹配。
    所以这个函数的唯一目的就是剥掉格式噪声，让比较回到语义层面。

    归一化做的事：转小写、丢掉 NoOp、参数排序（PutObject 的参数顺序
    不影响语义，有的模型写 (物体, 容器)，有的写 (容器, 物体)）。
    """
    out: list[str] = []
    for a in plan:
        # ① 去首尾空格
        a = a.strip()
        # ② 空行和 NoOp 直接丢掉。86% 的 gold 序列末尾都带个 NoOp()，
        #    不剥掉的话每条样本都凭空多差一步
        if not a or a.lower().startswith("noop"):
            continue
        # ③ 有括号的正常动作，拆出动作名和参数
        m = re.match(r"([A-Za-z_]+)\s*\((.*)\)", a)
        if m:
            # ④ 动作名转小写。gold 写 PickupObject，模型可能写 pickupobject
            act = m.group(1).lower()
            # ⑤ 参数转小写并排序。PutObject(物体, 容器) 和 PutObject(容器, 物体)
            #    语义完全一样，排序后变成同一个字符串，不再被算成不一样
            args = sorted(x.strip().lower() for x in m.group(2).split(",") if x.strip())
            out.append(f"{act}({','.join(args)})")
        else:
            # ⑥ 没括号的（比如光秃秃一个词）整体转小写
            out.append(a.lower())
    return out


def plan_exact_equal(a: list[str], b: list[str]) -> bool:
    """归一化后比较两条动作序列是否一致。

    注意：即使修好归一化，这个指标的区分度依然很弱
    （实测 315 条里只有 20 条完全一致，约 6%）。
    因为达成同一目标往往有多条等价路径，模型走的路和专家不同是正常的。

    所以它只能当辅助指标，不能当主指标。
    主指标永远是「目标是否达成」，因为它不关心路径，只关心结果。
    """
    return normalize_plan(a) == normalize_plan(b)


def normalized_edit_distance(a: list[str], b: list[str]) -> float:
    """归一化到 0-1，方便跨样本比较。

    除以两者最大长度，这样长短不同的样本之间也能比。
    0 表示完全一致，1 表示完全不同。
    """
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))
