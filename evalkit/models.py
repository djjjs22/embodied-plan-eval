"""模型调用层。

================================================================================
这个文件解决什么问题
================================================================================

被测对象是通过 API 调用的模型。这个文件的职责是：
    1. 把 prompt 发出去，拿回文本
    2. 从返回的文本里把动作序列抠出来

第 2 步比看起来重要得多。模型不会老老实实只输出动作，它会在前后加解释、
把内容包在代码块里、每行加序号，带思考功能的模型（M3、R1 一类）还会把
推理过程混进正文。不处理好这些，后面全判失败，而且很难排查 —— 你会以为
是模型规划能力差，其实是解析没做对。

================================================================================
为什么走 OpenAI 兼容协议
================================================================================

国内主流厂商（MiniMax、DeepSeek、通义、智谱）都提供 OpenAI 兼容端点，
统一用一个客户端，只换 base_url 和 key，省掉四套 SDK。
这是项目能快速切换被测模型的前提。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import ModelSpec, load_prompt

# 带思考过程的模型（M3、R1 一类）在开关失效时会把推理混进正文，
# 形如 <think>用户想统计销售额...<think>。不剥掉的话整条序列解析失败。
# 而且这个剥离必须发生在「找代码块」之前，顺序反了会把推理文本当成动作。
_THINK_BLOCK = re.compile(r"</?think>.*?(?:</think>|<think>|$)", re.DOTALL | re.IGNORECASE)

# 模型喜欢用 ```sql 或 ``` 把内容包起来
_CODE_FENCE = re.compile(r"```(?:[a-zA-Z]*)\s*(.*?)```", re.DOTALL)

# 动作名白名单。用它过滤掉解释性文字，只保留真正的动作行。
# 这比「取所有行」稳得多：模型可能在开头写一句「以下是我的规划：」，
# 那行不是动作，必须滤掉。
_ACTION_RE = re.compile(
    r"\b(GotoLocation|PickupObject|PutObject|CleanObject|HeatObject|CoolObject|SliceObject|ToggleObject|NoOp)\b",
    re.IGNORECASE,
)


@dataclass
class LLMResult:
    """一次调用的完整结果。

    保留 token 和耗时是为了算效率指标（方案里的效率层）。
    单次失败不抛异常，而是把错误记在 error 里 ——
    一条用例挂掉不该中断整轮评测。
    """
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMClient:
    """模型客户端。

    重试逻辑自己实现而不用 SDK 内置的 max_retries，
    是为了能统计失败次数 —— 频繁失败本身是个信号，说明配置有问题。
    """

    def __init__(self, spec: ModelSpec, timeout_sec: float = 60.0, max_retries: int = 3):
        self.spec = spec
        self._client = OpenAI(
            api_key=spec.api_key(),
            base_url=spec.resolved_base_url,
            timeout=timeout_sec,
            max_retries=0,
        )
        self._max_retries = max_retries

    def complete(self, system: str, user: str, temperature: float | None = None,
                 response_json: bool = False) -> LLMResult:
        """发一次对话。

        注意 token_param：新模型（M3）用 max_completion_tokens，
        老模型用 max_tokens，按配置走，写错会被服务端忽略导致输出截断。
        """
        # ① 组装请求参数。注意最后一个键是动态的：
        #    新模型（M3）用 max_completion_tokens，老模型用 max_tokens。
        #    按配置取键名，写错的话服务端会默默忽略，表现为输出被截断
        kwargs: dict = {
            "model": self.spec.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # ② 温度：这次没单独指定就用配置里的默认值
            "temperature": self.spec.temperature if temperature is None else temperature,
            self.spec.token_param: self.spec.max_tokens,
        }
        # ③ 要 JSON 输出时加这个参数（Judge 打分时用得上）
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        # ④ 厂商私有字段，比如 M3 的 thinking 开关
        if self.spec.extra_body:
            kwargs["extra_body"] = dict(self.spec.extra_body)

        # ⑤ 记开始时间，用来算耗时
        started = time.time()
        last_err = ""
        # ⑥ 自己实现重试而不用 SDK 内置的 max_retries，
        #    为的是能统计失败次数 —— 频繁失败本身就是「配置有问题」的信号
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                usage = resp.usage
                # ⑦ 成功：文本、两个 token 数、耗时一起带回去。
                #    getattr 兜底是因为有些端点不返回 usage
                return LLMResult(
                    text=resp.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=int((time.time() - started) * 1000),
                )
            except Exception as exc:  # noqa: BLE001
                # ⑧ 网络类异常种类太多，统一兜住。
                #    退避时间随次数递增（1.5 秒、3 秒…），
                #    别在限流的时候还一个劲猛打
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))

        # ⑨ 重试全失败：返回空文本 + error，不抛异常。
        #    一条用例挂掉不该中断整轮评测，让它算 0 分继续跑
        return LLMResult(
            text="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=int((time.time() - started) * 1000),
            error=last_err,
        )


def extract_plan(text: str) -> list[str]:
    """从模型输出里提取动作序列。这是本文件最关键的函数。

    模型常见的三种污染，逐个处理：

    1. 思考过程残留    <think>...<think>\nPickupObject(Apple)
    2. 代码块包裹      ```\nPickupObject(Apple)\n```
    3. 行首序号或解释  1. PickupObject(Apple)  # 拿起苹果

    处理顺序不能乱：先剥思考块，再取代码块内部，最后逐行过滤。
    """
    # ① 空输入直接返回空列表
    if not text:
        return []
    # ② 第一步：剥掉思考过程残留。
    #    必须排在最前面 —— 思考块里也会出现动作名，
    #    不先剥掉的话整段推理会被当成动作序列
    cleaned = _THINK_BLOCK.sub("", text)
    # ③ 第二步：模型爱用 ``` 把内容包起来，有的话只取块内；
    #    没有就用整段文本
    m = _CODE_FENCE.search(cleaned)
    body = m.group(1) if m else cleaned

    lines = []
    # ④ 第三步：逐行清理
    for raw in body.splitlines():
        line = raw.strip()
        # ⑤ 空行跳过
        if not line:
            continue
        # ⑥ 去行首序号（1. / 2、/ - / * / •）
        line = re.sub(r"^\s*(?:\d+[\.、\)]|[-*•])\s*", "", line)
        # ⑦ 只保留含已知动作名的行。模型开头常写「以下是我的规划：」，
        #    那行不是动作，靠白名单滤掉（白名单在文件上方 _ACTION_RE）
        if not _ACTION_RE.search(line):
            continue
        # ⑧ 去行尾注释（# 或 // 后面的都不算）
        line = re.split(r"\s+#|\s+//", line)[0].strip()
        lines.append(line)
    return lines


# ============================================================================
# Prompt 模板
# ============================================================================
#
# 这个模板决定了测的是什么，改它等于改实验条件，所以有版本号管理
# （见 config.py 的 PROMPT_VERSION，改模板必须升版，否则会命中旧缓存）。
#
# 模板里包含三块内容：能用的动作、硬性物理约束、输出格式。
# 「硬性约束」那段不能省 —— 如果不告诉模型「一次只能拿一个」，
# 它违反规则是我们没说清楚，不是它规划能力差。
# 把规则写进 prompt，才能保证测的是能力而不是信息缺失。

# prompt 内容在 prompts/ 目录。改 prompt 只需要改那里的 .md 文件，不用碰代码。
# 设计与注意事项见 prompts/README.md。
PROMPT_SYSTEM = load_prompt("plan_system")
PROMPT_USER = load_prompt("plan_user")
