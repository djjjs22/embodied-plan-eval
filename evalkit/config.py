"""配置层。

================================================================================
这个文件做什么
================================================================================

两件事：管理模型清单，管理版本号。

================================================================================
关于 API key
================================================================================

key 只从环境变量或项目根 .env 读，绝不进代码、绝不进 Git。
.env 已经在 .gitignore 里，这是唯一允许放真 key 的地方。
注意别把 key 写进 .env.example —— 那是模板文件，会跟着 Git 走。

================================================================================
关于版本号（这是容易被忽略但很关键的设计）
================================================================================

PROMPT_VERSION 和 RUBRIC_VERSION 都参与缓存键的计算。

为什么必须这样：假设你改了 prompt 但没升版本号，重跑评测时
每条样本都会命中旧缓存，拿到的还是改之前的结果。
你会以为新 prompt 有效，其实根本没生效 —— 这种假结果最坑人。

所以规则是：改 prompt 或 rubric，必须同时升版本号。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

MODELS_PATH = ROOT / "config" / "models.yaml"
DATASET_PATH = ROOT / "dataset" / "cases.jsonl"
PROMPT_DIR = ROOT / "prompts"
ALFRED_ROOT = ROOT / "data" / "alfred"
REPORT_DIR = ROOT / "reports"
CACHE_DIR = ROOT / ".cache"

# 改 prompt 必须手动改这里，它参与缓存键计算，否则会命中旧缓存
PROMPT_VERSION = "embodied-plan-v1"
# rubric 版本同理，改评分标准必须升版
RUBRIC_VERSION = "plan-quality-v1"


def _load_dotenv() -> None:
    """读项目根 .env。用 setdefault，shell 里已设的优先级更高。

    .env 已在 .gitignore 里，这是唯一允许放真 key 的地方。
    别把 key 写进 .env.example，那是模板文件。
    """
    # ① 只在项目根找 .env，不存在就直接返回（不报错，环境变量可能已经设好了）
    path = ROOT / ".env"
    if not path.exists():
        return
    # ② 逐行读
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # ③ 空行、注释行、没有等号的行，三类都跳过
        if not line or line.startswith("#") or "=" not in line:
            continue
        # ④ 按第一个等号切开。用 split("=", 1) 而不是 split("=")，
        #    因为值里可能还有等号（base64 之类的）
        k, v = line.split("=", 1)
        # ⑤ setdefault 而不是直接赋值：shell 里已经设过的优先级更高，
        #    这样临时换一个 key 测试不用去改 .env
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    base_url: str
    api_key_env: str
    family: str
    temperature: float = 0.0
    max_tokens: int = 1024
    base_url_env: str | None = None
    extra_body: dict | None = None
    token_param: str = "max_tokens"

    @property
    def resolved_base_url(self) -> str:
        # ① 配置里指定了「从哪个环境变量读 base_url」的话，先看环境变量。
        #    这是给国内/国际版切换留的口子 —— 两个平台的 key 不通用
        if self.base_url_env:
            override = os.environ.get(self.base_url_env)
            if override:
                return override
        # ② 环境变量没设，用 yaml 里写死的那个
        return self.base_url

    def api_key(self) -> str:
        # ① 只从环境变量拿。绝不从代码里读，也绝不写进任何文件
        key = os.environ.get(self.api_key_env)
        # ② 拿不到就抛异常，把两种配置方式都写在报错里。
        #    静默返回空字符串的话，后面会得到一个看不懂的 401
        if not key:
            raise RuntimeError(
                f"缺环境变量 {self.api_key_env}，模型 {self.display_name} 无法调用。\n"
                f"两种方式任选其一：\n"
                f"  1. export {self.api_key_env}=...\n"
                f"  2. 写进项目根 .env（该文件已在 .gitignore 里）"
            )
        return key


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    # ① 先把 .env 读进环境变量，后面解析 api_key_env 时才找得到
    _load_dotenv()
    # ② 没传路径就用默认的 config/models.yaml
    path = path or MODELS_PATH
    # ③ 配置文件不存在就退回内置的三款模型，
    #    保证第一次 clone 下来不配也能跑通链路
    if not path.exists():
        return _default_models()

    # ④ 读 yaml。or {} 兜底：空文件会解析出 None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: dict[str, ModelSpec] = {}
    # ⑤ 逐个模型转成 ModelSpec。
    #    base_url 和 api_key_env 用 cfg[...] 直接取（缺了就该报错），
    #    其余用 cfg.get 给默认值（缺了不影响跑）
    for name, cfg in (raw.get("models") or {}).items():
        specs[name] = ModelSpec(
            name=name,
            display_name=cfg.get("display_name", name),
            base_url=cfg["base_url"],
            api_key_env=cfg["api_key_env"],
            family=cfg.get("family", name),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1024),
            base_url_env=cfg.get("base_url_env"),
            extra_body=cfg.get("extra_body"),
            token_param=cfg.get("token_param", "max_tokens"),
        )
    return specs


def _default_models() -> dict[str, ModelSpec]:
    return {
        "MiniMax-M3": ModelSpec(
            name="MiniMax-M3",
            display_name="MiniMax-M3",
            base_url="https://api.minimax.chat/v1",
            api_key_env="MINIMAX_API_KEY",
            family="minimax",
            base_url_env="MINIMAX_BASE_URL",
            # M3 默认输出思考过程，不禁用的话正文里混着推理文本，动作序列提取会失败
            extra_body={"thinking": {"type": "disabled"}},
            token_param="max_completion_tokens",
        ),
        "deepseek-chat": ModelSpec(
            name="deepseek-chat",
            display_name="DeepSeek-V3",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            family="deepseek",
        ),
        "qwen-plus": ModelSpec(
            name="qwen-plus",
            display_name="通义千问-Plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            family="qwen",
        ),
    }


def load_prompt(name: str) -> str:
    """从 prompts/ 目录读一个 prompt。

    ============================================================================
    为什么 prompt 要单独放文件，不写在 Python 里
    ============================================================================

    prompt 是评测实验的一部分，跟代码不是一回事。它需要：

    1. 能一眼看到 —— 面试官问「你的 prompt 怎么设计的」，翻代码给他看不合适
    2. 方便改 —— 改 .md 不用碰 Python，也不容易改出语法错误
    3. 能进版本对比 —— 改 prompt 前后的 diff 要能看清改了哪句话

    埋在 py 文件里这三样都做不到。

    文件不存在时直接报错并列出可用清单，**不静默返回空字符串** ——
    空 prompt 会让模型自由发挥，跑出来的结果全是垃圾，而且很难排查到是 prompt 没读到。
    """
    # ① 按名字拼路径，prompts/<name>.md
    path = PROMPT_DIR / f"{name}.md"
    # ② 不存在就抛异常，并把当前可用的清单列出来。
    #    绝不静默返回空字符串 —— 空 prompt 会让模型自由发挥，
    #    跑出一堆垃圾结果，而且极难排查到根因是 prompt 没读到
    if not path.exists():
        available = sorted(p.stem for p in PROMPT_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"找不到 prompt「{name}」，当前可用：{available}\n"
            f"目录：{PROMPT_DIR}"
        )
    # ③ 读出来并去首尾空白。strip 是必要的，
    #    .md 文件末尾的换行会混进 prompt 影响输出
    return path.read_text(encoding="utf-8").strip()


def ensure_dirs() -> None:
    # ① 报告目录和缓存目录，不存在就建。
    #    exist_ok=True 让它重复调用不会报错
    for d in (REPORT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
