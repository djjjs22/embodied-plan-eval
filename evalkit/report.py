"""HTML 报告生成。

================================================================================
为什么报告重要到要单独一个文件
================================================================================

报告是这个项目唯一的对外产出。面试官不会看你跑了多少条样本，
他看的是这张报告能不能回答「这个模型行不行、差在哪」。

所以报告有三条硬要求：
    1. 每个百分比必须带置信区间，不能只有孤零零一个数
    2. 必须分层，不能只有总分（总分会被简单题稀释）
    3. 必须能看出失败构成，而不只是成功率

================================================================================
为什么不引 jinja2 之类的模板引擎
================================================================================

报告结构很固定，纯字符串拼接就够，少一个依赖少一分环境问题的风险。
真要做复杂可视化的时候再考虑换。
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
margin:0;padding:32px 40px;color:#1f2328;background:#fff;line-height:1.6;max-width:1080px}
h1{font-size:22px;font-weight:600;margin:0 0 4px}
h2{font-size:17px;font-weight:600;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 16px}
th,td{border:1px solid #e5e7eb;padding:7px 10px;text-align:left}
th{background:#f6f8fa;font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.card{flex:1;min-width:150px;background:#f6f8fa;border-radius:8px;padding:12px 14px}
.card .k{font-size:12px;color:#6b7280}
.card .v{font-size:22px;font-weight:600;margin-top:2px}
.card .s{font-size:11px;color:#9ca3af;margin-top:2px}
.note{background:#fffbeb;border-left:3px solid #f59e0b;padding:10px 14px;font-size:13px;margin:12px 0}
.ok{background:#ecfdf5;border-left:3px solid #10b981;padding:10px 14px;font-size:13px;margin:12px 0}
.muted{color:#6b7280;font-size:12px}
code{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:12px}
"""


class ReportBuilder:
    def __init__(self, title: str, subtitle: str = ""):
        # ① 记下标题和副标题，render() 时才用
        self.title = title
        self.subtitle = subtitle
        # ② parts 是攒 HTML 片段的列表。用列表而不是反复拼字符串，
        #    省掉一次次创建新字符串的开销
        self.parts: list[str] = []

    def h2(self, text: str) -> None:
        # ① 必须 escape。任务描述里可能带 < > & 这类字符，
        #    不转义会直接把 HTML 结构撑坏
        self.parts.append(f"<h2>{escape(text)}</h2>")

    def note(self, text: str, kind: str = "note") -> None:
        # ① kind=ok 用绿色（好消息），其余用黄色（提醒）
        cls = "ok" if kind == "ok" else "note"
        # ② text 这里故意不 escape —— 调用方会传带标签的 HTML 进来
        self.parts.append(f'<div class="{cls}">{text}</div>')

    def cards(self, items: list[tuple[str, str, str]]) -> None:
        # ① 每项 (小标题, 数值, 副说明) 生成一张卡片，三个字段都转义
        blocks = "".join(
            f'<div class="card"><div class="k">{escape(k)}</div>'
            f'<div class="v">{escape(v)}</div>'
            f'<div class="s">{escape(s)}</div></div>'
            for k, v, s in items
        )
        # ② 外面套 flex 容器，卡片多了自动换行
        self.parts.append(f'<div class="cards">{blocks}</div>')

    def table(self, headers: list[str], rows: list[list[str]], numeric_cols: set[int] | None = None) -> None:
        # ① numeric_cols 里的列右对齐。数字列右对齐才方便上下横向比
        numeric_cols = numeric_cols or set()
        # ② 表头行
        th = "".join(f"<th>{escape(h)}</th>" for h in headers)
        # ③ 表体：逐行逐格生成，列号在 numeric_cols 里就加 class="num"
        body = "".join(
            "<tr>"
            + "".join(
                f'<td class="num">{escape(str(c))}</td>' if i in numeric_cols else f"<td>{escape(str(c))}</td>"
                for i, c in enumerate(r)
            )
            + "</tr>"
            for r in rows
        )
        self.parts.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")

    def render(self) -> str:
        # ① 时间戳。报告里必须能看出是哪一轮跑的，否则版本一多就分不清
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        # ② 副标题是可选的，没有就输出空字符串，别留个空 <p>
        sub = f'<p class="muted">{escape(self.subtitle)}</p>' if self.subtitle else ""
        # ③ 拼完整 HTML：head（含 CSS）+ 大标题 + 副标题 + 时间 + 正文片段
        return (
            "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<title>{escape(self.title)}</title><style>{CSS}</style></head><body>"
            f"<h1>{escape(self.title)}</h1>{sub}"
            f'<p class="muted">生成时间 {stamp}</p>'
            f"{''.join(self.parts)}</body></html>"
        )

    def save(self, path) -> None:
        # ① 转成 Path 对象，这样传字符串也能用
        p = Path(path)
        # ② 目录不存在就建。parents=True 支持一次建多级
        p.parent.mkdir(parents=True, exist_ok=True)
        # ③ 写盘
        p.write_text(self.render(), encoding="utf-8")
