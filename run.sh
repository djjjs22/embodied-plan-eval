#!/usr/bin/env bash
# 评测启动器：自动挑一个装了依赖的解释器，不用为每个人的机器改路径。
#
# 优先级：
#   1. $EMBODIED_PYTHON  你自己指定的解释器
#   2. 项目里的 .venv
#   3. PATH 上第一个能 import yaml + openai 的 python3
#
# 用法（在任意目录都行）:
#   ./run.sh --limit 10 --workers 2        # 先跑通链路
#   ./run.sh --models MiniMax-M3           # 全量
#   ./run.sh --models MiniMax-M3 --workers 3 --no-cache
#
# 依赖没装？先跑：pip install -e ".[dev]"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 一个解释器算「可用」，得同时能 import 这两个包 —— evalkit 依赖它们，
# 缺任一个都会在半路炸掉，不如这里就挑干净。
usable() {
  [ -n "${1:-}" ] && [ -x "$1" ] && "$1" -c 'import yaml, openai' >/dev/null 2>&1
}

pick_python() {
  if usable "${EMBODIED_PYTHON:-}"; then
    echo "$EMBODIED_PYTHON"
    return
  fi

  if usable "$ROOT/.venv/bin/python"; then
    echo "$ROOT/.venv/bin/python"
    return
  fi

  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && usable "$(command -v "$cand")"; then
      command -v "$cand"
      return
    fi
  done
}

PY="$(pick_python)"

if [ -z "$PY" ]; then
  echo "找不到装了依赖的 Python（需要 yaml + openai）。" >&2
  echo "" >&2
  echo "  1) 在项目根跑： pip install -e \".[dev]\"" >&2
  echo "  2) 或者显式指路： EMBODIED_PYTHON=/path/to/python ./run.sh" >&2
  exit 1
fi

exec "$PY" "$ROOT/scripts/run_eval.py" "$@"
