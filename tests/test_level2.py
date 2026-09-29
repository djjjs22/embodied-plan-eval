"""关卡 2 验收测试。写完 exercise/level2.py 后跑：

    python -m pytest tests/test_level2.py -v

全绿才算过关。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exercise.level2 import World, Obj, apply_tool_action, resolve_target  # noqa: E402


def make_world(location=None, inventory=None):
    w = World()
    w.objects["cloth"] = Obj(name="cloth", otype="Cloth")
    w.objects["apple"] = Obj(name="apple", otype="Apple")
    w.location = location
    w.inventory = inventory
    return w


# ---------------------------------------------------------- resolve_target

def test_resolve_by_name():
    w = make_world()
    assert resolve_target(w, "cloth").name == "cloth"


def test_resolve_by_type_ignores_case():
    w = make_world()
    assert resolve_target(w, "Cloth").name == "cloth"


def test_resolve_empty_falls_back_to_inventory():
    """核心场景：CleanObject() 无参数时，作用在手上拿着的物体。"""
    w = make_world(inventory="cloth")
    assert resolve_target(w, "").name == "cloth"


def test_resolve_empty_with_empty_hands_returns_none():
    w = make_world(inventory=None)
    assert resolve_target(w, "") is None


def test_resolve_unknown_name_returns_none():
    w = make_world()
    assert resolve_target(w, "banana") is None


# ------------------------------------------------------- apply_tool_action

def test_clean_with_explicit_arg():
    w = make_world(location="sinkbasin", inventory="cloth")
    ok, why = apply_tool_action(w, "CleanObject", "cloth")
    assert ok, why
    assert w.objects["cloth"].cleaned is True


def test_clean_with_empty_arg_uses_inventory():
    """这就是修掉那条真 bug 的测试。"""
    w = make_world(location="sink", inventory="cloth")
    ok, why = apply_tool_action(w, "CleanObject", "")
    assert ok, why
    assert w.objects["cloth"].cleaned is True


def test_clean_requires_object_in_hand():
    w = make_world(location="sinkbasin", inventory=None)
    ok, why = apply_tool_action(w, "CleanObject", "cloth")
    assert not ok
    assert "手上" in why


def test_heat_requires_microwave():
    """拿对了东西但站错地方，必须拦住。这是模型最常犯的错。"""
    w = make_world(location="countertop", inventory="apple")
    ok, why = apply_tool_action(w, "HeatObject", "apple")
    assert not ok
    assert "microwave" in why.lower()
    assert w.objects["apple"].heated is False


def test_heat_ok_at_microwave():
    w = make_world(location="Microwave_1a2b3c", inventory="apple")
    ok, why = apply_tool_action(w, "HeatObject", "apple")
    assert ok, why
    assert w.objects["apple"].heated is True


def test_cool_ok_at_fridge():
    w = make_world(location="Fridge_9f8e7d", inventory="apple")
    ok, why = apply_tool_action(w, "CoolObject", "apple")
    assert ok, why
    assert w.objects["apple"].cooled is True


def test_unknown_action_rejected():
    w = make_world(location="sink", inventory="cloth")
    with pytest.raises(KeyError):
        apply_tool_action(w, "ExplodeObject", "cloth")
