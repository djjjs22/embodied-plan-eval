"""关卡 1 的验收测试。

写完 exercise/level1.py 之后跑：
    python -m pytest tests/test_level1.py -v

全绿即过关。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from exercise.level1 import Obj, World, apply_pickup, apply_put, parse_action_line


def make_world() -> World:
    return World(
        objects={
            "apple": Obj(name="Apple_1", otype="Apple", parent="Table_1"),
            "bread": Obj(name="Bread_1", otype="Bread", parent="Table_1"),
            "fridge": Obj(name="Fridge_1", otype="Fridge"),
        }
    )


def test_parse_single_arg():
    assert parse_action_line("PickupObject(Apple_1)") == ("PickupObject", ["Apple_1"])


def test_parse_two_args():
    assert parse_action_line("PutObject(Apple_1, Fridge_1)") == (
        "PutObject",
        ["Apple_1", "Fridge_1"],
    )


def test_parse_strips_leading_number():
    assert parse_action_line("1. PickupObject(Apple_1)") == ("PickupObject", ["Apple_1"])
    assert parse_action_line("2、GotoLocation(Fridge_1)") == ("GotoLocation", ["Fridge_1"])


def test_parse_empty_line():
    assert parse_action_line("") == (None, [])
    assert parse_action_line("   ") == (None, [])


def test_find_by_type_name_case_insensitive():
    w = make_world()
    assert w.find("apple") is not None
    assert w.find("Apple") is not None
    assert w.find("Apple_1") is not None
    assert w.find("banana") is None


def test_pickup_success():
    w = make_world()
    ok, reason = apply_pickup(w, "Apple_1")
    assert ok, reason
    assert w.inventory == "Apple_1"
    assert w.objects["apple"].parent is None


def test_pickup_missing_object():
    w = make_world()
    ok, reason = apply_pickup(w, "Banana_1")
    assert not ok
    assert "找不到" in reason


def test_pickup_hands_full():
    """手上已有物体不能再拿，这是 ALFRED 的硬规则。"""
    w = make_world()
    apply_pickup(w, "Apple_1")
    ok, reason = apply_pickup(w, "Bread_1")
    assert not ok
    assert "手上" in reason


def test_put_success():
    w = make_world()
    apply_pickup(w, "Apple_1")
    ok, reason = apply_put(w, "Apple_1", "Fridge_1")
    assert ok, reason
    assert w.inventory is None
    assert w.objects["apple"].parent == "Fridge_1"


def test_put_without_pickup_is_violation():
    """没拿到东西就要放下，物理上做不到。这类就是规划幻觉。"""
    w = make_world()
    ok, reason = apply_put(w, "Apple_1", "Fridge_1")
    assert not ok
    assert "手上" in reason


def test_put_wrong_object_in_hand():
    w = make_world()
    apply_pickup(w, "Apple_1")
    ok, reason = apply_put(w, "Bread_1", "Fridge_1")
    assert not ok
    assert "手上" in reason


def test_put_to_receptacle_not_in_objects():
    """容器可能没登记在 objects 里（固定器具），这种情况应当放行。"""
    w = make_world()
    apply_pickup(w, "Apple_1")
    ok, reason = apply_put(w, "Apple_1", "Microwave_1")
    assert ok, reason
    assert w.objects["apple"].parent == "Microwave_1"
