"""符号状态机单测。

这些用例先于真实数据验证核心逻辑。gold 数据到手之后，
还要再跑 scripts/verify_gold.py 做一次全量校验。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.metrics import edit_distance, normalized_edit_distance, wilson_interval, cohens_kappa
from evalkit.simulator import (
    FailReason,
    parse_action_line,
    run_plan,
)

SCENE = [
    {"objectId": "Apple_1", "objectType": "Apple", "parentReceptacles": ["Table_1"]},
    {"objectId": "Bread_1", "objectType": "Bread", "parentReceptacles": ["Table_1"]},
    {"objectId": "Knife_1", "objectType": "Knife", "parentReceptacles": ["CounterTop_1"]},
    {"objectId": "Table_1", "objectType": "Table", "parentReceptacles": []},
    {"objectId": "CounterTop_1", "objectType": "CounterTop", "parentReceptacles": []},
    {"objectId": "Fridge_1", "objectType": "Fridge", "parentReceptacles": []},
    {"objectId": "Microwave_1", "objectType": "Microwave", "parentReceptacles": []},
]

PARAMS_PLACE = {"object_target": "Apple", "parent_target": "Fridge"}


def test_parse_two_formats():
    assert parse_action_line("PickupObject(Apple_1)") == ("PickupObject", ["Apple_1"])
    assert parse_action_line("PutObject(Apple_1, Fridge_1)") == ("PutObject", ["Apple_1", "Fridge_1"])
    assert parse_action_line("1. PickupObject(Apple_1)") == ("PickupObject", ["Apple_1"])
    assert parse_action_line("") == (None, [])


def test_correct_plan_achieves_goal():
    res = run_plan(
        ["PickupObject(Apple_1)", "GotoLocation(Fridge_1)", "PutObject(Apple_1, Fridge_1)"],
        SCENE,
        PARAMS_PLACE,
    )
    assert res.goal_achieved
    assert res.violations == 0
    assert res.executed == 3


def test_put_without_pickup_is_violation():
    """没拿到东西就要放下，物理上做不到。这类就是规划幻觉。"""
    res = run_plan(["PutObject(Apple_1, Fridge_1)"], SCENE, PARAMS_PLACE)
    assert not res.goal_achieved
    assert res.violations == 1
    assert res.steps[0].reason == FailReason.NOT_HOLDING


def test_pickup_twice_is_violation():
    res = run_plan(["PickupObject(Apple_1)", "PickupObject(Bread_1)"], SCENE, PARAMS_PLACE)
    assert res.violations == 1
    assert res.steps[1].reason == FailReason.HANDS_FULL


def test_heat_requires_microwave():
    """加热要求手持物体且站在微波炉前，只拿不移动就该被抓。"""
    res = run_plan(["PickupObject(Bread_1)", "HeatObject(Bread_1)"], SCENE, PARAMS_PLACE)
    assert res.violations == 1
    assert res.steps[1].reason == FailReason.WRONG_LOCATION


def test_heat_after_goto_microwave_succeeds():
    res = run_plan(
        ["PickupObject(Bread_1)", "GotoLocation(Microwave_1)", "HeatObject(Bread_1)"],
        SCENE,
        PARAMS_PLACE,
    )
    assert res.violations == 0
    bread = res.final_state.find("Bread_1")
    assert bread.heated


def test_slice_requires_knife():
    res = run_plan(["PickupObject(Bread_1)", "SliceObject(Bread_1)"], SCENE, PARAMS_PLACE)
    assert res.violations == 1
    assert res.steps[1].reason == FailReason.NEED_KNIFE


def test_slice_with_knife_succeeds():
    res = run_plan(
        ["PickupObject(Knife_1)", "SliceObject(Bread_1)"],
        SCENE,
        {"object_target": "Bread", "parent_target": "Table", "object_sliced": True},
    )
    # 拿刀切片这一步不报错，但目标要求放到桌上，此时还没放
    assert res.violations == 0
    assert res.final_state.find("Bread_1").sliced


def test_unknown_action_is_violation():
    res = run_plan(["FlyToMars(Apple_1)"], SCENE, PARAMS_PLACE)
    assert res.violations == 1
    assert res.steps[0].reason == FailReason.UNKNOWN_ACTION


def test_stop_on_violation_halts():
    """违反之后不再推演，避免用错误状态继续算。"""
    res = run_plan(
        ["PutObject(Apple_1, Fridge_1)", "PickupObject(Apple_1)"],
        SCENE,
        PARAMS_PLACE,
        stop_on_violation=True,
    )
    assert len(res.steps) == 1


def test_wilson_interval_bounds():
    iv = wilson_interval(50, 100)
    assert 0.0 <= iv.low <= iv.point <= iv.high <= 1.0


def test_kappa_perfect():
    a = [True, False, True]
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-9


def test_edit_distance():
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    assert edit_distance(["a"], ["a", "b"]) == 1
    assert normalized_edit_distance(["a", "b", "c"], ["a", "b"]) == 1 / 3
