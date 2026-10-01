"""Pure-function coverage for the task-batch ordering helper (agents/shared/task_graph.py)."""

from __future__ import annotations

from agents.shared.task_graph import ordered_task_ids


def test_an_empty_dependency_graph_returns_declaration_order_unchanged() -> None:
    """Every task plan ever persisted declares no task dependencies -- this is that shape."""
    order, unresolvable = ordered_task_ids(["a", "b", "c"], {})

    assert order == ["a", "b", "c"]
    assert unresolvable == {}


def test_a_diamond_graph_returns_a_dependency_first_order() -> None:
    """b and c both wait only on a, and both become ready together; d waits on both."""
    order, unresolvable = ordered_task_ids(
        ["a", "b", "c", "d"], {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    )

    assert order == ["a", "b", "c", "d"]
    assert unresolvable == {}


def test_a_dependency_on_an_undeclared_task_is_reported_and_not_honoured() -> None:
    """An edge naming a task outside the set is not a real constraint on this order."""
    order, unresolvable = ordered_task_ids(["a", "b"], {"b": ["nope"]})

    assert order == ["a", "b"]
    assert unresolvable == {"b": ["nope"]}


def test_a_task_depending_on_itself_is_reported_and_not_honoured() -> None:
    order, unresolvable = ordered_task_ids(["a"], {"a": ["a"]})

    assert order == ["a"]
    assert unresolvable == {"a": ["a"]}


def test_a_cycle_still_returns_every_task_exactly_once_with_the_unmet_edges_named() -> None:
    """A caller that asked for an order must always receive a complete one."""
    order, unresolvable = ordered_task_ids(["a", "b"], {"a": ["b"], "b": ["a"]})

    assert sorted(order) == ["a", "b"]
    assert len(order) == 2
    assert unresolvable == {"a": ["b"], "b": ["a"]}


def test_a_ten_task_chain_orders_every_task_before_the_one_that_depends_on_it() -> None:
    task_ids = [f"t{index}" for index in range(10)]
    dependencies = {task_ids[i]: [task_ids[i - 1]] for i in range(1, 10)}

    order, unresolvable = ordered_task_ids(task_ids, dependencies)

    assert order == task_ids
    assert unresolvable == {}
