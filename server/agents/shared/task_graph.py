"""Deterministic ordering for the tasks inside one repository's workstream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def ordered_task_ids(
    task_ids: Sequence[str], dependencies: Mapping[str, Sequence[str]]
) -> tuple[list[str], dict[str, list[str]]]:
    """Return a dependency-first order over `task_ids`, and the edges it could not honour.

    Stable by construction: tasks that become ready together keep the order they were declared
    in, so a workstream declaring no dependency at all comes back exactly as it was written.
    That is the whole of today's behaviour and it has to stay byte for byte what it was for
    every plan that declares none.

    The second value names the edges no order can satisfy -- a dependency on a task this
    workstream never declared, a task waiting on itself, or a set of tasks waiting on each
    other. They are reported rather than resolved: the planner drops and records them before
    an artifact exists, and the task plan's own validator refuses a cycle that reaches it
    anyway. Every declared task appears in the returned order exactly once regardless, because
    a caller that asked for an order must always receive a complete one.
    """
    declared = list(dict.fromkeys(task_ids))
    known = set(declared)
    requested = {
        task_id: list(dict.fromkeys(dependencies.get(task_id, ()))) for task_id in declared
    }
    unresolvable = {
        task_id: [item for item in items if item == task_id or item not in known]
        for task_id, items in requested.items()
    }
    unresolvable = {task_id: items for task_id, items in unresolvable.items() if items}
    waiting = {
        task_id: {item for item in items if item != task_id and item in known}
        for task_id, items in requested.items()
    }
    ordered: list[str] = []
    placed: set[str] = set()
    while len(placed) < len(declared):
        ready = [
            task_id for task_id in declared if task_id not in placed and waiting[task_id] <= placed
        ]
        if not ready:
            # Everything left is waiting on something else left, which is what a cycle is.
            # Each unmet edge is reported and the rest are placed in declaration order, so the
            # caller still receives every task and the cycle stays visible in the graph itself.
            for task_id in declared:
                if task_id in placed:
                    continue
                unresolvable.setdefault(task_id, []).extend(sorted(waiting[task_id] - placed))
                ordered.append(task_id)
            break
        ordered.extend(ready)
        placed.update(ready)
    return ordered, unresolvable


__all__ = ["ordered_task_ids"]
