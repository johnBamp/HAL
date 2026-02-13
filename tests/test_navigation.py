from navigation import astar_path, compute_visible_cells, select_reachable_target
from main import _adjacent_cells, _build_blocked_cells, _collect_find_candidates


def test_astar_finds_shortest_path_in_empty_grid():
    path = astar_path(
        start=(0, 0),
        goal=(2, 2),
        blocked=set(),
        width=4,
        height=4,
    )
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5


def test_astar_avoids_blocked_cells():
    blocked = {(1, 0), (1, 1), (1, 2)}
    path = astar_path(
        start=(0, 1),
        goal=(2, 1),
        blocked=blocked,
        width=3,
        height=4,
    )
    assert path is not None
    for cell in path:
        assert cell not in blocked


def test_select_reachable_target_prefers_nearest_path():
    target, path = select_reachable_target(
        start=(1, 1),
        candidates=[(2, 1), (4, 4)],
        blocked=set(),
        width=6,
        height=6,
        tie_break="nearest",
    )
    assert target == (2, 1)
    assert path is not None


def test_select_reachable_target_returns_non_los_candidate_for_avoid():
    start = (4, 4)
    visible = {(4, 4), (5, 4), (5, 5), (4, 5)}
    candidates = [
        (x, y)
        for y in range(9)
        for x in range(9)
        if (x, y) not in visible
    ]
    target, path = select_reachable_target(
        start=start,
        candidates=candidates,
        blocked=set(),
        width=9,
        height=9,
        tie_break="nearest",
    )
    assert target is not None
    assert target not in visible
    assert path is not None


def test_compute_visible_cells_contains_origin_and_forward_cell():
    visible = compute_visible_cells(
        origin_cell=(2, 2),
        rotation_deg=0.0,
        render_distance_tiles=3,
        fov_degrees=45.0,
        num_vision_rays=61,
        width=9,
        height=9,
    )
    assert (2, 2) in visible
    assert (3, 2) in visible


def test_find_candidate_filter_respects_outside_los():
    class Obj:
        def __init__(self, kind, color_name):
            self.kind = kind
            self.color_name = color_name

    objects_by_cell = {
        (1, 1): Obj("apple", "red"),
        (2, 2): Obj("apple", "red"),
        (3, 3): Obj("mushroom", "red"),
    }
    visible = {(1, 1), (4, 4)}
    candidates = _collect_find_candidates(
        objects_by_cell,
        target_color="red",
        target_object="apple",
        adam_visible=visible,
        outside_only=True,
    )
    assert (1, 1) not in candidates
    assert (2, 2) in candidates


def test_find_candidate_filter_empty_when_only_in_los_match():
    class Obj:
        def __init__(self, kind, color_name):
            self.kind = kind
            self.color_name = color_name

    objects_by_cell = {(1, 1): Obj("apple", "orange")}
    visible = {(1, 1)}
    candidates = _collect_find_candidates(
        objects_by_cell,
        target_color="orange",
        target_object="apple",
        adam_visible=visible,
        outside_only=True,
    )
    assert candidates == []


def test_return_target_selection_reaches_adam_adjacency():
    adam_cell = (3, 3)
    eve_cell = (0, 3)
    objects_by_cell = {(2, 3): object()}
    blocked = _build_blocked_cells(objects_by_cell, adam_cell, eve_cell)
    adjacent = [cell for cell in _adjacent_cells(adam_cell) if 0 <= cell[0] < 7 and 0 <= cell[1] < 7]
    candidates = [cell for cell in adjacent if cell not in blocked]

    target, path = select_reachable_target(
        start=eve_cell,
        candidates=candidates,
        blocked=blocked,
        width=7,
        height=7,
        tie_break="nearest",
    )
    assert target in candidates
    assert path is not None
    assert path[-1] == target
