from navigation import astar_path, compute_visible_cells, select_reachable_target


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
