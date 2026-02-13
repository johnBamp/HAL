"""Grid navigation and LOS helpers for HAL agents."""

from __future__ import annotations

import math


Cell = tuple[int, int]


def _neighbors(cell: Cell, width: int, height: int) -> list[Cell]:
    x, y = cell
    out: list[Cell] = []
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < width and 0 <= ny < height:
            out.append((nx, ny))
    return out


def astar_path(
    start: Cell,
    goal: Cell,
    blocked: set[Cell],
    width: int,
    height: int,
) -> list[Cell] | None:
    """Return shortest 4-neighbor path from start to goal (inclusive)."""
    if start == goal:
        return [start]
    if goal in blocked and goal != start:
        return None

    open_set: set[Cell] = {start}
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, int] = {start: 0}
    f_score: dict[Cell, int] = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

    while open_set:
        current = min(open_set, key=lambda c: f_score.get(c, 10**9))
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        open_set.remove(current)
        for nxt in _neighbors(current, width, height):
            if nxt in blocked and nxt != goal:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(nxt, 10**9):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f_score[nxt] = tentative + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                open_set.add(nxt)

    return None


def select_reachable_target(
    start: Cell,
    candidates: list[Cell] | set[Cell],
    blocked: set[Cell],
    width: int,
    height: int,
    tie_break: str = "nearest",
) -> tuple[Cell | None, list[Cell] | None]:
    """Pick reachable candidate by shortest path length."""
    best_target: Cell | None = None
    best_path: list[Cell] | None = None

    sorted_candidates = sorted(set(candidates))
    for target in sorted_candidates:
        if target in blocked and target != start:
            continue
        path = astar_path(start, target, blocked, width, height)
        if path is None:
            continue

        if best_path is None:
            best_target = target
            best_path = path
            continue

        if tie_break == "farthest":
            is_better = len(path) > len(best_path)
        else:
            is_better = len(path) < len(best_path)

        if is_better:
            best_target = target
            best_path = path

    return best_target, best_path


def compute_visible_cells(
    origin_cell: Cell,
    rotation_deg: float,
    render_distance_tiles: int,
    fov_degrees: float,
    num_vision_rays: int,
    width: int,
    height: int,
    opaque: set[Cell] | None = None,
) -> set[Cell]:
    """Raycast visible cells using the same semantics as the pygame runtime.

    Stops rays when they hit opaque cells (walls).
    """
    cx, cy = origin_cell
    ox = cx + 0.5
    oy = cy + 0.5

    half_fov = fov_degrees / 2.0
    visible: set[Cell] = {(cx, cy)}

    for i in range(num_vision_rays):
        t = i / (num_vision_rays - 1) if num_vision_rays > 1 else 0.5
        angle_deg = rotation_deg - half_fov + t * fov_degrees
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        step = 0.05
        distance = 0.0
        while distance <= render_distance_tiles:
            sx = ox + dx * distance
            sy = oy + dy * distance
            tx = int(math.floor(sx))
            ty = int(math.floor(sy))

            if tx < 0 or ty < 0 or tx >= width or ty >= height:
                break

            visible.add((tx, ty))
            if opaque and (tx, ty) in opaque:
                break
            distance += step

    return visible
