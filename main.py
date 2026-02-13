import argparse
import math
import random
import sys

import settings
from hal import (
    CommandTrainingConfig,
    LanguageAgent,
    TrainingConfig,
    backtrace_factored_phrase,
    best_word,
    build_slot_id_maps,
    color_key,
    format_word,
    initialize_language,
    initialize_hidden_command_mapping,
    object_key,
    sweep_social_pressures,
    train_command_grounding,
    train_factored_language,
    train_language,
    write_lexicon_logs,
)
from navigation import compute_visible_cells, select_reachable_target

VISUAL_TILE_WIDTH = settings.LOGICAL_TILE_WIDTH * settings.VISUAL_SCALE
VISUAL_TILE_HEIGHT = settings.LOGICAL_TILE_HEIGHT * settings.VISUAL_SCALE

GRID_STRIDE_X = VISUAL_TILE_WIDTH + settings.TILE_GAP
GRID_STRIDE_Y = VISUAL_TILE_HEIGHT + settings.TILE_GAP

VIEW_PIXEL_WIDTH = settings.VIEW_TILES_X * VISUAL_TILE_WIDTH + (settings.VIEW_TILES_X - 1) * settings.TILE_GAP
VIEW_PIXEL_HEIGHT = settings.VIEW_TILES_Y * VISUAL_TILE_HEIGHT + (settings.VIEW_TILES_Y - 1) * settings.TILE_GAP
GRID_PIXEL_WIDTH = VIEW_PIXEL_WIDTH
GRID_PIXEL_HEIGHT = VIEW_PIXEL_HEIGHT

GRID_LEFT = settings.ORIGIN_X
GRID_TOP = settings.ORIGIN_Y + settings.INVENTORY_BAR_HEIGHT

MAP_WIDTH = settings.ORIGIN_X * 2 + VIEW_PIXEL_WIDTH
MAP_HEIGHT = settings.ORIGIN_Y + settings.INVENTORY_BAR_HEIGHT + VIEW_PIXEL_HEIGHT + settings.ORIGIN_Y
SCREEN_WIDTH = MAP_WIDTH + settings.PANEL_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT

INVENTORY_SLOT_SIZE = 56
INVENTORY_SLOT_GAP = 14

COLOR_WHEEL = [
    ("red", (196, 40, 38)),
    ("orange", (228, 128, 40)),
    ("yellow", (220, 184, 48)),
    ("green", (56, 168, 74)),
    ("blue", (54, 114, 214)),
    ("indigo", (76, 64, 176)),
    ("violet", (142, 76, 204)),
]
COLOR_RGB = {name: rgb for name, rgb in COLOR_WHEEL}

DEFAULT_OBJECT_TYPES = list(settings.LEXICON_OBJECT_TYPES)
DEFAULT_COLOR_NAMES = [name for name in settings.LEXICON_COLOR_NAMES if name in COLOR_RGB]
DEFAULT_MODE = "factored"


class WorldAgent:
    def __init__(self, name, color, language):
        self.name = name
        self.color = color
        self.language = language
        self.cell = None
        self.rotation_deg = 0.0
        self.vision_slice = []
        self.subjective_world = {}
        self.subjective_prior = {}
        self.subjective_classes = []
        self.subjective_observed = set()
        self.subjective_last_seen = {}


class Tile:
    def __init__(self, pygame, grid_x, grid_y, is_wall=False):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.is_wall = is_wall

        self.floor_surface = pygame.Surface((VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT), pygame.SRCALPHA)
        self.floor_surface.fill((0, 0, 0, 0))
        pygame.draw.rect(
            self.floor_surface,
            (64, 68, 76, 255),
            pygame.Rect(0, 0, VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT),
            width=1,
        )

        self.wall_surface = pygame.Surface((VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT))
        self.wall_surface.fill((58, 50, 40))
        pygame.draw.rect(
            self.wall_surface,
            (84, 74, 60),
            pygame.Rect(0, 0, VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT),
            width=1,
        )
        inner = pygame.Rect(3, 3, max(2, VISUAL_TILE_WIDTH - 6), max(2, VISUAL_TILE_HEIGHT - 6))
        pygame.draw.rect(self.wall_surface, (44, 38, 30), inner)
        pygame.draw.rect(self.wall_surface, (70, 62, 50), inner, width=1)
        pygame.draw.line(self.wall_surface, (34, 30, 24), (2, 2), (VISUAL_TILE_WIDTH - 3, VISUAL_TILE_HEIGHT - 3), 1)
        pygame.draw.line(self.wall_surface, (34, 30, 24), (VISUAL_TILE_WIDTH - 3, 2), (2, VISUAL_TILE_HEIGHT - 3), 1)

    def draw(self, pygame, screen, camera_origin, fill_color=None):
        tile_x, tile_y = cell_to_screen(self.grid_x, self.grid_y, camera_origin)
        if self.is_wall:
            tile_surface = self.wall_surface.copy()
            if fill_color is not None:
                mixed = tuple((base + tint) // 2 for base, tint in zip((58, 50, 40), fill_color))
                tile_surface.fill(mixed)
                inner = pygame.Rect(3, 3, max(2, VISUAL_TILE_WIDTH - 6), max(2, VISUAL_TILE_HEIGHT - 6))
                pygame.draw.rect(tile_surface, tuple(max(0, c - 18) for c in mixed), inner)
                pygame.draw.rect(tile_surface, tuple(min(255, c + 10) for c in mixed), inner, width=1)
                pygame.draw.rect(tile_surface, tuple(min(255, c + 18) for c in mixed), pygame.Rect(0, 0, VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT), width=1)
        else:
            tile_surface = self.floor_surface.copy()
            edge = fill_color if fill_color is not None else (64, 68, 76)
            center = tuple(max(0, min(255, int(c * 0.22))) for c in edge)
            inner = pygame.Rect(3, 3, max(2, VISUAL_TILE_WIDTH - 6), max(2, VISUAL_TILE_HEIGHT - 6))
            pygame.draw.rect(tile_surface, (center[0], center[1], center[2], 90), inner)
            pygame.draw.rect(tile_surface, edge, pygame.Rect(0, 0, VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT), width=1)

        screen.blit(tile_surface, (tile_x, tile_y))


class WorldObject:
    def __init__(self, kind, color_name, cell, concept, sprite):
        self.kind = kind
        self.color_name = color_name
        self.cell = cell
        self.concept = concept
        self.sprite = sprite

    def draw(self, pygame, screen, camera_origin):
        tile_x, tile_y = cell_to_screen(*self.cell, camera_origin)
        scaled = pygame.transform.scale(self.sprite, (VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT))
        screen.blit(scaled, (tile_x, tile_y))


def concept_key(object_type, color_name):
    return f"{color_name}_{object_type}"


def concept_label(object_type, color_name):
    return f"{color_name.title()} {object_type.title()}"


def is_backtrace_enabled(mode):
    return mode == "factored"


def camera_limits():
    max_x = max(0, settings.TILE_COUNT_X - settings.VIEW_TILES_X)
    max_y = max(0, settings.TILE_COUNT_Y - settings.VIEW_TILES_Y)
    return max_x, max_y


def clamp_camera_origin(camera_origin):
    max_x, max_y = camera_limits()
    cx = max(0, min(max_x, int(camera_origin[0])))
    cy = max(0, min(max_y, int(camera_origin[1])))
    return (cx, cy)


def pan_camera(camera_origin, dx, dy):
    return clamp_camera_origin((camera_origin[0] + dx, camera_origin[1] + dy))


def center_camera_on_cell(cell):
    target_x = cell[0] - (settings.VIEW_TILES_X // 2)
    target_y = cell[1] - (settings.VIEW_TILES_Y // 2)
    return clamp_camera_origin((target_x, target_y))


def viewport_bounds(camera_origin):
    x0 = camera_origin[0]
    y0 = camera_origin[1]
    x1 = min(settings.TILE_COUNT_X, x0 + settings.VIEW_TILES_X)
    y1 = min(settings.TILE_COUNT_Y, y0 + settings.VIEW_TILES_Y)
    return x0, y0, x1, y1


def is_in_viewport(cell, camera_origin):
    x0, y0, x1, y1 = viewport_bounds(camera_origin)
    return x0 <= cell[0] < x1 and y0 <= cell[1] < y1


def cell_to_screen(grid_x, grid_y, camera_origin=(0, 0)):
    x = GRID_LEFT + (grid_x - camera_origin[0]) * GRID_STRIDE_X
    y = GRID_TOP + (grid_y - camera_origin[1]) * GRID_STRIDE_Y
    return x, y


def screen_to_cell(px, py, camera_origin=(0, 0)):
    rel_x = px - GRID_LEFT
    rel_y = py - GRID_TOP
    if rel_x < 0 or rel_y < 0:
        return None

    gx = rel_x // GRID_STRIDE_X + camera_origin[0]
    gy = rel_y // GRID_STRIDE_Y + camera_origin[1]
    if gx >= settings.TILE_COUNT_X or gy >= settings.TILE_COUNT_Y or gx < 0 or gy < 0:
        return None

    if (rel_x % GRID_STRIDE_X) >= VISUAL_TILE_WIDTH:
        return None
    if (rel_y % GRID_STRIDE_Y) >= VISUAL_TILE_HEIGHT:
        return None

    return int(gx), int(gy)


def _count_neighboring_walls(x, y, walls):
    count = 0
    for ny in range(y - 1, y + 2):
        for nx in range(x - 1, x + 2):
            if nx == x and ny == y:
                continue
            if not is_in_bounds(nx, ny):
                count += 1
                continue
            if (nx, ny) in walls:
                count += 1
    return count


def _smooth_cave_walls(walls):
    next_walls = set()
    for y in range(settings.TILE_COUNT_Y):
        for x in range(settings.TILE_COUNT_X):
            cell = (x, y)
            border = x == 0 or y == 0 or x == settings.TILE_COUNT_X - 1 or y == settings.TILE_COUNT_Y - 1
            if border:
                next_walls.add(cell)
                continue
            neighbors = _count_neighboring_walls(x, y, walls)
            if cell in walls:
                if neighbors >= settings.CAVE_SURVIVE_LIMIT:
                    next_walls.add(cell)
            elif neighbors > settings.CAVE_BIRTH_LIMIT:
                next_walls.add(cell)
    return next_walls


def _largest_walkable_component(walkable_cells):
    if not walkable_cells:
        return set()

    remaining = set(walkable_cells)
    best = set()
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            cx, cy = stack.pop()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                nxt = (nx, ny)
                if nxt in remaining:
                    remaining.remove(nxt)
                    component.add(nxt)
                    stack.append(nxt)
        if len(component) > len(best):
            best = component
    return best


def generate_cave_layout(rng=None):
    rng = rng or random.Random()
    all_cells = {
        (x, y)
        for y in range(settings.TILE_COUNT_Y)
        for x in range(settings.TILE_COUNT_X)
    }

    walls = set()
    for y in range(settings.TILE_COUNT_Y):
        for x in range(settings.TILE_COUNT_X):
            cell = (x, y)
            border = x == 0 or y == 0 or x == settings.TILE_COUNT_X - 1 or y == settings.TILE_COUNT_Y - 1
            if border or rng.random() < settings.CAVE_FILL_PROB:
                walls.add(cell)

    for _ in range(max(0, settings.CAVE_SMOOTH_STEPS)):
        walls = _smooth_cave_walls(walls)

    walkable = all_cells - walls
    walkable = _largest_walkable_component(walkable)

    if len(walkable) < 2:
        walkable = {
            (x, y)
            for y in range(1, max(1, settings.TILE_COUNT_Y - 1))
            for x in range(1, max(1, settings.TILE_COUNT_X - 1))
        }
    walls = all_cells - walkable
    return walkable, walls


def shade_color(rgb, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def mix_with_white(rgb, amount):
    return tuple(max(0, min(255, int(c + (255 - c) * amount))) for c in rgb)


def to_rgba(rgb, alpha=255):
    return (rgb[0], rgb[1], rgb[2], alpha)


def build_apple_sprite(pygame, accent_rgb):
    sprite = pygame.Surface((settings.LOGICAL_TILE_WIDTH, settings.LOGICAL_TILE_HEIGHT), pygame.SRCALPHA)

    body = to_rgba(accent_rgb)
    shadow = to_rgba(shade_color(accent_rgb, 0.62))
    highlight = to_rgba(mix_with_white(accent_rgb, 0.58), 220)
    stem = (80, 52, 22, 255)
    leaf = (88, 186, 92, 255)

    pygame.draw.circle(sprite, shadow, (7, 9), 5)
    pygame.draw.circle(sprite, shadow, (10, 9), 5)
    pygame.draw.circle(sprite, body, (7, 8), 5)
    pygame.draw.circle(sprite, body, (10, 8), 5)
    pygame.draw.rect(sprite, body, pygame.Rect(5, 8, 7, 4))
    pygame.draw.circle(sprite, highlight, (6, 6), 2)
    pygame.draw.rect(sprite, stem, pygame.Rect(8, 2, 2, 3))
    pygame.draw.ellipse(sprite, leaf, pygame.Rect(9, 2, 4, 3))
    pygame.draw.ellipse(sprite, (56, 130, 65, 255), pygame.Rect(10, 3, 2, 2))
    return sprite


def build_mushroom_sprite(pygame, accent_rgb):
    sprite = pygame.Surface((settings.LOGICAL_TILE_WIDTH, settings.LOGICAL_TILE_HEIGHT), pygame.SRCALPHA)

    cap = to_rgba(accent_rgb)
    cap_shadow = to_rgba(shade_color(accent_rgb, 0.58))
    cap_highlight = to_rgba(mix_with_white(accent_rgb, 0.48), 220)
    stem = (244, 224, 193, 255)
    stem_shadow = (205, 172, 138, 255)
    spot = (252, 246, 236, 255)

    pygame.draw.ellipse(sprite, cap_shadow, pygame.Rect(2, 4, 12, 7))
    pygame.draw.ellipse(sprite, cap, pygame.Rect(2, 3, 12, 7))
    pygame.draw.ellipse(sprite, cap_highlight, pygame.Rect(4, 4, 5, 2))

    pygame.draw.ellipse(sprite, stem_shadow, pygame.Rect(5, 8, 6, 6))
    pygame.draw.ellipse(sprite, stem, pygame.Rect(5, 7, 6, 6))

    pygame.draw.circle(sprite, spot, (5, 6), 1)
    pygame.draw.circle(sprite, spot, (8, 5), 1)
    pygame.draw.circle(sprite, spot, (10, 6), 1)
    pygame.draw.line(sprite, stem_shadow, (6, 10), (10, 10), 1)
    return sprite


def get_sprite(pygame, sprite_cache, kind, color_name):
    key = (kind, color_name)
    if key in sprite_cache:
        return sprite_cache[key]

    accent = COLOR_RGB.get(color_name, COLOR_RGB["red"])
    if kind == "apple":
        sprite = build_apple_sprite(pygame, accent)
    elif kind == "mushroom":
        sprite = build_mushroom_sprite(pygame, accent)
    else:
        sprite = pygame.Surface((settings.LOGICAL_TILE_WIDTH, settings.LOGICAL_TILE_HEIGHT), pygame.SRCALPHA)

    sprite_cache[key] = sprite
    return sprite


def generate_map(pygame, wall_cells):
    tiles = {}
    for y in range(settings.TILE_COUNT_Y):
        for x in range(settings.TILE_COUNT_X):
            cell = (x, y)
            tiles[cell] = Tile(pygame, x, y, is_wall=(cell in wall_cells))
    return tiles


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def choose_agent_cells(agents, walkable_cells, rng=None):
    if len(agents) < 2:
        return

    walkable = list(walkable_cells)
    if not walkable:
        raise ValueError("No walkable cells available for agent spawn.")

    rng = rng or random.Random()
    sample_count = min(96, len(walkable))
    sample_cells = rng.sample(walkable, sample_count) if sample_count < len(walkable) else walkable

    best_pair = (walkable[0], walkable[0])
    best_dist = -1
    for src in sample_cells:
        dst = max(walkable, key=lambda cell: _manhattan(src, cell))
        dist = _manhattan(src, dst)
        if dist > best_dist:
            best_pair = (src, dst)
            best_dist = dist

    agents[0].cell = best_pair[0]
    agents[1].cell = best_pair[1]

    dx = agents[1].cell[0] - agents[0].cell[0]
    dy = agents[1].cell[1] - agents[0].cell[1]
    agents[0].rotation_deg = math.degrees(math.atan2(dy, dx))
    agents[1].rotation_deg = math.degrees(math.atan2(-dy, -dx))


def is_in_bounds(x, y):
    return 0 <= x < settings.TILE_COUNT_X and 0 <= y < settings.TILE_COUNT_Y


def raycast_visible_cells(agent, opaque=None):
    return compute_visible_cells(
        origin_cell=agent.cell,
        rotation_deg=agent.rotation_deg,
        render_distance_tiles=settings.RENDER_DISTANCE_TILES,
        fov_degrees=settings.FOV_DEGREES,
        num_vision_rays=settings.NUM_VISION_RAYS,
        width=settings.TILE_COUNT_X,
        height=settings.TILE_COUNT_Y,
        opaque=opaque,
    )


def build_vision_slice(agent, visible_cells, occupancy, objects_by_cell):
    size = settings.RENDER_DISTANCE_TILES * 2 + 1
    center = settings.RENDER_DISTANCE_TILES
    vision = [[None for _ in range(size)] for _ in range(size)]

    ax, ay = agent.cell
    for dy in range(-settings.RENDER_DISTANCE_TILES, settings.RENDER_DISTANCE_TILES + 1):
        for dx in range(-settings.RENDER_DISTANCE_TILES, settings.RENDER_DISTANCE_TILES + 1):
            wx = ax + dx
            wy = ay + dy
            lx = center + dx
            ly = center + dy

            if not is_in_bounds(wx, wy):
                vision[ly][lx] = {"in_bounds": False}
                continue

            is_visible = (wx, wy) in visible_cells
            object_at_cell = objects_by_cell.get((wx, wy))
            vision[ly][lx] = {
                "in_bounds": True,
                "world": (wx, wy),
                "visible": is_visible,
                "occupied_by": occupancy.get((wx, wy)),
                "object_type": object_at_cell.kind if object_at_cell else None,
                "object_color": object_at_cell.color_name if object_at_cell else None,
                "object_concept": object_at_cell.concept if object_at_cell else None,
                "object_cell": object_at_cell.cell if object_at_cell else None,
            }

    agent.vision_slice = vision


def classify_vision_cells(vision_slice):
    for row in vision_slice:
        for cell in row:
            categories = []
            if not cell.get("in_bounds", False):
                categories.append("empty")
                cell["categories"] = categories
                cell["catagories"] = categories
                continue

            has_object = cell.get("object_type") is not None
            has_agent = cell.get("occupied_by") is not None

            categories.append("full" if has_object or has_agent else "empty")

            obj_type = cell.get("object_type")
            color_name = cell.get("object_color")
            if obj_type:
                categories.append(obj_type)
            if color_name:
                categories.append(color_name)

            cell["categories"] = categories
            cell["catagories"] = categories


def draw_agent(pygame, screen, agent, camera_origin):
    gx, gy = agent.cell
    tile_x, tile_y = cell_to_screen(gx, gy, camera_origin)

    margin = max(2, VISUAL_TILE_WIDTH // 6)
    agent_rect = pygame.Rect(
        tile_x + margin,
        tile_y + margin,
        VISUAL_TILE_WIDTH - margin * 2,
        VISUAL_TILE_HEIGHT - margin * 2,
    )
    pygame.draw.rect(screen, agent.color, agent_rect, border_radius=4)

    cx = tile_x + VISUAL_TILE_WIDTH // 2
    cy = tile_y + VISUAL_TILE_HEIGHT // 2
    length = max(6, VISUAL_TILE_WIDTH // 2 - margin)
    ang = math.radians(agent.rotation_deg)
    ex = int(cx + math.cos(ang) * length)
    ey = int(cy + math.sin(ang) * length)
    pygame.draw.line(screen, (255, 255, 255), (cx, cy), (ex, ey), 2)


def draw_carried_object(pygame, screen, carrier, carried_object, sprite_cache, camera_origin):
    if carried_object is None:
        return
    tile_x, tile_y = cell_to_screen(*carrier.cell, camera_origin)
    sprite = get_sprite(pygame, sprite_cache, carried_object.kind, carried_object.color_name)
    size = max(10, int(min(VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT) * 0.56))
    scaled = pygame.transform.scale(sprite, (size, size))
    offset_x = VISUAL_TILE_WIDTH - size - 2
    offset_y = 2
    screen.blit(scaled, (tile_x + offset_x, tile_y + offset_y))


def draw_speech_bubble(pygame, screen, small_font, text, anchor_x, anchor_y, fill=(242, 246, 252), fg=(22, 26, 34)):
    if not text:
        return

    pad_x = 7
    pad_y = 4
    text_surf = small_font.render(text, True, fg)
    bubble_w = text_surf.get_width() + pad_x * 2
    bubble_h = text_surf.get_height() + pad_y * 2
    bubble_x = anchor_x - bubble_w // 2
    bubble_y = anchor_y - bubble_h - 10
    bubble_rect = pygame.Rect(bubble_x, bubble_y, bubble_w, bubble_h)

    pygame.draw.rect(screen, fill, bubble_rect, border_radius=7)
    pygame.draw.rect(screen, (82, 92, 110), bubble_rect, width=1, border_radius=7)
    tip = [(anchor_x - 5, bubble_rect.bottom), (anchor_x + 5, bubble_rect.bottom), (anchor_x, anchor_y - 3)]
    pygame.draw.polygon(screen, fill, tip)
    pygame.draw.polygon(screen, (82, 92, 110), tip, width=1)
    screen.blit(text_surf, (bubble_rect.x + pad_x, bubble_rect.y + pad_y))


def _decode_eve_command_action(command_state, token):
    row = command_state["eve_action_q"].get(token)
    if not row:
        return "approach"
    ordered_actions = sorted(row.keys())
    return max(ordered_actions, key=lambda action: row[action])


def _build_blocked_cells(objects_by_cell, adam_cell, eve_cell, wall_cells=None, passable_cells=None):
    blocked = set(objects_by_cell.keys())
    if wall_cells:
        blocked |= set(wall_cells)
    if passable_cells:
        passable = set(passable_cells)
        if wall_cells:
            passable -= set(wall_cells)
        blocked -= passable
    blocked.add(adam_cell)
    blocked.discard(eve_cell)
    return blocked


def _all_grid_cells():
    return [
        (x, y)
        for y in range(settings.TILE_COUNT_Y)
        for x in range(settings.TILE_COUNT_X)
    ]


def _normalize_distribution(distribution):
    total = sum(max(0.0, value) for value in distribution.values())
    if total <= 1e-12:
        count = max(1, len(distribution))
        uniform = 1.0 / count
        return {key: uniform for key in distribution}
    return {key: max(0.0, value) / total for key, value in distribution.items()}


def _concept_universe(color_names, object_types):
    return [concept_key(object_type, color_name) for object_type in object_types for color_name in color_names]


def _initialize_subjective_world(agent, color_names, object_types):
    concept_classes = _concept_universe(color_names, object_types)
    classes = concept_classes + ["__none__"]
    prior = 1.0 / max(1, len(classes))

    agent.subjective_classes = classes
    agent.subjective_prior = {label: prior for label in classes}
    agent.subjective_world = {cell: dict(agent.subjective_prior) for cell in _all_grid_cells()}
    agent.subjective_observed = set()
    agent.subjective_last_seen = {cell: -10 ** 9 for cell in _all_grid_cells()}


def _bayes_observation_update(prior, observed_label, classes):
    hit = settings.BELIEF_OBS_HIT
    none_hit = settings.BELIEF_OBS_NONE

    if observed_label == "__none__":
        remaining = max(0.0, 1.0 - none_hit)
        per_other = remaining / max(1, len(classes) - 1)
        likelihood = {label: (none_hit if label == "__none__" else per_other) for label in classes}
    else:
        remaining = max(0.0, 1.0 - hit)
        none_share = min(remaining, 1.0 - hit)
        other_count = max(1, len(classes) - 2)
        per_other = max(0.0, remaining - none_share) / other_count
        likelihood = {}
        for label in classes:
            if label == observed_label:
                likelihood[label] = hit
            elif label == "__none__":
                likelihood[label] = none_share
            else:
                likelihood[label] = per_other

    posterior = {label: prior.get(label, 0.0) * likelihood[label] for label in classes}
    return _normalize_distribution(posterior)


def _observe_subjective_cell(agent, cell, observed_label):
    prior = agent.subjective_world.get(cell, dict(agent.subjective_prior))
    posterior = _bayes_observation_update(prior, observed_label, agent.subjective_classes)
    agent.subjective_world[cell] = posterior
    agent.subjective_observed.add(cell)


def _predict_subjective_world(agent, visible_cells, current_frame):
    drift = settings.BELIEF_DRIFT
    if drift <= 0:
        return
    for cell, distribution in agent.subjective_world.items():
        if cell in visible_cells:
            continue
        age = current_frame - agent.subjective_last_seen.get(cell, -10 ** 9)
        stale = age >= settings.BELIEF_STALE_AFTER_FRAMES
        stale_revert = settings.BELIEF_STALE_REVERT if stale else 0.0
        for label in agent.subjective_classes:
            base = distribution[label]
            toward_prior = agent.subjective_prior[label]
            value = (1.0 - drift) * base + (drift * toward_prior)
            if stale_revert > 0:
                value = (1.0 - stale_revert) * value + stale_revert * toward_prior
            distribution[label] = value


def _update_subjective_world(agent, visible_cells, objects_by_cell, current_frame):
    if not agent.subjective_world:
        return
    _predict_subjective_world(agent, visible_cells, current_frame)
    for cell in visible_cells:
        observed = objects_by_cell.get(cell)
        if observed is None:
            observed_label = "__none__"
        else:
            observed_label = concept_key(observed.kind, observed.color_name)
        _observe_subjective_cell(agent, cell, observed_label)
        agent.subjective_last_seen[cell] = current_frame


def _belief_probability(agent, cell, concept_label):
    distribution = agent.subjective_world.get(cell)
    if not distribution:
        return 0.0
    return distribution.get(concept_label, 0.0)


def _belief_candidates_for_concept(agent, concept_label, adam_visible, adam_cell, outside_only, current_frame):
    scored = []
    for cell in agent.subjective_observed:
        if outside_only and cell in adam_visible:
            continue
        if cell == adam_cell:
            continue
        probability = _belief_probability(agent, cell, concept_label)
        if probability < settings.BELIEF_TARGET_THRESHOLD:
            continue
        none_prob = agent.subjective_world[cell].get("__none__", 0.0)
        age = current_frame - agent.subjective_last_seen.get(cell, -10 ** 9)
        stale_bonus = 0.12 if age >= settings.BELIEF_STALE_AFTER_FRAMES else 0.0
        score = probability - (0.35 * none_prob) + stale_bonus
        scored.append((score, cell))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [cell for _, cell in scored]


def _frontier_targets(agent, blocked_cells, adam_visible, outside_only, current_frame):
    frontiers = []
    stale_first = []
    for cell in _all_grid_cells():
        if cell == agent.cell:
            continue
        in_observed = cell in agent.subjective_observed
        if in_observed:
            age = current_frame - agent.subjective_last_seen.get(cell, -10 ** 9)
            if age < settings.BELIEF_STALE_AFTER_FRAMES:
                continue
        if cell in blocked_cells:
            continue
        if outside_only and cell in adam_visible:
            continue
        for neighbor in _adjacent_cells(cell):
            if not is_in_bounds(*neighbor):
                continue
            if neighbor in agent.subjective_observed:
                if in_observed:
                    stale_first.append(cell)
                else:
                    frontiers.append(cell)
                break
    return stale_first + frontiers


def _adjacent_cells(cell):
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _max_fetch_steps():
    # Scale fetch budget with map size so a single command can finish on large caves.
    area_scaled = (settings.TILE_COUNT_X + settings.TILE_COUNT_Y) * 10
    return max(settings.COMMAND_MAX_FETCH_STEPS, area_scaled)


def _collect_find_candidates(objects_by_cell, target_color, target_object, adam_visible, outside_only):
    candidates = []
    for cell, obj in objects_by_cell.items():
        if obj.kind != target_object or obj.color_name != target_color:
            continue
        if outside_only and cell in adam_visible:
            continue
        candidates.append(cell)
    return candidates


def _choose_eve_target_path(action, eve_cell, adam_cell, adam_visible, blocked_cells):
    if action == "approach":
        candidates = [cell for cell in adam_visible if cell != adam_cell and cell not in blocked_cells]
    else:
        candidates = [cell for cell in _all_grid_cells() if cell not in adam_visible and cell not in blocked_cells]

    return select_reachable_target(
        start=eve_cell,
        candidates=candidates,
        blocked=blocked_cells,
        width=settings.TILE_COUNT_X,
        height=settings.TILE_COUNT_Y,
        tie_break="nearest",
    )


def _finalize_command_feedback(command_state, reward, eve_in_los, outcome):
    command_state["last_reward"] = reward
    command_state["last_visibility"] = "in_los" if eve_in_los else "out_los"
    command_state["last_outcome"] = outcome
    command_state["feedback_text"] = "GOOD" if reward > 0 else "BAD"
    command_state["feedback_timer"] = settings.FEEDBACK_BUBBLE_FRAMES
    command_state["active_path"] = []
    command_state["path_steps"] = 0
    command_state["fetch_steps"] = 0
    command_state["move_cooldown"] = 0
    command_state["current_intent"] = None
    command_state["current_action"] = None
    command_state["current_phase"] = "idle"
    command_state["current_target_cell"] = None
    command_state["return_target_cell"] = None
    command_state["search_mode"] = None
    command_state["pending_find_target"] = None
    command_state["carried_object"] = None
    command_state["command_active"] = False


def _finalize_visibility_command(command_state, eve_in_los):
    desired_in_los = command_state.get("current_intent") == "come"
    reward = 1 if eve_in_los == desired_in_los else -1
    outcome = "match" if reward > 0 else "mismatch"
    _finalize_command_feedback(command_state, reward=reward, eve_in_los=eve_in_los, outcome=outcome)


def _finalize_find_command(command_state, eve_in_los, success, outcome):
    reward = 1 if success else -1
    _finalize_command_feedback(command_state, reward=reward, eve_in_los=eve_in_los, outcome=outcome)


def _format_find_speech(command_state, command_token, color_token, object_token, color_name, object_name, mode):
    if mode == "factored" and color_token is not None and object_token is not None:
        return f"[{format_word(command_token)}, {format_word(color_token)}, {format_word(object_token)}]"
    return f"[{format_word(command_token)}, {color_name.title()}, {object_name.title()}]"


def _decode_find_target(command_state, eve, color_token, object_token, mode):
    fallback_color = command_state.get("current_target_color")
    fallback_object = command_state.get("current_target_object")
    if not fallback_color or not fallback_object:
        command_state["decoded_target_label"] = "UNKNOWN"
        command_state["decoded_target_quality"] = "invalid"
        command_state["decoded_target_notes"] = "missing fallback target selection"
        return None, None

    if mode != "factored" or color_token is None or object_token is None:
        command_state["decoded_target_label"] = concept_label(fallback_object, fallback_color)
        command_state["decoded_target_quality"] = "direct"
        command_state["decoded_target_notes"] = ""
        return fallback_color, fallback_object

    decoded = backtrace_factored_phrase(
        eve.language,
        color_token=f"{color_token:04d}",
        object_token=f"{object_token:04d}",
        color_names=command_state["all_colors"],
        object_types=command_state["all_objects"],
        seen_pairs=command_state["seen_pairs"],
    )
    if decoded.decoded_color_label == "UNKNOWN" or decoded.decoded_object_label == "UNKNOWN":
        command_state["decoded_target_label"] = "UNKNOWN"
        command_state["decoded_target_quality"] = decoded.decode_quality
        command_state["decoded_target_notes"] = decoded.notes
        return None, None

    color_name = decoded.decoded_color_label.strip().lower()
    object_name = decoded.decoded_object_label.strip().lower()
    command_state["decoded_target_label"] = decoded.decoded_concept_label
    command_state["decoded_target_quality"] = decoded.decode_quality
    command_state["decoded_target_notes"] = decoded.notes
    return color_name, object_name


def _maybe_drop_carried_object(command_state, objects_by_cell, eve, adam, wall_cells):
    carried = command_state.get("carried_object")
    if carried is None:
        return
    if eve.cell == adam.cell:
        return
    if eve.cell in wall_cells:
        return
    if eve.cell in objects_by_cell:
        return
    carried.cell = eve.cell
    objects_by_cell[eve.cell] = carried
    command_state["carried_object"] = None


def _scan_surroundings(agent, objects_by_cell, wall_cells):
    # Rotate through configured headings to update subjective beliefs without moving.
    union_visible = set()
    original_heading = agent.rotation_deg
    for heading in settings.SCAN_ROTATIONS:
        agent.rotation_deg = heading
        union_visible |= raycast_visible_cells(agent, opaque=wall_cells)
    agent.rotation_deg = original_heading
    _update_subjective_world(agent, union_visible, objects_by_cell, current_frame=agent.subjective_last_seen.get("__clock__", 0))


def _begin_return_phase(command_state, adam, eve, objects_by_cell, wall_cells):
    adjacent = [cell for cell in _adjacent_cells(adam.cell) if is_in_bounds(*cell)]
    blocked_cells = _build_blocked_cells(objects_by_cell, adam.cell, eve.cell, wall_cells=wall_cells)
    candidates = [cell for cell in adjacent if cell not in blocked_cells]
    if not candidates:
        return False

    target, path = select_reachable_target(
        start=eve.cell,
        candidates=candidates,
        blocked=blocked_cells,
        width=settings.TILE_COUNT_X,
        height=settings.TILE_COUNT_Y,
        tie_break="nearest",
    )
    if target is None or path is None:
        return False

    command_state["current_phase"] = "carrying_return"
    command_state["return_target_cell"] = target
    command_state["active_path"] = path[1:] if len(path) > 1 else []
    return True


def _begin_fetch_phase(command_state, adam, eve, adam_visible, objects_by_cell, wall_cells, mode):
    target_color = command_state.get("current_target_color")
    target_object = command_state.get("current_target_object")

    color_token = None
    object_token = None
    if mode == "factored":
        c_sem = color_key(target_color)
        o_sem = object_key(target_object)
        if c_sem in adam.language.q_table and o_sem in adam.language.q_table:
            color_token = best_word(adam.language, c_sem)
            object_token = best_word(adam.language, o_sem)

    command_state["spoken_text"] = _format_find_speech(
        command_state,
        command_state["find_token"],
        color_token,
        object_token,
        target_color,
        target_object,
        mode,
    )

    decoded_color, decoded_object = _decode_find_target(
        command_state,
        eve=eve,
        color_token=color_token,
        object_token=object_token,
        mode=mode,
    )
    if not decoded_color or not decoded_object:
        return False, "decode_failed"

    command_state["current_target_color"] = decoded_color
    command_state["current_target_object"] = decoded_object
    target_concept = concept_key(decoded_object, decoded_color)

    outside_preferences = [settings.COMMAND_FIND_TARGET_OUTSIDE_LOS]
    if settings.COMMAND_FIND_TARGET_OUTSIDE_LOS:
        outside_preferences.append(False)

    for outside_only in outside_preferences:
        direct_candidates = _collect_find_candidates(
            objects_by_cell,
            target_color=decoded_color,
            target_object=decoded_object,
            adam_visible=adam_visible,
            outside_only=outside_only,
        )
        direct_candidates = [cell for cell in direct_candidates if cell not in wall_cells]
        if direct_candidates:
            blocked_direct = _build_blocked_cells(
                objects_by_cell,
                adam.cell,
                eve.cell,
                wall_cells=wall_cells,
                passable_cells=direct_candidates,
            )
            target, path = select_reachable_target(
                start=eve.cell,
                candidates=direct_candidates,
                blocked=blocked_direct,
                width=settings.TILE_COUNT_X,
                height=settings.TILE_COUNT_Y,
                tie_break="nearest",
            )
            if target is not None and path is not None:
                command_state["current_phase"] = "navigate_to_target"
                command_state["search_mode"] = "direct_outside_los" if outside_only else "direct_anywhere"
                command_state["current_target_cell"] = target
                command_state["active_path"] = path[1:] if len(path) > 1 else []
                command_state["fetch_steps"] = 0
                return True, ""

        blocked_base = _build_blocked_cells(objects_by_cell, adam.cell, eve.cell, wall_cells=wall_cells)
        belief_candidates = _belief_candidates_for_concept(
            eve,
            concept_label=target_concept,
            adam_visible=adam_visible,
            adam_cell=adam.cell,
            outside_only=outside_only,
            current_frame=command_state["frame"],
        )
        belief_candidates = [cell for cell in belief_candidates if cell not in wall_cells]
        if belief_candidates:
            blocked_cells = _build_blocked_cells(
                objects_by_cell,
                adam.cell,
                eve.cell,
                wall_cells=wall_cells,
                passable_cells=belief_candidates,
            )
            target, path = select_reachable_target(
                start=eve.cell,
                candidates=belief_candidates,
                blocked=blocked_cells,
                width=settings.TILE_COUNT_X,
                height=settings.TILE_COUNT_Y,
                tie_break="nearest",
            )
            if target is not None and path is not None:
                command_state["current_phase"] = "navigate_to_target"
                command_state["search_mode"] = "belief_outside_los" if outside_only else "belief_anywhere"
                command_state["current_target_cell"] = target
                command_state["active_path"] = path[1:] if len(path) > 1 else []
                command_state["fetch_steps"] = 0
                if settings.SCAN_AT_FRONTIER:
                    _scan_surroundings(eve, objects_by_cell, wall_cells)
                return True, ""

        frontier_cells = _frontier_targets(
            eve,
            blocked_cells=blocked_base,
            adam_visible=adam_visible,
            outside_only=outside_only,
            current_frame=command_state["frame"],
        )
        if not frontier_cells:
            continue

        frontier_target, frontier_path = select_reachable_target(
            start=eve.cell,
            candidates=frontier_cells,
            blocked=blocked_base,
            width=settings.TILE_COUNT_X,
            height=settings.TILE_COUNT_Y,
            tie_break="nearest",
        )
        if frontier_target is None or frontier_path is None:
            continue

        command_state["current_phase"] = "searching"
        command_state["search_mode"] = "frontier_outside_los" if outside_only else "frontier_anywhere"
        command_state["current_target_cell"] = frontier_target
        command_state["active_path"] = frontier_path[1:] if len(frontier_path) > 1 else []
        command_state["fetch_steps"] = 0
        return True, ""

    return False, "search_exhausted"


def _start_command_execution(command_state, token, adam, eve, adam_visible, objects_by_cell, wall_cells, rng):
    command_state["command_active"] = True
    command_state["spoken_token"] = token
    command_state["speech_timer"] = settings.SPEECH_BUBBLE_FRAMES
    command_state["current_intent"] = command_state["token_to_intent"].get(token, "come")
    command_state["current_action"] = _decode_eve_command_action(command_state, token)
    command_state["path_steps"] = 0
    command_state["fetch_steps"] = 0
    command_state["decoded_target_label"] = "-"
    command_state["decoded_target_quality"] = "n/a"
    command_state["decoded_target_notes"] = ""

    if command_state["current_intent"] == "find":
        if command_state["pending_find_target"] is not None:
            target_color, target_object = command_state["pending_find_target"]
            command_state["pending_find_target"] = None
        else:
            target_color = rng.choice(command_state["all_colors"])
            target_object = rng.choice(command_state["all_objects"])
        command_state["current_target_color"] = target_color
        command_state["current_target_object"] = target_object

        if command_state["current_action"] != "fetch":
            command_state["spoken_text"] = _format_find_speech(
                command_state,
                command_state["find_token"],
                None,
                None,
                target_color,
                target_object,
                command_state["mode"],
            )
            _finalize_find_command(command_state, eve.cell in adam_visible, False, "mismatch")
            return

        ok, failure = _begin_fetch_phase(
            command_state=command_state,
            adam=adam,
            eve=eve,
            adam_visible=adam_visible,
            objects_by_cell=objects_by_cell,
            wall_cells=wall_cells,
            mode=command_state["mode"],
        )
        if not ok:
            _finalize_find_command(command_state, eve.cell in adam_visible, False, failure)
            return

        if command_state["current_phase"] == "navigate_to_target" and eve.cell == command_state["current_target_cell"]:
            observed_obj = objects_by_cell.get(eve.cell)
            expected_color = command_state.get("current_target_color")
            expected_object = command_state.get("current_target_object")
            if observed_obj is not None:
                observed_label = concept_key(observed_obj.kind, observed_obj.color_name)
            else:
                observed_label = "__none__"
            _observe_subjective_cell(eve, eve.cell, observed_label)

            target_match = (
                observed_obj is not None
                and observed_obj.kind == expected_object
                and observed_obj.color_name == expected_color
            )
            if not target_match:
                ok, failure = _begin_fetch_phase(
                    command_state=command_state,
                    adam=adam,
                    eve=eve,
                    adam_visible=adam_visible,
                    objects_by_cell=objects_by_cell,
                    wall_cells=wall_cells,
                    mode=command_state["mode"],
                )
                if not ok:
                    _finalize_find_command(command_state, eve.cell in adam_visible, False, failure)
                return

            target_obj = objects_by_cell.pop(eve.cell, None)
            if target_obj is None:
                _finalize_find_command(command_state, eve.cell in adam_visible, False, "missing_target")
                return
            command_state["carried_object"] = target_obj
            if not _begin_return_phase(command_state, adam, eve, objects_by_cell, wall_cells):
                _maybe_drop_carried_object(command_state, objects_by_cell, eve, adam, wall_cells)
                _finalize_find_command(command_state, eve.cell in adam_visible, False, "no_path_to_return")
                return
            if command_state["current_phase"] == "carrying_return" and not command_state["active_path"]:
                carried = command_state.get("carried_object")
                if carried is not None and eve.cell not in objects_by_cell and eve.cell != adam.cell:
                    carried.cell = eve.cell
                    objects_by_cell[eve.cell] = carried
                    command_state["carried_object"] = None
                    _finalize_find_command(command_state, eve.cell in adam_visible, True, "fetched")
                    return
    else:
        command_state["spoken_text"] = format_word(token)
        blocked_cells = _build_blocked_cells(objects_by_cell, adam.cell, eve.cell, wall_cells=wall_cells)
        _, path = _choose_eve_target_path(
            command_state["current_action"],
            eve.cell,
            adam.cell,
            adam_visible,
            blocked_cells,
        )
        if path and len(path) > 1:
            command_state["active_path"] = path[1:]
        else:
            command_state["active_path"] = []
            _finalize_visibility_command(command_state, eve.cell in adam_visible)


def update_command_state(command_state, adam, eve, adam_visible, objects_by_cell, wall_cells, rng):
    command_state["frame"] += 1
    eve.subjective_last_seen["__clock__"] = command_state["frame"]
    adam.subjective_last_seen["__clock__"] = command_state["frame"]

    if command_state["speech_timer"] > 0:
        command_state["speech_timer"] -= 1
    if command_state["feedback_timer"] > 0:
        command_state["feedback_timer"] -= 1

    if command_state["move_cooldown"] > 0:
        command_state["move_cooldown"] -= 1

    def handle_find_no_path():
        phase = command_state.get("current_phase")
        if phase == "navigate_to_target":
            observed_obj = objects_by_cell.get(eve.cell)
            expected_color = command_state.get("current_target_color")
            expected_object = command_state.get("current_target_object")
            if observed_obj is not None:
                observed_label = concept_key(observed_obj.kind, observed_obj.color_name)
            else:
                observed_label = "__none__"
            _observe_subjective_cell(eve, eve.cell, observed_label)
            eve.subjective_last_seen[eve.cell] = command_state["frame"]

            target_match = (
                observed_obj is not None
                and observed_obj.kind == expected_object
                and observed_obj.color_name == expected_color
            )
            if not target_match:
                ok, failure = _begin_fetch_phase(
                    command_state=command_state,
                    adam=adam,
                    eve=eve,
                    adam_visible=adam_visible,
                    objects_by_cell=objects_by_cell,
                    wall_cells=wall_cells,
                    mode=command_state["mode"],
                )
                if not ok:
                    _finalize_find_command(command_state, eve.cell in adam_visible, False, failure)
                return

            target_obj = objects_by_cell.pop(eve.cell, None)
            if target_obj is None:
                _finalize_find_command(command_state, eve.cell in adam_visible, False, "missing_target")
                return
            command_state["carried_object"] = target_obj
            if not _begin_return_phase(command_state, adam, eve, objects_by_cell, wall_cells):
                _maybe_drop_carried_object(command_state, objects_by_cell, eve, adam, wall_cells)
                _finalize_find_command(command_state, eve.cell in adam_visible, False, "no_path_to_return")
                return
            if not command_state["active_path"]:
                carried = command_state.get("carried_object")
                if carried is not None and eve.cell not in objects_by_cell and eve.cell != adam.cell:
                    carried.cell = eve.cell
                    objects_by_cell[eve.cell] = carried
                    command_state["carried_object"] = None
                    _finalize_find_command(command_state, eve.cell in adam_visible, True, "fetched")
                else:
                    _finalize_find_command(command_state, eve.cell in adam_visible, False, "drop_failed")
            return

        if phase == "searching":
            if settings.SCAN_AT_FRONTIER:
                _scan_surroundings(eve, objects_by_cell, wall_cells)
            ok, failure = _begin_fetch_phase(
                command_state=command_state,
                adam=adam,
                eve=eve,
                adam_visible=adam_visible,
                objects_by_cell=objects_by_cell,
                wall_cells=wall_cells,
                mode=command_state["mode"],
            )
            if not ok:
                _finalize_find_command(command_state, eve.cell in adam_visible, False, failure)
            return

        if phase == "carrying_return":
            carried = command_state.get("carried_object")
            if carried is None:
                _finalize_find_command(command_state, eve.cell in adam_visible, False, "missing_carry")
                return
            if eve.cell == adam.cell or eve.cell in objects_by_cell:
                adj = [cell for cell in _adjacent_cells(adam.cell) if is_in_bounds(*cell)]
                blocked = set(objects_by_cell.keys()) | {adam.cell} | set(wall_cells)
                free = [c for c in adj if c not in blocked]
                if free:
                    carried.cell = free[0]
                    objects_by_cell[free[0]] = carried
                    command_state["carried_object"] = None
                    _finalize_find_command(command_state, eve.cell in adam_visible, True, "fetched")
                    return
                _maybe_drop_carried_object(command_state, objects_by_cell, eve, adam, wall_cells)
                _finalize_find_command(command_state, eve.cell in adam_visible, False, "drop_failed")
                return
            carried.cell = eve.cell
            objects_by_cell[eve.cell] = carried
            command_state["carried_object"] = None
            _finalize_find_command(command_state, eve.cell in adam_visible, True, "fetched")
            return

        _finalize_find_command(command_state, eve.cell in adam_visible, False, "stalled")

    if command_state["active_path"]:
        if command_state["move_cooldown"] == 0:
            next_cell = command_state["active_path"].pop(0)
            dx = next_cell[0] - eve.cell[0]
            dy = next_cell[1] - eve.cell[1]
            if dx != 0 or dy != 0:
                eve.rotation_deg = math.degrees(math.atan2(dy, dx))
            eve.cell = next_cell
            command_state["path_steps"] += 1
            if command_state.get("current_intent") == "find":
                command_state["fetch_steps"] += 1
                command_state["move_cooldown"] = settings.COMMAND_FETCH_MOVE_COOLDOWN
            else:
                command_state["move_cooldown"] = 2

            # quick scan every few steps to refresh belief during motion
            if command_state["path_steps"] % max(1, len(settings.SCAN_ROTATIONS)) == 0:
                _scan_surroundings(eve, objects_by_cell, wall_cells)

            if command_state.get("current_intent") == "find":
                if command_state["fetch_steps"] >= _max_fetch_steps():
                    _maybe_drop_carried_object(command_state, objects_by_cell, eve, adam, wall_cells)
                    _finalize_find_command(command_state, eve.cell in adam_visible, False, "max_steps")
                    return

            if command_state.get("current_intent") == "find":
                if not command_state["active_path"]:
                    handle_find_no_path()
                    return
            else:
                if (
                    not command_state["active_path"]
                    or command_state["path_steps"] >= settings.MAX_COMMAND_PATH_STEPS
                ):
                    _finalize_visibility_command(command_state, eve.cell in adam_visible)
        return

    if command_state.get("command_active"):
        if command_state.get("current_intent") == "find":
            handle_find_no_path()
        else:
            _finalize_visibility_command(command_state, eve.cell in adam_visible)
        return

    token = None
    if command_state["pending_override"] is not None:
        token = command_state["pending_override"]
        command_state["pending_override"] = None
    else:
        frames_since_issue = command_state["frame"] - command_state["last_issue_frame"]
        if command_state["auto_enabled"] and frames_since_issue >= settings.COMMAND_INTERVAL_FRAMES:
            if rng.random() < settings.COMMAND_AUTO_FIND_RATE:
                token = command_state["find_token"]
                command_state["pending_find_target"] = (
                    rng.choice(command_state["all_colors"]),
                    rng.choice(command_state["all_objects"]),
                )
            else:
                token = rng.choice([command_state["come_token"], command_state["leave_token"]])

    if token is None:
        return

    command_state["last_issue_frame"] = command_state["frame"]
    _start_command_execution(command_state, token, adam, eve, adam_visible, objects_by_cell, wall_cells, rng)


def build_palette_slots(pygame, object_types):
    slots = []
    start_x = GRID_LEFT + 12
    y = settings.ORIGIN_Y + 18
    for idx, obj_type in enumerate(object_types):
        x = start_x + idx * (INVENTORY_SLOT_SIZE + INVENTORY_SLOT_GAP)
        rect = pygame.Rect(x, y, INVENTORY_SLOT_SIZE, INVENTORY_SLOT_SIZE)
        slots.append({"kind": obj_type, "rect": rect})
    return slots


def get_color_dropdown_geometry(pygame, palette_slots, color_names):
    if palette_slots:
        x = palette_slots[-1]["rect"].right + 24
    else:
        x = GRID_LEFT + 12
    y = settings.ORIGIN_Y + 22
    width = 176
    row_h = 28

    header_rect = pygame.Rect(x, y, width, row_h)
    option_rects = {
        color_name: pygame.Rect(x, y + row_h * (idx + 1), width, row_h)
        for idx, color_name in enumerate(color_names)
    }
    return header_rect, option_rects


def draw_color_dropdown(pygame, screen, small_font, palette_slots, color_names, selected_color, dropdown_open):
    header_rect, option_rects = get_color_dropdown_geometry(pygame, palette_slots, color_names)

    pygame.draw.rect(screen, (36, 41, 50), header_rect, border_radius=7)
    pygame.draw.rect(screen, (98, 106, 120), header_rect, width=2, border_radius=7)

    label = f"Color: {selected_color.title()}"
    label_surf = small_font.render(label, True, (224, 231, 242))
    screen.blit(label_surf, (header_rect.x + 10, header_rect.y + 6))

    swatch = COLOR_RGB[selected_color]
    pygame.draw.circle(screen, swatch, (header_rect.right - 34, header_rect.centery), 7)
    pygame.draw.circle(screen, (236, 236, 240), (header_rect.right - 34, header_rect.centery), 7, width=1)

    arrow = "v" if not dropdown_open else "^"
    arrow_surf = small_font.render(arrow, True, (220, 226, 238))
    screen.blit(arrow_surf, (header_rect.right - 16, header_rect.y + 6))

    if dropdown_open:
        for color_name in color_names:
            rect = option_rects[color_name]
            bg = (44, 48, 58) if color_name != selected_color else (58, 66, 80)
            pygame.draw.rect(screen, bg, rect, border_radius=6)
            pygame.draw.rect(screen, (98, 106, 120), rect, width=1, border_radius=6)

            pygame.draw.circle(screen, COLOR_RGB[color_name], (rect.x + 12, rect.centery), 6)
            pygame.draw.circle(screen, (236, 236, 240), (rect.x + 12, rect.centery), 6, width=1)

            txt = small_font.render(color_name.title(), True, (224, 231, 242))
            screen.blit(txt, (rect.x + 24, rect.y + 6))


def get_delete_zone_rect(pygame):
    width = 116
    height = 30
    x = MAP_WIDTH - width - 14
    y = settings.ORIGIN_Y + 6
    return pygame.Rect(x, y, width, height)


def draw_inventory_bar(
    pygame,
    screen,
    font,
    small_font,
    palette_slots,
    sprite_cache,
    color_names,
    selected_color,
    dropdown_open,
    dragging_active=False,
    delete_hover=False,
):
    bar_rect = pygame.Rect(0, 0, MAP_WIDTH, GRID_TOP)
    pygame.draw.rect(screen, (14, 17, 22), bar_rect)
    left = GRID_LEFT - 2
    right = GRID_LEFT + GRID_PIXEL_WIDTH + 2
    pygame.draw.line(screen, (68, 78, 92), (left, GRID_TOP - 1), (right, GRID_TOP - 1), 2)

    title = font.render("Object Bar", True, (235, 240, 252))
    screen.blit(title, (GRID_LEFT + 6, settings.ORIGIN_Y + 2))

    delete_rect = get_delete_zone_rect(pygame)
    delete_bg = (88, 46, 46) if delete_hover else (56, 34, 34)
    delete_border = (196, 96, 96) if dragging_active else (122, 82, 82)
    pygame.draw.rect(screen, delete_bg, delete_rect, border_radius=6)
    pygame.draw.rect(screen, delete_border, delete_rect, width=2, border_radius=6)
    delete_label = "Drop Here To Delete"
    delete_text = small_font.render(delete_label, True, (240, 214, 214))
    dx = delete_rect.x + (delete_rect.width - delete_text.get_width()) // 2
    dy = delete_rect.y + (delete_rect.height - delete_text.get_height()) // 2
    screen.blit(delete_text, (dx, dy))

    for slot in palette_slots:
        kind = slot["kind"]
        rect = slot["rect"]
        pygame.draw.rect(screen, (36, 41, 50), rect, border_radius=7)
        pygame.draw.rect(screen, (98, 106, 120), rect, width=2, border_radius=7)

        sprite = get_sprite(pygame, sprite_cache, kind, selected_color)
        scaled = pygame.transform.scale(sprite, (INVENTORY_SLOT_SIZE - 16, INVENTORY_SLOT_SIZE - 16))
        sx = rect.x + (rect.width - scaled.get_width()) // 2
        sy = rect.y + (rect.height - scaled.get_height()) // 2
        screen.blit(scaled, (sx, sy))

        label = small_font.render(kind.title(), True, (220, 226, 238))
        label_y = rect.y + rect.height + 6
        screen.blit(label, (rect.x + 2, label_y))

    draw_color_dropdown(pygame, screen, small_font, palette_slots, color_names, selected_color, dropdown_open)


def _format_cell(cell):
    if cell is None:
        return "--"
    return f"({cell[0]},{cell[1]})"


def compute_drag_perception(
    dragging_kind,
    dragging_color,
    drag_pos,
    agents,
    visible_by_agent,
    mode,
    camera_origin,
    walkable_cells,
):
    if dragging_kind is None or dragging_color is None:
        return {
            "kind": None,
            "color_name": None,
            "concept": None,
            "hover_cell": None,
            "preview_cell": None,
            "rows": [],
            "syntax": "color_object",
        }

    hover_cell = screen_to_cell(*drag_pos, camera_origin)
    preview_cell = None
    if (
        hover_cell is not None
        and hover_cell in walkable_cells
        and not any(agent.cell == hover_cell for agent in agents)
    ):
        preview_cell = hover_cell

    rows = []
    for agent in agents:
        visible = visible_by_agent.get(agent.name, set())
        sees_object = preview_cell is not None and preview_cell in visible

        if sees_object:
            if mode == "factored":
                c_sem = color_key(dragging_color)
                o_sem = object_key(dragging_kind)
                if c_sem in agent.language.q_table and o_sem in agent.language.q_table:
                    c_word = format_word(best_word(agent.language, c_sem))
                    o_word = format_word(best_word(agent.language, o_sem))
                    utterance = f"[{c_word}, {o_word}]"
                else:
                    utterance = "----"
            else:
                concept = concept_key(dragging_kind, dragging_color)
                if concept in agent.language.q_table:
                    utterance = format_word(best_word(agent.language, concept))
                else:
                    utterance = "----"
        else:
            utterance = "----"

        rows.append(
            {
                "agent_name": agent.name,
                "sees_object": sees_object,
                "utterance": utterance,
            }
        )

    return {
        "kind": dragging_kind,
        "color_name": dragging_color,
        "concept": concept_key(dragging_kind, dragging_color),
        "hover_cell": hover_cell,
        "preview_cell": preview_cell,
        "rows": rows,
        "syntax": "color_object",
    }


def get_lexicon_panel_layout():
    panel_x = MAP_WIDTH
    panel_inner_x = panel_x + settings.PANEL_MARGIN
    panel_inner_w = settings.PANEL_WIDTH - 2 * settings.PANEL_MARGIN

    section_top = 84
    gap = 10
    perception_h = 220
    bottom_pad = 8

    available = SCREEN_HEIGHT - section_top - perception_h - gap - bottom_pad
    if available < 220:
        slot_h = max(90, (available - gap) // 2)
        phrase_h = max(90, available - gap - slot_h)
    else:
        slot_h = max(120, min(180, int(available * 0.38)))
        phrase_h = max(120, available - gap - slot_h)

    slot_rect = (panel_inner_x, section_top, panel_inner_w, slot_h)
    phrase_rect = (panel_inner_x, section_top + slot_h + gap, panel_inner_w, phrase_h)
    perception_rect = (panel_inner_x, section_top + slot_h + gap + phrase_h + gap, panel_inner_w, perception_h)

    return {
        "panel_x": panel_x,
        "slot_rect": slot_rect,
        "phrase_rect": phrase_rect,
        "perception_rect": perception_rect,
    }


def point_in_rect(pos, rect):
    x, y = pos
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def get_inspector_controls():
    layout = get_lexicon_panel_layout()
    px, py, pw, _ = layout["perception_rect"]

    tab_y = py + 6
    controls_y = py + 34
    row2_y = controls_y + 30

    return {
        "perception_tab": (px + 8, tab_y, 96, 22),
        "backtrace_tab": (px + 110, tab_y, 96, 22),
        "command_tab": (px + 212, tab_y, 96, 22),
        "mode_toggle": (px + 8, controls_y, 128, 22),
        "autofill": (px + pw - 204, controls_y, 116, 22),
        "run": (px + pw - 82, controls_y, 72, 22),
        "color_prev": (px + 8, row2_y, 22, 22),
        "color_next": (px + 158, row2_y, 22, 22),
        "color_label": (px + 34, row2_y, 120, 22),
        "object_prev": (px + 186, row2_y, 22, 22),
        "object_next": (px + 336, row2_y, 22, 22),
        "object_label": (px + 212, row2_y, 120, 22),
        "token_color": (px + 8, row2_y + 30, 86, 22),
        "token_object": (px + 104, row2_y + 30, 86, 22),
        "cmd_auto": (px + 8, controls_y, 70, 22),
        "cmd_force_a": (px + 84, controls_y, 88, 22),
        "cmd_force_b": (px + 178, controls_y, 88, 22),
        "cmd_force_c": (px + 272, controls_y, 126, 22),
        "cmd_color_prev": (px + 8, row2_y, 22, 22),
        "cmd_color_label": (px + 34, row2_y, 130, 22),
        "cmd_color_next": (px + 168, row2_y, 22, 22),
        "cmd_object_prev": (px + 196, row2_y, 22, 22),
        "cmd_object_label": (px + 222, row2_y, 130, 22),
        "cmd_object_next": (px + 356, row2_y, 22, 22),
    }


def _slot_options(slot_rows, slot_type, fallback_values):
    values = [row.get("slot_label", "").strip().lower() for row in slot_rows if row.get("slot_type") == slot_type]
    values = [value for value in values if value]
    if values:
        return list(dict.fromkeys(values))
    return list(fallback_values)


def _cycle_selected(values, current_value, direction):
    if not values:
        return current_value
    if current_value not in values:
        return values[0]
    idx = values.index(current_value)
    idx = (idx + direction) % len(values)
    return values[idx]


def _semantic_tokens_from_selection(agent, color_name, object_name, color_options, object_options):
    slot_maps = build_slot_id_maps(agent, color_options, object_options)
    color_token = slot_maps["color_label_to_token"].get(color_name)
    object_token = slot_maps["object_label_to_token"].get(object_name)
    if color_token is None or object_token is None:
        return "", ""
    return f"{color_token:04d}", f"{object_token:04d}"


def _build_backtrace_rows(
    agents,
    color_token,
    object_token,
    color_options,
    object_options,
    seen_pairs,
):
    rows = []
    for agent in agents:
        result = backtrace_factored_phrase(
            agent.language,
            color_token=color_token,
            object_token=object_token,
            color_names=color_options,
            object_types=object_options,
            seen_pairs=seen_pairs,
        )
        rows.append(
            {
                "agent_name": agent.name,
                "color_token_input": result.color_token_input,
                "object_token_input": result.object_token_input,
                "decoded_concept_label": result.decoded_concept_label,
                "decoded_concept_key": result.decoded_concept_key,
                "seen_in_training": result.seen_in_training,
                "decode_quality": result.decode_quality,
                "notes": result.notes,
            }
        )
    return rows


def _to_rect(pygame, rect_tuple):
    return pygame.Rect(rect_tuple[0], rect_tuple[1], rect_tuple[2], rect_tuple[3])


def _draw_tab_button(pygame, screen, rect, label, active, small_font):
    bg = (62, 80, 108) if active else (34, 40, 52)
    fg = (232, 239, 249) if active else (188, 204, 228)
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    pygame.draw.rect(screen, (108, 122, 148), rect, width=1, border_radius=6)
    text = small_font.render(label, True, fg)
    tx = rect.x + (rect.width - text.get_width()) // 2
    ty = rect.y + (rect.height - text.get_height()) // 2
    screen.blit(text, (tx, ty))


def draw_lexicon_panel(
    pygame,
    screen,
    font,
    small_font,
    mode,
    slot_rows,
    phrase_rows,
    metrics,
    drag_perception,
    inspector_tab,
    backtrace_state,
    command_state,
    slot_scroll_px,
    phrase_scroll_px,
):
    layout = get_lexicon_panel_layout()
    panel_x = layout["panel_x"]

    panel_rect = pygame.Rect(panel_x, 0, settings.PANEL_WIDTH, SCREEN_HEIGHT)
    pygame.draw.rect(screen, (18, 18, 24), panel_rect)
    pygame.draw.line(screen, (70, 70, 90), (panel_x, 0), (panel_x, SCREEN_HEIGHT), 2)

    title = font.render("Lexicon", True, (235, 235, 245))
    screen.blit(title, (panel_x + settings.PANEL_MARGIN, 12))

    mode_text = f"Mode: {mode}"
    metrics_text = (
        f"Target unique: {metrics.get('target_unique', 'n/a')}  "
        f"Achieved: {metrics.get('achieved_unique', 'n/a')}"
    )
    consensus_text = f"Phrase consensus: {metrics.get('all_phrase_consensus', 'n/a')}"

    screen.blit(small_font.render(mode_text, True, (185, 205, 230)), (panel_x + settings.PANEL_MARGIN, 36))
    screen.blit(small_font.render(metrics_text, True, (185, 205, 230)), (panel_x + settings.PANEL_MARGIN, 52))
    screen.blit(small_font.render(consensus_text, True, (185, 205, 230)), (panel_x + settings.PANEL_MARGIN, 68))

    slot_rect = pygame.Rect(*layout["slot_rect"])
    phrase_rect = pygame.Rect(*layout["phrase_rect"])
    perception_rect = pygame.Rect(*layout["perception_rect"])

    pygame.draw.rect(screen, (12, 12, 16), slot_rect, border_radius=6)
    pygame.draw.rect(screen, (12, 12, 16), phrase_rect, border_radius=6)
    pygame.draw.rect(screen, (12, 12, 16), perception_rect, border_radius=6)

    # Section A: slot lexicon.
    slot_title = small_font.render("Base Factored Lexicon", True, (185, 205, 230))
    screen.blit(slot_title, (slot_rect.x + 8, slot_rect.y + 8))

    slot_headers = ["Type", "Semantic", "Adam", "Eve", "Shared"]
    slot_x = [8, 66, 210, 266, 320]
    base_y = slot_rect.y + 30
    for i, header in enumerate(slot_headers):
        txt = small_font.render(header, True, (166, 186, 210))
        screen.blit(txt, (slot_rect.x + slot_x[i], base_y))

    slot_rows_area = pygame.Rect(slot_rect.x + 4, slot_rect.y + 52, slot_rect.width - 10, slot_rect.height - 58)
    slot_row_h = 22
    slot_content_h = len(slot_rows) * slot_row_h
    slot_max_scroll = max(0, slot_content_h - slot_rows_area.height)
    slot_scroll_px = max(0, min(slot_scroll_px, slot_max_scroll))

    clip_prev = screen.get_clip()
    screen.set_clip(slot_rows_area)
    row_y = slot_rows_area.y - slot_scroll_px

    if slot_rows:
        for row in slot_rows:
            shared = row.get("shared_word", "NO_CONSENSUS")
            shared_color = (164, 228, 172) if shared != "NO_CONSENSUS" else (235, 156, 156)
            values = [
                row.get("slot_type", "-"),
                row.get("slot_label", row.get("concept_label", "-")),
                row.get("adam_word", "----"),
                row.get("eve_word", "----"),
                shared,
            ]
            colors = [(222, 222, 232), (222, 222, 232), (222, 222, 232), (222, 222, 232), shared_color]
            for i, text in enumerate(values):
                txt = small_font.render(text, True, colors[i])
                screen.blit(txt, (slot_rect.x + slot_x[i], row_y))
            row_y += slot_row_h
    else:
        txt = small_font.render("No slot lexicon in holistic mode.", True, (154, 166, 182))
        screen.blit(txt, (slot_rect.x + 8, row_y))

    screen.set_clip(clip_prev)

    if slot_max_scroll > 0:
        bar_x = slot_rect.right - 6
        bar_y = slot_rows_area.y
        bar_h = slot_rows_area.height
        pygame.draw.rect(screen, (60, 60, 80), pygame.Rect(bar_x, bar_y, 4, bar_h))
        thumb_h = max(22, int(bar_h * (slot_rows_area.height / slot_content_h)))
        thumb_range = bar_h - thumb_h
        thumb_y = bar_y + int((slot_scroll_px / slot_max_scroll) * thumb_range)
        pygame.draw.rect(screen, (150, 150, 180), pygame.Rect(bar_x, thumb_y, 4, thumb_h))

    # Section B: composed phrase lexicon.
    phrase_title = small_font.render("Composed Phrases", True, (185, 205, 230))
    screen.blit(phrase_title, (phrase_rect.x + 8, phrase_rect.y + 8))

    phrase_headers = ["Phrase", "Adam IDs", "Eve IDs", "Shared IDs", "Step"]
    phrase_x = [8, 150, 248, 336, 398]
    phrase_base_y = phrase_rect.y + 30
    for i, header in enumerate(phrase_headers):
        txt = small_font.render(header, True, (166, 186, 210))
        screen.blit(txt, (phrase_rect.x + phrase_x[i], phrase_base_y))

    phrase_rows_area = pygame.Rect(phrase_rect.x + 4, phrase_rect.y + 52, phrase_rect.width - 10, phrase_rect.height - 58)
    phrase_row_h = 22
    phrase_content_h = len(phrase_rows) * phrase_row_h
    phrase_max_scroll = max(0, phrase_content_h - phrase_rows_area.height)
    phrase_scroll_px = max(0, min(phrase_scroll_px, phrase_max_scroll))

    clip_prev = screen.get_clip()
    screen.set_clip(phrase_rows_area)
    row_y = phrase_rows_area.y - phrase_scroll_px

    for row in phrase_rows:
        shared = row.get("shared_phrase_ids", row.get("shared_word", "NO_CONSENSUS"))
        shared_color = (164, 228, 172) if shared != "NO_CONSENSUS" else (235, 156, 156)
        values = [
            row.get("concept_label", "-"),
            row.get("adam_phrase_ids", row.get("adam_word", "----")),
            row.get("eve_phrase_ids", row.get("eve_word", "----")),
            shared,
            str(row.get("converged_step", "NONE")),
        ]
        colors = [(222, 222, 232), (222, 222, 232), (222, 222, 232), shared_color, (150, 170, 190)]
        for i, text in enumerate(values):
            txt = small_font.render(text, True, colors[i])
            screen.blit(txt, (phrase_rect.x + phrase_x[i], row_y))
        row_y += phrase_row_h

    screen.set_clip(clip_prev)

    if phrase_max_scroll > 0:
        bar_x = phrase_rect.right - 6
        bar_y = phrase_rows_area.y
        bar_h = phrase_rows_area.height
        pygame.draw.rect(screen, (60, 60, 80), pygame.Rect(bar_x, bar_y, 4, bar_h))
        thumb_h = max(22, int(bar_h * (phrase_rows_area.height / phrase_content_h)))
        thumb_range = bar_h - thumb_h
        thumb_y = bar_y + int((phrase_scroll_px / phrase_max_scroll) * thumb_range)
        pygame.draw.rect(screen, (150, 150, 180), pygame.Rect(bar_x, thumb_y, 4, thumb_h))

    controls = get_inspector_controls()
    perception_tab_rect = _to_rect(pygame, controls["perception_tab"])
    backtrace_tab_rect = _to_rect(pygame, controls["backtrace_tab"])
    command_tab_rect = _to_rect(pygame, controls["command_tab"])
    _draw_tab_button(pygame, screen, perception_tab_rect, "Perception", inspector_tab == "perception", small_font)
    _draw_tab_button(pygame, screen, backtrace_tab_rect, "Backtrace", inspector_tab == "backtrace", small_font)
    _draw_tab_button(pygame, screen, command_tab_rect, "Commands", inspector_tab == "command", small_font)

    body_top = perception_rect.y + 34

    if inspector_tab == "perception":
        menu_title = small_font.render("Perception Menu (Dragged Object)", True, (185, 205, 230))
        screen.blit(menu_title, (perception_rect.x + 8, body_top))

        if drag_perception["kind"] is None:
            hint = small_font.render("Drag an object from the bar to inspect perception.", True, (154, 166, 182))
            screen.blit(hint, (perception_rect.x + 8, body_top + 22))
            return slot_scroll_px, phrase_scroll_px

        concept_txt = concept_label(drag_perception["kind"], drag_perception["color_name"])
        hover_text = f"Hover: {_format_cell(drag_perception['hover_cell'])}"
        preview_text = f"Preview drop: {_format_cell(drag_perception['preview_cell'])}"
        syntax_text = "Syntax: Color -> Object"

        screen.blit(small_font.render(f"Object: {concept_txt}", True, (222, 222, 232)), (perception_rect.x + 8, body_top + 22))
        screen.blit(small_font.render(hover_text, True, (154, 166, 182)), (perception_rect.x + 8, body_top + 40))
        screen.blit(small_font.render(preview_text, True, (154, 166, 182)), (perception_rect.x + 8, body_top + 58))
        screen.blit(small_font.render(syntax_text, True, (154, 166, 182)), (perception_rect.x + 8, body_top + 76))

        headers = ["Agent", "Sees", "Utterance"]
        x_offsets = [8, 96, 162]
        base_y = body_top + 102
        for i, header in enumerate(headers):
            txt = small_font.render(header, True, (166, 186, 210))
            screen.blit(txt, (perception_rect.x + x_offsets[i], base_y))

        row_y = base_y + 22
        for row in drag_perception["rows"]:
            sees_label = "YES" if row["sees_object"] else "NO"
            sees_color = (164, 228, 172) if row["sees_object"] else (235, 156, 156)

            screen.blit(small_font.render(row["agent_name"], True, (222, 222, 232)), (perception_rect.x + x_offsets[0], row_y))
            screen.blit(small_font.render(sees_label, True, sees_color), (perception_rect.x + x_offsets[1], row_y))
            screen.blit(small_font.render(row["utterance"], True, (222, 222, 232)), (perception_rect.x + x_offsets[2], row_y))
            row_y += 22
        return slot_scroll_px, phrase_scroll_px

    if inspector_tab == "command":
        menu_title = small_font.render("Command Panel", True, (185, 205, 230))
        screen.blit(menu_title, (perception_rect.x + 8, body_top))

        auto_rect = _to_rect(pygame, controls["cmd_auto"])
        force_a_rect = _to_rect(pygame, controls["cmd_force_a"])
        force_b_rect = _to_rect(pygame, controls["cmd_force_b"])
        force_c_rect = _to_rect(pygame, controls["cmd_force_c"])
        cmd_color_prev_rect = _to_rect(pygame, controls["cmd_color_prev"])
        cmd_color_label_rect = _to_rect(pygame, controls["cmd_color_label"])
        cmd_color_next_rect = _to_rect(pygame, controls["cmd_color_next"])
        cmd_object_prev_rect = _to_rect(pygame, controls["cmd_object_prev"])
        cmd_object_label_rect = _to_rect(pygame, controls["cmd_object_label"])
        cmd_object_next_rect = _to_rect(pygame, controls["cmd_object_next"])

        auto_on = bool(command_state.get("auto_enabled", False))
        auto_label = "Auto: ON" if auto_on else "Auto: OFF"
        for rect, label in [
            (auto_rect, auto_label),
            (force_a_rect, f"Force A {format_word(command_state.get('come_token'))}"),
            (force_b_rect, f"Force B {format_word(command_state.get('leave_token'))}"),
            (force_c_rect, f"Force C {format_word(command_state.get('find_token'))}"),
        ]:
            pygame.draw.rect(screen, (42, 52, 68), rect, border_radius=5)
            pygame.draw.rect(screen, (102, 120, 144), rect, width=1, border_radius=5)
            txt = small_font.render(label, True, (222, 230, 242))
            screen.blit(txt, (rect.x + 6, rect.y + 4))

        for rect, label in [
            (cmd_color_prev_rect, "<"),
            (cmd_color_next_rect, ">"),
            (cmd_object_prev_rect, "<"),
            (cmd_object_next_rect, ">"),
        ]:
            pygame.draw.rect(screen, (42, 49, 62), rect, border_radius=5)
            pygame.draw.rect(screen, (96, 114, 138), rect, width=1, border_radius=5)
            txt = small_font.render(label, True, (222, 230, 242))
            screen.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2, rect.y + 3))

        selected_color = command_state.get("find_target_color", "red")
        selected_object = command_state.get("find_target_object", "apple")
        for rect, label in [
            (cmd_color_label_rect, f"Find Color: {selected_color.title()}"),
            (cmd_object_label_rect, f"Find Obj: {selected_object.title()}"),
        ]:
            pygame.draw.rect(screen, (30, 36, 48), rect, border_radius=5)
            pygame.draw.rect(screen, (88, 102, 124), rect, width=1, border_radius=5)
            txt = small_font.render(label, True, (214, 224, 236))
            screen.blit(txt, (rect.x + 7, rect.y + 4))

        status_y = auto_rect.y + 58
        status_lines = [
            f"Spoken phrase: {command_state.get('spoken_text', '----')}",
            f"Intent/Action: {command_state.get('current_intent') or 'idle'} / {command_state.get('current_action') or 'idle'}",
            f"Target: {concept_label(command_state.get('current_target_object') or selected_object, command_state.get('current_target_color') or selected_color)}",
            f"Decoded: {command_state.get('decoded_target_label', '-')}",
            f"Fetch phase: {command_state.get('current_phase', 'idle')}",
            f"Search mode: {command_state.get('search_mode') or '-'}",
            f"Carrying: {command_state.get('carried_object').color_name.title() + ' ' + command_state.get('carried_object').kind.title() if command_state.get('carried_object') else 'None'}",
            f"Last reward: {command_state.get('last_reward') if command_state.get('last_reward') is not None else 'n/a'}",
            f"Visibility: {command_state.get('last_visibility', 'n/a')}  Outcome: {command_state.get('last_outcome', 'n/a')}",
        ]
        if settings.COMMAND_PANEL_SHOW_VERBOSE:
            status_lines.extend(
                [
                    f"Decode quality: {command_state.get('decoded_target_quality', 'n/a')}",
                    f"Decode notes: {command_state.get('decoded_target_notes', '') or '-'}",
                ]
            )
        for idx, line in enumerate(status_lines):
            color = (206, 220, 238) if idx < 2 else (176, 194, 216)
            txt = small_font.render(line, True, color)
            screen.blit(txt, (perception_rect.x + 8, status_y + idx * 18))

        return slot_scroll_px, phrase_scroll_px

    menu_title = small_font.render("Backtrace Tester", True, (185, 205, 230))
    screen.blit(menu_title, (perception_rect.x + 8, body_top))

    if not is_backtrace_enabled(mode):
        disabled = small_font.render("Backtrace is available only in factored mode.", True, (194, 152, 152))
        screen.blit(disabled, (perception_rect.x + 8, body_top + 24))
        return slot_scroll_px, phrase_scroll_px

    mode_toggle_rect = _to_rect(pygame, controls["mode_toggle"])
    autofill_rect = _to_rect(pygame, controls["autofill"])
    run_rect = _to_rect(pygame, controls["run"])
    color_prev_rect = _to_rect(pygame, controls["color_prev"])
    color_next_rect = _to_rect(pygame, controls["color_next"])
    color_label_rect = _to_rect(pygame, controls["color_label"])
    object_prev_rect = _to_rect(pygame, controls["object_prev"])
    object_next_rect = _to_rect(pygame, controls["object_next"])
    object_label_rect = _to_rect(pygame, controls["object_label"])
    token_color_rect = _to_rect(pygame, controls["token_color"])
    token_object_rect = _to_rect(pygame, controls["token_object"])

    input_mode = backtrace_state.get("input_mode", "semantic")
    selected_color = backtrace_state.get("color_name", "red")
    selected_object = backtrace_state.get("object_name", "apple")
    token_color = backtrace_state.get("token_color", "")
    token_object = backtrace_state.get("token_object", "")
    focused_field = backtrace_state.get("focus_field")
    result_rows = backtrace_state.get("results", [])

    pygame.draw.rect(screen, (38, 49, 62), mode_toggle_rect, border_radius=5)
    pygame.draw.rect(screen, (96, 116, 142), mode_toggle_rect, width=1, border_radius=5)
    mode_text = "Input: Semantic" if input_mode == "semantic" else "Input: Token IDs"
    screen.blit(small_font.render(mode_text, True, (222, 230, 242)), (mode_toggle_rect.x + 8, mode_toggle_rect.y + 4))

    for rect, label in [(autofill_rect, "Use Selection"), (run_rect, "Run")]:
        pygame.draw.rect(screen, (44, 54, 70), rect, border_radius=5)
        pygame.draw.rect(screen, (102, 120, 144), rect, width=1, border_radius=5)
        txt = small_font.render(label, True, (222, 230, 242))
        screen.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2, rect.y + 4))

    for rect, label in [(color_prev_rect, "<"), (color_next_rect, ">"), (object_prev_rect, "<"), (object_next_rect, ">")]:
        pygame.draw.rect(screen, (42, 49, 62), rect, border_radius=5)
        pygame.draw.rect(screen, (96, 114, 138), rect, width=1, border_radius=5)
        txt = small_font.render(label, True, (222, 230, 242))
        screen.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2, rect.y + 3))

    for rect, label in [(color_label_rect, f"Color: {selected_color.title()}"), (object_label_rect, f"Object: {selected_object.title()}")]:
        pygame.draw.rect(screen, (30, 36, 48), rect, border_radius=5)
        pygame.draw.rect(screen, (88, 102, 124), rect, width=1, border_radius=5)
        txt = small_font.render(label, True, (214, 224, 236))
        screen.blit(txt, (rect.x + 7, rect.y + 4))

    token_label = small_font.render("Token Input:", True, (166, 186, 210))
    screen.blit(token_label, (token_color_rect.x, token_color_rect.y - 16))

    for rect, value, name in [
        (token_color_rect, token_color, "color"),
        (token_object_rect, token_object, "object"),
    ]:
        border = (150, 180, 220) if focused_field == name else (92, 106, 130)
        pygame.draw.rect(screen, (28, 32, 42), rect, border_radius=5)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=5)
        txt = small_font.render(value if value else "----", True, (224, 232, 244))
        screen.blit(txt, (rect.x + 7, rect.y + 4))

    result_y = token_color_rect.y + 30
    headers = ["Agent", "Tokens", "Concept", "Seen", "Quality"]
    x_offsets = [8, 72, 150, 282, 338]
    for idx, header in enumerate(headers):
        txt = small_font.render(header, True, (166, 186, 210))
        screen.blit(txt, (perception_rect.x + x_offsets[idx], result_y))

    row_y = result_y + 18
    if not result_rows:
        hint = small_font.render("Run backtrace to decode a concept.", True, (154, 166, 182))
        screen.blit(hint, (perception_rect.x + 8, row_y + 2))
    else:
        for row in result_rows:
            seen = row.get("seen_in_training")
            if seen is None:
                seen_label = "N/A"
                seen_color = (188, 194, 206)
            elif seen:
                seen_label = "Seen"
                seen_color = (164, 228, 172)
            else:
                seen_label = "Novel"
                seen_color = (235, 196, 140)

            quality = row.get("decode_quality", "?")
            quality_color = (164, 228, 172) if quality == "exact" else (224, 208, 152)
            if quality == "invalid":
                quality_color = (235, 156, 156)
            if quality == "ambiguous":
                quality_color = (232, 186, 142)

            token_text = f"[{row.get('color_token_input', '----')}, {row.get('object_token_input', '----')}]"

            screen.blit(small_font.render(row.get("agent_name", "?"), True, (222, 222, 232)), (perception_rect.x + x_offsets[0], row_y))
            screen.blit(small_font.render(token_text, True, (222, 222, 232)), (perception_rect.x + x_offsets[1], row_y))
            screen.blit(small_font.render(row.get("decoded_concept_label", "UNKNOWN"), True, (222, 222, 232)), (perception_rect.x + x_offsets[2], row_y))
            screen.blit(small_font.render(seen_label, True, seen_color), (perception_rect.x + x_offsets[3], row_y))
            screen.blit(small_font.render(quality, True, quality_color), (perception_rect.x + x_offsets[4], row_y))
            row_y += 19

            notes = row.get("notes", "")
            if notes:
                clipped = notes if len(notes) <= 72 else notes[:69] + "..."
                screen.blit(
                    small_font.render(clipped, True, (154, 166, 182)),
                    (perception_rect.x + 8, row_y),
                )
                row_y += 17

    return slot_scroll_px, phrase_scroll_px


def run_training(mode, object_types, color_names, seed, steps, write_logs, run_sweep, holdout_pairs=None):
    rng = random.Random(seed) if seed is not None else random.Random()
    cfg = TrainingConfig(learning_steps=steps)

    adam_lang = LanguageAgent("Adam")
    eve_lang = LanguageAgent("Eve")

    results_by_object = {}
    slot_rows = []
    phrase_rows = []
    metrics = {}
    seen_pairs = []
    heldout_pairs = []

    if mode == "holistic":
        concept_specs = [
            (object_type, color_name, concept_key(object_type, color_name))
            for object_type in object_types
            for color_name in color_names
        ]
        concept_types = [concept for _, _, concept in concept_specs]

        initialize_language(adam_lang, concept_types, cfg, rng)
        initialize_language(eve_lang, concept_types, cfg, rng)

        for object_type, color_name, concept in concept_specs:
            result = train_language(adam_lang, eve_lang, concept, cfg, rng)
            results_by_object[concept] = result

            adam_word = format_word(best_word(adam_lang, concept))
            eve_word = format_word(best_word(eve_lang, concept))
            shared = adam_word if adam_word == eve_word else "NO_CONSENSUS"

            phrase_rows.append(
                {
                    "object_type": object_type,
                    "color_name": color_name,
                    "concept": concept,
                    "concept_label": concept_label(object_type, color_name),
                    "syntax_order": "color_object",
                    "adam_phrase_ids": adam_word,
                    "eve_phrase_ids": eve_word,
                    "shared_phrase_ids": shared,
                    "converged_step": result.converged_step,
                    # Backwards-compatible aliases.
                    "adam_word": adam_word,
                    "eve_word": eve_word,
                    "shared_word": shared,
                }
            )

        shared_words = [row["shared_phrase_ids"] for row in phrase_rows if row["shared_phrase_ids"] != "NO_CONSENSUS"]
        metrics = {
            "target_unique": len(phrase_rows),
            "achieved_unique": len(set(shared_words)),
            "all_phrase_consensus": all(row["shared_phrase_ids"] != "NO_CONSENSUS" for row in phrase_rows),
            "all_slot_consensus": False,
        }
        seen_pairs = [f"{color_name}:{object_type}" for object_type in object_types for color_name in color_names]
        heldout_pairs = []

    else:
        color_semantics = [color_key(color_name) for color_name in color_names]
        object_semantics = [object_key(object_type) for object_type in object_types]
        initialize_language(adam_lang, color_semantics + object_semantics, cfg, rng)
        initialize_language(eve_lang, color_semantics + object_semantics, cfg, rng)

        factored = train_factored_language(
            adam_lang,
            eve_lang,
            color_names,
            object_types,
            cfg,
            rng,
            holdout_pairs=holdout_pairs,
        )
        slot_rows = factored.slot_rows
        phrase_rows = factored.phrase_rows
        metrics = factored.metrics
        seen_pairs = factored.seen_pairs
        heldout_pairs = factored.heldout_pairs

    command_cfg = CommandTrainingConfig(word_space_size=cfg.word_space_size)
    command_rng = random.Random(seed + 1337) if seed is not None else random.Random()
    come_token, leave_token, find_token = initialize_hidden_command_mapping(command_cfg, command_rng)
    command_result = train_command_grounding(
        command_cfg,
        come_token,
        leave_token,
        find_token,
        seen_pairs=seen_pairs,
        rng=command_rng,
    )

    lexicon_rows = phrase_rows

    log_paths = {}
    if write_logs:
        log_paths = write_lexicon_logs(
            results_by_object=results_by_object,
            lexicon_rows=lexicon_rows,
            seed=seed,
            cfg=cfg,
            mode=mode,
            syntax_order="color_object",
            slot_lexicon=slot_rows,
            phrase_lexicon=phrase_rows,
            metrics=metrics,
            seen_pairs=seen_pairs,
            holdout_pairs=heldout_pairs,
            command_tokens={
                "come_token": command_result.come_token,
                "leave_token": command_result.leave_token,
                "find_token": command_result.find_token,
            },
            command_training={
                "episodes": command_result.episodes,
                "accuracy": command_result.accuracy,
                "accuracy_by_intent": command_result.accuracy_by_intent,
                "eve_action_q": command_result.eve_action_q,
            },
        )

    sweep_path = None
    if run_sweep:
        if mode == "factored" and object_types and color_names:
            target_semantic = concept_key(object_types[0], color_names[0])
            sweep_path = sweep_social_pressures(object_type=target_semantic, cfg=cfg)
        elif mode == "holistic" and object_types and color_names:
            target_semantic = concept_key(object_types[0], color_names[0])
            sweep_path = sweep_social_pressures(object_type=target_semantic, cfg=cfg)

    return (
        cfg,
        adam_lang,
        eve_lang,
        results_by_object,
        slot_rows,
        phrase_rows,
        metrics,
        log_paths,
        sweep_path,
        seen_pairs,
        heldout_pairs,
        command_result,
    )


def run_visualization(
    mode,
    slot_rows,
    phrase_rows,
    metrics,
    adam_lang,
    eve_lang,
    object_types,
    color_names,
    seen_pairs,
    command_result,
):
    try:
        import pygame
    except ModuleNotFoundError:
        print("pygame is not installed; use --train-only to run headless training.")
        return

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("HAL - Lexicon Emergence")
    font = pygame.font.SysFont("Menlo", 16)
    small_font = pygame.font.SysFont("Menlo", 14)

    walkable_cells, wall_cells = generate_cave_layout()
    tiles = generate_map(pygame, wall_cells)
    sprite_cache = {}
    palette_slots = build_palette_slots(pygame, object_types)

    adam = WorldAgent("Adam", (180, 70, 70), adam_lang)
    eve = WorldAgent("Eve", (70, 130, 210), eve_lang)
    agents = [adam, eve]
    choose_agent_cells(agents, walkable_cells)
    camera_origin = center_camera_on_cell(adam.cell)
    _initialize_subjective_world(adam, color_names, object_types)
    _initialize_subjective_world(eve, color_names, object_types)

    objects_by_cell = {}
    selected_color = color_names[0] if color_names else "red"
    color_dropdown_open = False
    dragging_kind = None
    dragging_color = None
    drag_pos = (0, 0)
    dragging_source = None
    dragging_world_object = None
    dragging_origin_cell = None

    slot_scroll_px = 0
    phrase_scroll_px = 0

    inspector_tab = "perception"
    seen_pairs_set = set(seen_pairs or [])
    backtrace_color_options = _slot_options(slot_rows, "color", color_names)
    backtrace_object_options = _slot_options(slot_rows, "object", object_types)
    backtrace_state = {
        "input_mode": "semantic",
        "color_name": backtrace_color_options[0] if backtrace_color_options else "red",
        "object_name": backtrace_object_options[0] if backtrace_object_options else "apple",
        "token_color": "",
        "token_object": "",
        "focus_field": None,
        "results": [],
    }
    command_state = {
        "come_token": command_result.come_token,
        "leave_token": command_result.leave_token,
        "find_token": command_result.find_token,
        "token_to_intent": {
            command_result.come_token: "come",
            command_result.leave_token: "leave",
            command_result.find_token: "find",
        },
        "eve_action_q": command_result.eve_action_q,
        "auto_enabled": True,
        "pending_override": None,
        "pending_find_target": None,
        "all_colors": list(color_names),
        "all_objects": list(object_types),
        "find_target_color": color_names[0] if color_names else "red",
        "find_target_object": object_types[0] if object_types else "apple",
        "seen_pairs": seen_pairs_set,
        "mode": mode,
        "frame": 0,
        "last_issue_frame": -settings.COMMAND_INTERVAL_FRAMES,
        "spoken_token": None,
        "spoken_text": "----",
        "speech_timer": 0,
        "feedback_text": "",
        "feedback_timer": 0,
        "active_path": [],
        "path_steps": 0,
        "fetch_steps": 0,
        "move_cooldown": 0,
        "current_intent": None,
        "current_action": None,
        "current_phase": "idle",
        "current_target_color": None,
        "current_target_object": None,
        "current_target_cell": None,
        "return_target_cell": None,
        "search_mode": None,
        "decoded_target_label": "-",
        "decoded_target_quality": "n/a",
        "decoded_target_notes": "",
        "carried_object": None,
        "last_reward": None,
        "last_visibility": "n/a",
        "last_outcome": "n/a",
    }
    command_rng = random.Random(
        (command_result.come_token * 131)
        + (command_result.leave_token * 17)
        + (command_result.find_token * 7)
    )

    def run_backtrace_query(use_semantic_selection):
        if not is_backtrace_enabled(mode):
            return

        if use_semantic_selection or backtrace_state["input_mode"] == "semantic":
            color_token, object_token = _semantic_tokens_from_selection(
                adam.language,
                backtrace_state["color_name"],
                backtrace_state["object_name"],
                backtrace_color_options,
                backtrace_object_options,
            )
            backtrace_state["token_color"] = color_token
            backtrace_state["token_object"] = object_token
        else:
            color_token = backtrace_state["token_color"]
            object_token = backtrace_state["token_object"]

        backtrace_state["results"] = _build_backtrace_rows(
            agents,
            color_token=color_token,
            object_token=object_token,
            color_options=backtrace_color_options,
            object_options=backtrace_object_options,
            seen_pairs=seen_pairs_set,
        )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue

                if (
                    inspector_tab == "backtrace"
                    and is_backtrace_enabled(mode)
                    and backtrace_state["input_mode"] == "token"
                    and backtrace_state.get("focus_field") in {"color", "object"}
                ):
                    field = "token_color" if backtrace_state["focus_field"] == "color" else "token_object"
                    value = backtrace_state.get(field, "")

                    if event.key == pygame.K_BACKSPACE:
                        backtrace_state[field] = value[:-1]
                    elif event.key == pygame.K_TAB:
                        backtrace_state["focus_field"] = "object" if field == "token_color" else "color"
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        run_backtrace_query(use_semantic_selection=False)
                    elif event.unicode and event.unicode.isdigit() and len(value) < 6:
                        backtrace_state[field] = value + event.unicode
            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                layout = get_lexicon_panel_layout()

                slot_rows_area = (
                    layout["slot_rect"][0] + 4,
                    layout["slot_rect"][1] + 52,
                    layout["slot_rect"][2] - 10,
                    layout["slot_rect"][3] - 58,
                )
                phrase_rows_area = (
                    layout["phrase_rect"][0] + 4,
                    layout["phrase_rect"][1] + 52,
                    layout["phrase_rect"][2] - 10,
                    layout["phrase_rect"][3] - 58,
                )

                if point_in_rect(mouse_pos, slot_rows_area):
                    slot_scroll_px -= event.y * 24
                elif point_in_rect(mouse_pos, phrase_rows_area):
                    phrase_scroll_px -= event.y * 24
                else:
                    phrase_scroll_px -= event.y * 24
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                controls = get_inspector_controls()
                panel_layout = get_lexicon_panel_layout()

                if point_in_rect(event.pos, controls["perception_tab"]):
                    inspector_tab = "perception"
                    backtrace_state["focus_field"] = None
                    continue
                if point_in_rect(event.pos, controls["backtrace_tab"]):
                    inspector_tab = "backtrace"
                    continue
                if point_in_rect(event.pos, controls["command_tab"]):
                    inspector_tab = "command"
                    continue

                if inspector_tab == "backtrace":
                    if point_in_rect(event.pos, controls["mode_toggle"]):
                        backtrace_state["input_mode"] = "token" if backtrace_state["input_mode"] == "semantic" else "semantic"
                        backtrace_state["focus_field"] = None
                        continue

                    if is_backtrace_enabled(mode):
                        if point_in_rect(event.pos, controls["color_prev"]):
                            backtrace_state["color_name"] = _cycle_selected(
                                backtrace_color_options,
                                backtrace_state["color_name"],
                                -1,
                            )
                            continue
                        if point_in_rect(event.pos, controls["color_next"]):
                            backtrace_state["color_name"] = _cycle_selected(
                                backtrace_color_options,
                                backtrace_state["color_name"],
                                1,
                            )
                            continue
                        if point_in_rect(event.pos, controls["object_prev"]):
                            backtrace_state["object_name"] = _cycle_selected(
                                backtrace_object_options,
                                backtrace_state["object_name"],
                                -1,
                            )
                            continue
                        if point_in_rect(event.pos, controls["object_next"]):
                            backtrace_state["object_name"] = _cycle_selected(
                                backtrace_object_options,
                                backtrace_state["object_name"],
                                1,
                            )
                            continue
                        if point_in_rect(event.pos, controls["autofill"]):
                            run_backtrace_query(use_semantic_selection=True)
                            backtrace_state["focus_field"] = None
                            continue
                        if point_in_rect(event.pos, controls["run"]):
                            run_backtrace_query(use_semantic_selection=False)
                            backtrace_state["focus_field"] = None
                            continue
                        if point_in_rect(event.pos, controls["token_color"]):
                            backtrace_state["focus_field"] = "color"
                            continue
                        if point_in_rect(event.pos, controls["token_object"]):
                            backtrace_state["focus_field"] = "object"
                            continue

                    if point_in_rect(event.pos, panel_layout["perception_rect"]):
                        backtrace_state["focus_field"] = None
                elif inspector_tab == "command":
                    if point_in_rect(event.pos, controls["cmd_auto"]):
                        command_state["auto_enabled"] = not command_state["auto_enabled"]
                        continue
                    if point_in_rect(event.pos, controls["cmd_force_a"]):
                        command_state["pending_override"] = command_state["come_token"]
                        continue
                    if point_in_rect(event.pos, controls["cmd_force_b"]):
                        command_state["pending_override"] = command_state["leave_token"]
                        continue
                    if point_in_rect(event.pos, controls["cmd_force_c"]):
                        command_state["pending_override"] = command_state["find_token"]
                        command_state["pending_find_target"] = (
                            command_state["find_target_color"],
                            command_state["find_target_object"],
                        )
                        continue
                    if point_in_rect(event.pos, controls["cmd_color_prev"]):
                        command_state["find_target_color"] = _cycle_selected(
                            command_state["all_colors"],
                            command_state["find_target_color"],
                            -1,
                        )
                        continue
                    if point_in_rect(event.pos, controls["cmd_color_next"]):
                        command_state["find_target_color"] = _cycle_selected(
                            command_state["all_colors"],
                            command_state["find_target_color"],
                            1,
                        )
                        continue
                    if point_in_rect(event.pos, controls["cmd_object_prev"]):
                        command_state["find_target_object"] = _cycle_selected(
                            command_state["all_objects"],
                            command_state["find_target_object"],
                            -1,
                        )
                        continue
                    if point_in_rect(event.pos, controls["cmd_object_next"]):
                        command_state["find_target_object"] = _cycle_selected(
                            command_state["all_objects"],
                            command_state["find_target_object"],
                            1,
                        )
                        continue

                header_rect, option_rects = get_color_dropdown_geometry(pygame, palette_slots, color_names)

                if header_rect.collidepoint(event.pos):
                    color_dropdown_open = not color_dropdown_open
                    continue

                if color_dropdown_open:
                    selected = None
                    for color_name in color_names:
                        if option_rects[color_name].collidepoint(event.pos):
                            selected = color_name
                            break
                    color_dropdown_open = False
                    if selected is not None:
                        selected_color = selected
                        continue

                for slot in palette_slots:
                    if slot["rect"].collidepoint(event.pos):
                        dragging_kind = slot["kind"]
                        dragging_color = selected_color
                        drag_pos = event.pos
                        dragging_source = "palette"
                        dragging_world_object = None
                        dragging_origin_cell = None
                        break
                else:
                    clicked_cell = screen_to_cell(*event.pos, camera_origin)
                    if clicked_cell is not None:
                        existing_object = objects_by_cell.get(clicked_cell)
                        if existing_object is not None:
                            dragging_kind = existing_object.kind
                            dragging_color = existing_object.color_name
                            drag_pos = event.pos
                            dragging_source = "world"
                            dragging_world_object = objects_by_cell.pop(clicked_cell)
                            dragging_origin_cell = clicked_cell
            elif event.type == pygame.MOUSEMOTION and dragging_kind is not None:
                drag_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging_kind is not None:
                drag_pos = event.pos
                delete_rect = get_delete_zone_rect(pygame)

                if dragging_source == "world":
                    dropped = False
                    if delete_rect.collidepoint(event.pos):
                        dropped = True
                    else:
                        drop_cell = screen_to_cell(*event.pos, camera_origin)
                        if drop_cell is not None:
                            occupied_by_agent = any(agent.cell == drop_cell for agent in agents)
                            occupied_by_object = drop_cell in objects_by_cell
                            walkable_drop = drop_cell in walkable_cells
                            if (
                                walkable_drop
                                and not occupied_by_agent
                                and not occupied_by_object
                                and dragging_world_object is not None
                            ):
                                dragging_world_object.cell = drop_cell
                                dragging_world_object.concept = concept_key(
                                    dragging_world_object.kind,
                                    dragging_world_object.color_name,
                                )
                                dragging_world_object.sprite = get_sprite(
                                    pygame,
                                    sprite_cache,
                                    dragging_world_object.kind,
                                    dragging_world_object.color_name,
                                )
                                objects_by_cell[drop_cell] = dragging_world_object
                                dropped = True
                    if not dropped and dragging_world_object is not None and dragging_origin_cell is not None:
                        if dragging_origin_cell not in objects_by_cell:
                            dragging_world_object.cell = dragging_origin_cell
                            objects_by_cell[dragging_origin_cell] = dragging_world_object
                else:
                    drop_cell = screen_to_cell(*event.pos, camera_origin)
                    if drop_cell is not None:
                        occupied_by_agent = any(agent.cell == drop_cell for agent in agents)
                        occupied_by_object = drop_cell in objects_by_cell
                        if drop_cell in walkable_cells and not occupied_by_agent and not occupied_by_object:
                            concept = concept_key(dragging_kind, dragging_color)
                            sprite = get_sprite(pygame, sprite_cache, dragging_kind, dragging_color)
                            objects_by_cell[drop_cell] = WorldObject(
                                kind=dragging_kind,
                                color_name=dragging_color,
                                cell=drop_cell,
                                concept=concept,
                                sprite=sprite,
                            )
                dragging_kind = None
                dragging_color = None
                dragging_source = None
                dragging_world_object = None
                dragging_origin_cell = None

        keys = pygame.key.get_pressed()
        pan_dx = 0
        pan_dy = 0
        if keys[pygame.K_a]:
            pan_dx -= settings.CAMERA_SPEED_CELLS
        if keys[pygame.K_d]:
            pan_dx += settings.CAMERA_SPEED_CELLS
        if keys[pygame.K_w]:
            pan_dy -= settings.CAMERA_SPEED_CELLS
        if keys[pygame.K_s]:
            pan_dy += settings.CAMERA_SPEED_CELLS
        if pan_dx or pan_dy:
            camera_origin = pan_camera(camera_origin, pan_dx, pan_dy)

        screen.fill((8, 8, 8))
        delete_hover = (
            dragging_kind is not None
            and dragging_source == "world"
            and get_delete_zone_rect(pygame).collidepoint(drag_pos)
        )
        draw_inventory_bar(
            pygame,
            screen,
            font,
            small_font,
            palette_slots,
            sprite_cache,
            color_names,
            selected_color,
            color_dropdown_open,
            dragging_active=(dragging_kind is not None and dragging_source == "world"),
            delete_hover=delete_hover,
        )

        pre_visible_by_agent = {}
        for agent in agents:
            pre_visible_by_agent[agent.name] = raycast_visible_cells(agent, opaque=wall_cells)
        for agent in agents:
            _update_subjective_world(agent, pre_visible_by_agent[agent.name], objects_by_cell, command_state["frame"])

        adam_visible_for_command = pre_visible_by_agent.get(adam.name, set())
        update_command_state(
            command_state=command_state,
            adam=adam,
            eve=eve,
            adam_visible=adam_visible_for_command,
            objects_by_cell=objects_by_cell,
            wall_cells=wall_cells,
            rng=command_rng,
        )

        occupancy_names = {agent.cell: agent.name for agent in agents if agent.cell is not None}
        occupancy_colors = {agent.cell: agent.color for agent in agents if agent.cell is not None}

        all_visible = set()
        visible_by_agent = {}
        for agent in agents:
            visible_cells = raycast_visible_cells(agent, opaque=wall_cells)
            visible_by_agent[agent.name] = visible_cells
            all_visible |= visible_cells
            build_vision_slice(agent, visible_cells, occupancy_names, objects_by_cell)
            classify_vision_cells(agent.vision_slice)

        x0, y0, x1, y1 = viewport_bounds(camera_origin)
        for y in range(y0, y1):
            for x in range(x0, x1):
                cell = (x, y)
                tile = tiles[cell]
                if cell in wall_cells:
                    highlight = (78, 66, 50)
                    if cell in all_visible:
                        highlight = (118, 102, 78)
                else:
                    highlight = (44, 48, 56)
                    if cell in all_visible:
                        highlight = (90, 102, 128)
                if cell in occupancy_colors:
                    highlight = occupancy_colors[cell]
                tile.draw(pygame=pygame, screen=screen, camera_origin=camera_origin, fill_color=highlight)

        for obj in objects_by_cell.values():
            if is_in_viewport(obj.cell, camera_origin):
                obj.draw(pygame=pygame, screen=screen, camera_origin=camera_origin)

        for agent in agents:
            if is_in_viewport(agent.cell, camera_origin):
                draw_agent(pygame, screen, agent, camera_origin)
        draw_carried_object(
            pygame,
            screen,
            carrier=eve,
            carried_object=command_state.get("carried_object"),
            sprite_cache=sprite_cache,
            camera_origin=camera_origin,
        )

        adam_x, adam_y = cell_to_screen(*adam.cell, camera_origin)
        bubble_anchor_x = adam_x + VISUAL_TILE_WIDTH // 2
        bubble_anchor_y = adam_y
        if command_state.get("speech_timer", 0) > 0:
            draw_speech_bubble(
                pygame,
                screen,
                small_font,
                command_state.get("spoken_text", "----"),
                bubble_anchor_x,
                bubble_anchor_y,
                fill=(228, 238, 250),
                fg=(24, 34, 46),
            )
        if command_state.get("feedback_timer", 0) > 0:
            draw_speech_bubble(
                pygame,
                screen,
                small_font,
                command_state.get("feedback_text", ""),
                bubble_anchor_x,
                bubble_anchor_y - 28,
                fill=(246, 236, 214),
                fg=(36, 32, 18),
            )

        draw_color_dropdown(
            pygame,
            screen,
            small_font,
            palette_slots,
            color_names,
            selected_color,
            color_dropdown_open,
        )

        if dragging_kind is not None and dragging_color is not None:
            ghost = pygame.transform.scale(
                get_sprite(pygame, sprite_cache, dragging_kind, dragging_color),
                (VISUAL_TILE_WIDTH, VISUAL_TILE_HEIGHT),
            )
            ghost.set_alpha(215)
            gx = drag_pos[0] - ghost.get_width() // 2
            gy = drag_pos[1] - ghost.get_height() // 2
            screen.blit(ghost, (gx, gy))

        drag_perception = compute_drag_perception(
            dragging_kind,
            dragging_color,
            drag_pos,
            agents,
            visible_by_agent,
            mode,
            camera_origin,
            walkable_cells,
        )

        slot_scroll_px, phrase_scroll_px = draw_lexicon_panel(
            pygame,
            screen,
            font,
            small_font,
            mode,
            slot_rows,
            phrase_rows,
            metrics,
            drag_perception,
            inspector_tab,
            backtrace_state,
            command_state,
            slot_scroll_px,
            phrase_scroll_px,
        )

        pygame.display.flip()

    pygame.quit()


def parse_object_types(raw):
    if raw is None:
        return DEFAULT_OBJECT_TYPES
    object_types = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not object_types:
        return DEFAULT_OBJECT_TYPES

    return list(dict.fromkeys(object_types))


def parse_color_names(raw):
    if raw is None:
        return DEFAULT_COLOR_NAMES
    color_names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    color_names = [item for item in color_names if item in COLOR_RGB]
    if not color_names:
        return DEFAULT_COLOR_NAMES

    return list(dict.fromkeys(color_names))


def parse_holdout_phrases(raw, color_names, object_types):
    if raw is None or not raw.strip():
        return [], []

    color_set = set(color_names)
    object_set = set(object_types)
    pairs = []
    warnings = []

    for token in raw.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if ":" not in item:
            warnings.append(f"Ignoring malformed holdout '{item}'. Expected color:object.")
            continue

        color_name, object_type = [part.strip() for part in item.split(":", 1)]
        if not color_name or not object_type:
            warnings.append(f"Ignoring malformed holdout '{item}'. Expected color:object.")
            continue
        if color_name not in color_set or object_type not in object_set:
            warnings.append(
                f"Ignoring holdout '{item}' because it is outside selected --colors/--objects."
            )
            continue

        pairs.append((color_name, object_type))

    return list(dict.fromkeys(pairs)), warnings


def parse_args():
    parser = argparse.ArgumentParser(description="HAL language emergence demo")
    parser.add_argument(
        "--mode",
        choices=["factored", "holistic"],
        default=DEFAULT_MODE,
        help="Training mode: factored (slot words) or holistic (one word per concept)",
    )
    parser.add_argument(
        "--objects",
        default=",".join(DEFAULT_OBJECT_TYPES),
        help="Comma-separated object list for lexicon training",
    )
    parser.add_argument(
        "--colors",
        default=",".join(DEFAULT_COLOR_NAMES),
        help="Comma-separated colors for concept generation",
    )
    parser.add_argument(
        "--holdout-phrases",
        default="",
        help="Optional held-out concept pairs as color:object,color:object",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible training")
    parser.add_argument(
        "--steps",
        type=int,
        default=settings.LEARNING_STEPS,
        help="Training steps to run",
    )
    parser.add_argument("--train-only", action="store_true", help="Run headless training + logs, no pygame")
    parser.add_argument("--no-log-files", action="store_true", help="Disable writing artifacts/*.json and *.txt")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run social-pressure sweep and write artifacts/sweep_soft_pressures_debug.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    object_types = parse_object_types(args.objects)
    color_names = parse_color_names(args.colors)
    holdout_pairs, holdout_warnings = parse_holdout_phrases(args.holdout_phrases, color_names, object_types)
    for warning in holdout_warnings:
        print(f"[holdout] {warning}")

    (
        _,
        adam_lang,
        eve_lang,
        _,
        slot_rows,
        phrase_rows,
        metrics,
        log_paths,
        sweep_path,
        seen_pairs,
        heldout_pairs,
        command_result,
    ) = run_training(
        mode=mode,
        object_types=object_types,
        color_names=color_names,
        seed=args.seed,
        steps=args.steps,
        write_logs=not args.no_log_files,
        run_sweep=args.sweep,
        holdout_pairs=holdout_pairs,
    )

    if args.train_only:
        print("HAL Lexicon")
        print("-" * 44)
        print(f"Mode: {mode}")
        print(
            f"Metrics: target_unique={metrics.get('target_unique', 'n/a')} "
            f"achieved_unique={metrics.get('achieved_unique', 'n/a')} "
            f"phrase_consensus={metrics.get('all_phrase_consensus', 'n/a')}"
        )
        if mode == "factored":
            print(f"Holdouts: trained={len(seen_pairs)} heldout={len(heldout_pairs)}")
        print(
            "Command grounding: "
            f"A={format_word(command_result.come_token)} "
            f"B={format_word(command_result.leave_token)} "
            f"C={format_word(command_result.find_token)} "
            f"accuracy={command_result.accuracy:.3f} "
            f"episodes={command_result.episodes}"
        )
        by_intent = command_result.accuracy_by_intent
        print(
            "Command accuracy by intent: "
            f"come={by_intent.get('come', 0.0):.3f} "
            f"leave={by_intent.get('leave', 0.0):.3f} "
            f"find={by_intent.get('find', 0.0):.3f}"
        )

        if slot_rows:
            print("\nBase Factored Lexicon")
            print("-" * 44)
            for row in slot_rows:
                print(
                    f"{row['slot_type']:<7} {row['slot_label']:<12} "
                    f"Adam:{row['adam_word']} Eve:{row['eve_word']} Shared:{row['shared_word']}"
                )

        print("\nComposed Phrases")
        print("-" * 44)
        for row in phrase_rows:
            trained_flag = "T" if row.get("was_trained", True) else "H"
            print(
                f"[{trained_flag}] {row['concept_label']:<16} Adam:{row.get('adam_phrase_ids', row.get('adam_word', '----')):<14} "
                f"Eve:{row.get('eve_phrase_ids', row.get('eve_word', '----')):<14} "
                f"Shared:{row.get('shared_phrase_ids', row.get('shared_word', 'NO_CONSENSUS'))}"
            )

        if log_paths:
            print("\nWrote training logs:")
            for key, value in sorted(log_paths.items()):
                print(f"- {key}: {value}")

        if sweep_path:
            print(f"\nWrote sweep report: {sweep_path}")
        return

    palette_object_types = [obj for obj in object_types if obj in DEFAULT_OBJECT_TYPES]
    if not palette_object_types:
        palette_object_types = DEFAULT_OBJECT_TYPES

    run_visualization(
        mode=mode,
        slot_rows=slot_rows,
        phrase_rows=phrase_rows,
        metrics=metrics,
        adam_lang=adam_lang,
        eve_lang=eve_lang,
        object_types=palette_object_types,
        color_names=color_names,
        seen_pairs=seen_pairs,
        command_result=command_result,
    )
    sys.exit()


if __name__ == "__main__":
    main()
