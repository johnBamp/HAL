from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

TileId = int
Word = tuple[int, ...]


@dataclass(frozen=True)
class GridConfig:
    width: int = 128
    height: int = 128


@dataclass(frozen=True)
class FOVConfig:
    angle_deg: float = 90.0
    range_tiles: float = 10.0


@dataclass(frozen=True)
class StageConfig:
    anchor_xy: tuple[int, int]
    anchor_heading_deg: float
    fov: FOVConfig


@dataclass
class AgentState:
    x: int
    y: int
    heading_deg: float


@dataclass
class InteractionEvent:
    step: int
    word: Word
    x_word: Word
    y_word: Word
    success: bool
    confidence: float
    confidence_x: float
    confidence_y: float


@dataclass
class WordStats:
    successes: int = 0
    failures: int = 0
    uses: int = 0

    def smoothed_success_rate(self) -> float:
        return (self.successes + 1.0) / (self.uses + 2.0)


@dataclass
class NamingTables:
    speaker_x_inventory: dict[int, dict[Word, WordStats]] = field(default_factory=dict)
    speaker_y_inventory: dict[int, dict[Word, WordStats]] = field(default_factory=dict)
    listener_x_assoc: dict[Word, dict[int, int]] = field(default_factory=dict)
    listener_y_assoc: dict[Word, dict[int, int]] = field(default_factory=dict)


def _angle_diff_deg(lhs: float, rhs: float) -> float:
    return (lhs - rhs + 180.0) % 360.0 - 180.0


class HalEnv:
    """Discrete environment for Stage-0 HAL training."""

    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        stages: Optional[list[StageConfig]] = None,
        seed: int = 0,
    ) -> None:
        self.grid = grid_config or GridConfig()
        self.rng = random.Random(seed)

        if stages:
            self.stages = stages
        else:
            center = (self.grid.width // 2, self.grid.height // 2)
            self.stages = [
                StageConfig(anchor_xy=center, anchor_heading_deg=0.0, fov=FOVConfig())
            ]

        self.current_stage_index = 0
        self._visible_cache: dict[int, list[TileId]] = {}
        self._visible_set_cache: dict[int, set[TileId]] = {}

        stage0 = self.current_stage
        self.agent1 = AgentState(
            x=stage0.anchor_xy[0],
            y=stage0.anchor_xy[1],
            heading_deg=stage0.anchor_heading_deg,
        )

        visible = self.visible_tiles()
        if not visible:
            raise ValueError("Current stage contains no visible tiles.")

        start_tile = self.rng.choice(visible)
        start_x, start_y = self.tile_xy(start_tile)
        self.agent2 = AgentState(
            x=start_x,
            y=start_y,
            heading_deg=self.rng.choice([0.0, 90.0, 180.0, 270.0]),
        )

    @property
    def current_stage(self) -> StageConfig:
        return self.stages[self.current_stage_index]

    @property
    def agent2_tile_id(self) -> TileId:
        return self.tile_id(self.agent2.x, self.agent2.y)

    def set_stage(self, index: int) -> None:
        if index < 0 or index >= len(self.stages):
            raise IndexError(f"Invalid stage index {index}.")
        self.current_stage_index = index
        stage = self.current_stage
        self.agent1 = AgentState(
            x=stage.anchor_xy[0],
            y=stage.anchor_xy[1],
            heading_deg=stage.anchor_heading_deg,
        )
        if self.agent2_tile_id not in self.visible_tile_set():
            self._reset_agent2_position()

    def tile_id(self, x: int, y: int) -> TileId:
        if not self.is_in_bounds(x, y):
            raise ValueError(f"Out-of-bounds tile coordinates ({x}, {y}).")
        return y * self.grid.width + x

    def tile_xy(self, tile_id: TileId) -> tuple[int, int]:
        if tile_id < 0 or tile_id >= self.grid.width * self.grid.height:
            raise ValueError(f"Out-of-bounds tile id {tile_id}.")
        y, x = divmod(tile_id, self.grid.width)
        return x, y

    def relative_coords(
        self,
        tile_id: TileId,
        stage: Optional[StageConfig] = None,
    ) -> tuple[int, int]:
        stage = stage or self.current_stage
        x, y = self.tile_xy(tile_id)
        ax, ay = stage.anchor_xy
        return x - ax, y - ay

    def tile_from_relative(
        self,
        x_rel: int,
        y_rel: int,
        stage: Optional[StageConfig] = None,
    ) -> Optional[TileId]:
        stage = stage or self.current_stage
        ax, ay = stage.anchor_xy
        x = ax + x_rel
        y = ay + y_rel
        if not self.is_in_bounds(x, y):
            return None
        return self.tile_id(x, y)

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid.width and 0 <= y < self.grid.height

    def visible_tiles(self, stage: Optional[StageConfig] = None) -> list[TileId]:
        if stage is not None:
            return self._compute_visible_tiles(stage)

        idx = self.current_stage_index
        if idx not in self._visible_cache:
            tiles = self._compute_visible_tiles(self.current_stage)
            self._visible_cache[idx] = tiles
            self._visible_set_cache[idx] = set(tiles)
        return self._visible_cache[idx]

    def visible_tile_set(self) -> set[TileId]:
        _ = self.visible_tiles()
        return self._visible_set_cache[self.current_stage_index]

    def _compute_visible_tiles(self, stage: StageConfig) -> list[TileId]:
        tiles: list[TileId] = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                tile = self.tile_id(x, y)
                if self.is_in_fov(tile, stage):
                    tiles.append(tile)
        return tiles

    def is_in_fov(self, tile_id: TileId, stage: Optional[StageConfig] = None) -> bool:
        stage = stage or self.current_stage
        x, y = self.tile_xy(tile_id)
        ax, ay = stage.anchor_xy
        dx = x - ax
        dy = y - ay

        distance = math.hypot(dx, dy)
        if distance > stage.fov.range_tiles + 1e-9:
            return False
        if distance <= 1e-12:
            return True

        bearing = math.degrees(math.atan2(dy, dx))
        half_angle = stage.fov.angle_deg / 2.0
        angle_delta = abs(_angle_diff_deg(bearing, stage.anchor_heading_deg))
        return angle_delta <= half_angle + 1e-9

    def _reset_agent2_position(self) -> None:
        visible = self.visible_tiles()
        tile = self.rng.choice(visible)
        x, y = self.tile_xy(tile)
        self.agent2.x = x
        self.agent2.y = y
        self.agent2.heading_deg = self.rng.choice([0.0, 90.0, 180.0, 270.0])

    def sample_agent2_move(self) -> None:
        """Random constrained walk inside current visible set."""
        current_heading = self.agent2.heading_deg
        moves = [
            (1, 0, 0.0),
            (0, 1, 90.0),
            (-1, 0, 180.0),
            (0, -1, 270.0),
            (0, 0, current_heading),
        ]

        candidates: list[tuple[int, int, float]] = []
        visible = self.visible_tile_set()
        for dx, dy, heading in moves:
            nx = self.agent2.x + dx
            ny = self.agent2.y + dy
            if not self.is_in_bounds(nx, ny):
                continue
            tile = self.tile_id(nx, ny)
            if tile in visible:
                candidates.append((nx, ny, heading))

        if not candidates:
            return

        nx, ny, heading = self.rng.choice(candidates)
        self.agent2.x = nx
        self.agent2.y = ny
        self.agent2.heading_deg = heading


class StageScheduler(ABC):
    @abstractmethod
    def maybe_advance(self, trainer: "NamingGameTrainer") -> bool:
        raise NotImplementedError


class NoOpScheduler(StageScheduler):
    def maybe_advance(self, trainer: "NamingGameTrainer") -> bool:
        _ = trainer
        return False


class NamingGameTrainer:
    def __init__(
        self,
        env: HalEnv,
        alphabet_size: int = 8,
        max_word_len: int = 4,
        energy_penalty: float = 0.01,
        seed: int = 0,
        scheduler: Optional[StageScheduler] = None,
    ) -> None:
        if alphabet_size <= 0:
            raise ValueError("alphabet_size must be > 0.")
        if max_word_len <= 0:
            raise ValueError("max_word_len must be > 0.")

        self.env = env
        self.alphabet_size = alphabet_size
        self.max_word_len = max_word_len
        self.energy_penalty = energy_penalty
        self.pause_token = alphabet_size
        self.rng = random.Random(seed)
        self.scheduler = scheduler or NoOpScheduler()

        self.tables = NamingTables()
        self.events: list[InteractionEvent] = []
        self.history_success: list[bool] = []
        self.successful_tiles: set[TileId] = set()

        self.step_count = 0
        self.last_prediction_tile: Optional[TileId] = None
        self.last_target_tile: Optional[TileId] = None

    def _generate_word(self) -> Word:
        length = self.rng.randint(1, self.max_word_len)
        return tuple(self.rng.randrange(self.alphabet_size) for _ in range(length))

    def _invent_word(self, inventory: dict[Word, WordStats]) -> Word:
        for _ in range(256):
            candidate = self._generate_word()
            if candidate not in inventory:
                return candidate

        # Dense fallback in the rare event of repeated collisions.
        for length in range(1, self.max_word_len + 1):
            limit = self.alphabet_size**length
            for value in range(limit):
                digits = [0] * length
                carry = value
                for idx in range(length - 1, -1, -1):
                    digits[idx] = carry % self.alphabet_size
                    carry //= self.alphabet_size
                candidate = tuple(digits)
                if candidate not in inventory:
                    return candidate
        raise RuntimeError("Unable to invent a new word.")

    def _word_utility(self, word: Word, stats: WordStats) -> float:
        return stats.smoothed_success_rate() - (self.energy_penalty * len(word))

    def _speak_axis(self, value: int, table: dict[int, dict[Word, WordStats]]) -> Word:
        inventory = table.setdefault(value, {})
        if not inventory:
            invented = self._invent_word(inventory)
            inventory[invented] = WordStats()
            return invented

        ranked = sorted(
            inventory.items(),
            key=lambda item: (
                -self._word_utility(item[0], item[1]),
                len(item[0]),
                item[0],
            ),
        )
        return ranked[0][0]

    def _visible_rel_pairs(self) -> list[tuple[int, int]]:
        return [self.env.relative_coords(tile) for tile in self.env.visible_tiles()]

    def _visible_axis_values(self) -> tuple[list[int], list[int]]:
        pairs = self._visible_rel_pairs()
        x_vals = sorted({pair[0] for pair in pairs})
        y_vals = sorted({pair[1] for pair in pairs})
        return x_vals, y_vals

    def pack_utterance(self, x_word: Word, y_word: Word) -> Word:
        return x_word + (self.pause_token,) + y_word

    def unpack_utterance(self, utterance: Word) -> tuple[Word, Word]:
        if self.pause_token not in utterance:
            return utterance, tuple()
        split = utterance.index(self.pause_token)
        return utterance[:split], utterance[split + 1 :]

    def speak_terms(self, tile_id: TileId) -> tuple[Word, Word]:
        x_rel, y_rel = self.env.relative_coords(tile_id)
        x_word = self._speak_axis(x_rel, self.tables.speaker_x_inventory)
        y_word = self._speak_axis(y_rel, self.tables.speaker_y_inventory)
        return x_word, y_word

    def speak(self, tile_id: TileId) -> Word:
        x_word, y_word = self.speak_terms(tile_id)
        return self.pack_utterance(x_word, y_word)

    def _decode_axis(
        self,
        word: Word,
        assoc_table: dict[Word, dict[int, int]],
        fallback_values: list[int],
        fallback_random: bool,
    ) -> tuple[int, float]:
        assoc = assoc_table.get(word)
        if assoc:
            max_count = max(assoc.values())
            best_values = [value for value, count in assoc.items() if count == max_count]
            best_value = min(best_values)
            total = sum(assoc.values())
            confidence = (max_count / total) if total > 0 else 0.0
            return best_value, confidence

        if not fallback_values:
            raise RuntimeError("No axis fallback values available.")
        value = self.rng.choice(fallback_values) if fallback_random else fallback_values[0]
        return value, 0.0

    def decode_terms(
        self,
        x_word: Word,
        y_word: Word,
        fallback_random: bool = True,
    ) -> tuple[TileId, float, float, float, int, int]:
        x_values, y_values = self._visible_axis_values()
        pred_x_rel, confidence_x = self._decode_axis(
            x_word,
            self.tables.listener_x_assoc,
            x_values,
            fallback_random,
        )
        pred_y_rel, confidence_y = self._decode_axis(
            y_word,
            self.tables.listener_y_assoc,
            y_values,
            fallback_random,
        )

        pred_tile = self.env.tile_from_relative(pred_x_rel, pred_y_rel)
        if pred_tile is None:
            visible = self.env.visible_tiles()
            if not visible:
                raise RuntimeError("No visible tiles available for fallback decode.")
            pred_tile = self.rng.choice(visible) if fallback_random else visible[0]

        confidence = (confidence_x + confidence_y) / 2.0
        return pred_tile, confidence, confidence_x, confidence_y, pred_x_rel, pred_y_rel

    def decode_word(self, word: Word, fallback_random: bool = True) -> tuple[TileId, float]:
        x_word, y_word = self.unpack_utterance(word)
        pred_tile, confidence, _, _, _, _ = self.decode_terms(
            x_word,
            y_word,
            fallback_random=fallback_random,
        )
        return pred_tile, confidence

    def listen(
        self,
        word_or_x_word: Word,
        y_word: Optional[Word] = None,
    ) -> tuple[TileId, float]:
        if y_word is None:
            return self.decode_word(word_or_x_word, fallback_random=True)
        pred_tile, confidence, _, _, _, _ = self.decode_terms(
            word_or_x_word,
            y_word,
            fallback_random=True,
        )
        return pred_tile, confidence

    def _reinforce_listener(
        self,
        table: dict[Word, dict[int, int]],
        word: Word,
        value: int,
    ) -> None:
        assoc = table.setdefault(word, {})
        assoc[value] = assoc.get(value, 0) + 1

    def _update_axis(
        self,
        true_value: int,
        pred_value: int,
        word: Word,
        speaker_table: dict[int, dict[Word, WordStats]],
        listener_table: dict[Word, dict[int, int]],
    ) -> bool:
        inventory = speaker_table.setdefault(true_value, {})
        stats = inventory.setdefault(word, WordStats())
        stats.uses += 1

        success = true_value == pred_value
        if success:
            stats.successes += 1
            for other in [candidate for candidate in inventory.keys() if candidate != word]:
                del inventory[other]
        else:
            stats.failures += 1

        self._reinforce_listener(listener_table, word, true_value)
        return success

    def update_terms(
        self,
        tile_id: TileId,
        x_word: Word,
        y_word: Word,
        pred_tile_id: TileId,
        pred_x_rel: int,
        pred_y_rel: int,
    ) -> bool:
        true_x_rel, true_y_rel = self.env.relative_coords(tile_id)

        x_success = self._update_axis(
            true_value=true_x_rel,
            pred_value=pred_x_rel,
            word=x_word,
            speaker_table=self.tables.speaker_x_inventory,
            listener_table=self.tables.listener_x_assoc,
        )
        y_success = self._update_axis(
            true_value=true_y_rel,
            pred_value=pred_y_rel,
            word=y_word,
            speaker_table=self.tables.speaker_y_inventory,
            listener_table=self.tables.listener_y_assoc,
        )

        success = x_success and y_success and (tile_id == pred_tile_id)
        if success:
            self.successful_tiles.add(tile_id)

        self.history_success.append(success)
        return success

    def update(self, tile_id: TileId, word: Word, pred_tile_id: TileId) -> bool:
        x_word, y_word = self.unpack_utterance(word)
        pred_x_rel, pred_y_rel = self.env.relative_coords(pred_tile_id)
        return self.update_terms(
            tile_id=tile_id,
            x_word=x_word,
            y_word=y_word,
            pred_tile_id=pred_tile_id,
            pred_x_rel=pred_x_rel,
            pred_y_rel=pred_y_rel,
        )

    def rolling_accuracy(self, window: int = 1000) -> float:
        if not self.history_success:
            return 0.0
        if window <= 0 or window >= len(self.history_success):
            sample = self.history_success
        else:
            sample = self.history_success[-window:]
        return sum(sample) / len(sample)

    def coverage_fraction(self) -> float:
        visible = set(self.env.visible_tiles())
        if not visible:
            return 0.0
        covered = len(visible.intersection(self.successful_tiles))
        return covered / len(visible)

    def is_complete(self) -> bool:
        visible = set(self.env.visible_tiles())
        if not visible:
            return False
        coverage_done = visible.issubset(self.successful_tiles)
        return coverage_done and self.rolling_accuracy(window=1000) >= 0.95

    def step(self) -> InteractionEvent:
        target_tile = self.env.agent2_tile_id
        x_word, y_word = self.speak_terms(target_tile)
        utterance = self.pack_utterance(x_word, y_word)

        (
            pred_tile,
            confidence,
            confidence_x,
            confidence_y,
            pred_x_rel,
            pred_y_rel,
        ) = self.decode_terms(x_word, y_word, fallback_random=True)

        success = self.update_terms(
            tile_id=target_tile,
            x_word=x_word,
            y_word=y_word,
            pred_tile_id=pred_tile,
            pred_x_rel=pred_x_rel,
            pred_y_rel=pred_y_rel,
        )

        event = InteractionEvent(
            step=self.step_count,
            word=utterance,
            x_word=x_word,
            y_word=y_word,
            success=success,
            confidence=confidence,
            confidence_x=confidence_x,
            confidence_y=confidence_y,
        )
        self.events.append(event)
        self.last_prediction_tile = pred_tile
        self.last_target_tile = target_tile
        self.step_count += 1

        self.scheduler.maybe_advance(self)
        self.env.sample_agent2_move()
        return event
