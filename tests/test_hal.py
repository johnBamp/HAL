from __future__ import annotations

from hal import (
    FOVConfig,
    GridConfig,
    HalEnv,
    NamingGameTrainer,
    StageConfig,
    WordStats,
)


def test_fov_boundaries_and_rejections() -> None:
    env = HalEnv(seed=10)
    stage = env.current_stage
    ax, ay = stage.anchor_xy

    boundary_up = env.tile_id(ax + 7, ay - 7)
    boundary_down = env.tile_id(ax + 7, ay + 7)
    outside_angle = env.tile_id(ax + 6, ay + 7)
    outside_range = env.tile_id(ax + 11, ay)

    assert env.is_in_fov(boundary_up, stage)
    assert env.is_in_fov(boundary_down, stage)
    assert not env.is_in_fov(outside_angle, stage)
    assert not env.is_in_fov(outside_range, stage)


def test_constrained_random_walk_stays_visible() -> None:
    env = HalEnv(seed=22)
    visible = set(env.visible_tiles())
    for _ in range(5000):
        env.sample_agent2_move()
        assert env.agent2_tile_id in visible


def test_unknown_terms_fallback_returns_visible_tile() -> None:
    env = HalEnv(seed=3)
    trainer = NamingGameTrainer(env=env, seed=4)

    tile, confidence = trainer.listen((7, 7, 7), (6, 6, 6))
    assert tile in set(env.visible_tiles())
    assert confidence == 0.0


def test_success_update_prunes_competing_synonyms_per_axis() -> None:
    env = HalEnv(seed=5)
    trainer = NamingGameTrainer(env=env, seed=6)
    tile = env.visible_tiles()[0]
    x_rel, y_rel = env.relative_coords(tile)

    x_primary = (1,)
    x_other = (2, 2)
    y_primary = (3,)
    y_other = (4, 4)

    trainer.tables.speaker_x_inventory[x_rel] = {
        x_primary: WordStats(successes=1, failures=0, uses=1),
        x_other: WordStats(successes=1, failures=0, uses=1),
    }
    trainer.tables.speaker_y_inventory[y_rel] = {
        y_primary: WordStats(successes=1, failures=0, uses=1),
        y_other: WordStats(successes=1, failures=0, uses=1),
    }

    success = trainer.update_terms(
        tile_id=tile,
        x_word=x_primary,
        y_word=y_primary,
        pred_tile_id=tile,
        pred_x_rel=x_rel,
        pred_y_rel=y_rel,
    )

    assert success
    assert list(trainer.tables.speaker_x_inventory[x_rel].keys()) == [x_primary]
    assert list(trainer.tables.speaker_y_inventory[y_rel].keys()) == [y_primary]
    assert trainer.tables.listener_x_assoc[x_primary][x_rel] >= 1
    assert trainer.tables.listener_y_assoc[y_primary][y_rel] >= 1


def test_failure_update_reinforces_true_axis_mapping() -> None:
    env = HalEnv(seed=7)
    trainer = NamingGameTrainer(env=env, seed=8)
    tile = env.visible_tiles()[0]
    other_tile = env.visible_tiles()[1]
    true_x, true_y = env.relative_coords(tile)
    pred_x, pred_y = env.relative_coords(other_tile)

    x_word = (3, 3)
    y_word = (0, 0)

    success = trainer.update_terms(
        tile_id=tile,
        x_word=x_word,
        y_word=y_word,
        pred_tile_id=other_tile,
        pred_x_rel=pred_x,
        pred_y_rel=pred_y,
    )

    assert not success
    x_stats = trainer.tables.speaker_x_inventory[true_x][x_word]
    y_stats = trainer.tables.speaker_y_inventory[true_y][y_word]
    assert x_stats.uses == 1
    assert y_stats.uses == 1
    assert x_stats.failures == 1
    assert y_stats.failures == 1
    assert trainer.tables.listener_x_assoc[x_word][true_x] == 1
    assert trainer.tables.listener_y_assoc[y_word][true_y] == 1


def test_utility_prefers_shorter_x_term_when_stats_equal() -> None:
    env = HalEnv(seed=11)
    trainer = NamingGameTrainer(env=env, seed=12, energy_penalty=0.01)
    tile = env.visible_tiles()[0]
    x_rel, y_rel = env.relative_coords(tile)

    short_word = (1,)
    long_word = (1, 1, 1)

    trainer.tables.speaker_x_inventory[x_rel] = {
        short_word: WordStats(successes=1, failures=1, uses=2),
        long_word: WordStats(successes=1, failures=1, uses=2),
    }
    trainer.tables.speaker_y_inventory[y_rel] = {(2,): WordStats(successes=1, failures=0, uses=1)}

    x_word, _ = trainer.speak_terms(tile)
    assert x_word == short_word


def test_completion_requires_coverage_and_accuracy() -> None:
    grid = GridConfig(width=10, height=10)
    stage = StageConfig(
        anchor_xy=(5, 5),
        anchor_heading_deg=0.0,
        fov=FOVConfig(angle_deg=360.0, range_tiles=2.0),
    )
    env = HalEnv(grid_config=grid, stages=[stage], seed=0)
    trainer = NamingGameTrainer(env=env, seed=1)
    visible = set(env.visible_tiles())

    trainer.successful_tiles = set(list(visible)[:-1])
    trainer.history_success = [True] * 1000
    assert not trainer.is_complete()

    trainer.successful_tiles = set(visible)
    trainer.history_success = [True] * 900 + [False] * 100
    assert not trainer.is_complete()

    trainer.history_success = [True] * 980 + [False] * 20
    assert trainer.is_complete()


def test_deterministic_trace_for_fixed_seed() -> None:
    env_a = HalEnv(seed=19)
    env_b = HalEnv(seed=19)
    trainer_a = NamingGameTrainer(env=env_a, seed=29)
    trainer_b = NamingGameTrainer(env=env_b, seed=29)

    trace_a = []
    trace_b = []
    for _ in range(200):
        event_a = trainer_a.step()
        event_b = trainer_b.step()
        trace_a.append(
            (
                event_a.word,
                event_a.success,
                round(event_a.confidence, 8),
                env_a.agent2_tile_id,
            )
        )
        trace_b.append(
            (
                event_b.word,
                event_b.success,
                round(event_b.confidence, 8),
                env_b.agent2_tile_id,
            )
        )

    assert trace_a == trace_b


def test_compositional_decode_combines_unseen_x_y_pair() -> None:
    grid = GridConfig(width=11, height=11)
    stage = StageConfig(
        anchor_xy=(5, 5),
        anchor_heading_deg=0.0,
        fov=FOVConfig(angle_deg=360.0, range_tiles=3.0),
    )
    env = HalEnv(grid_config=grid, stages=[stage], seed=13)
    trainer = NamingGameTrainer(env=env, seed=14)

    x_word = (1,)
    y_word = (2,)
    trainer.tables.listener_x_assoc[x_word] = {1: 5}
    trainer.tables.listener_y_assoc[y_word] = {1: 7}

    pred_tile, confidence, confidence_x, confidence_y, pred_x, pred_y = trainer.decode_terms(
        x_word,
        y_word,
        fallback_random=False,
    )

    expected = env.tile_from_relative(1, 1)
    assert expected is not None
    assert pred_x == 1
    assert pred_y == 1
    assert pred_tile == expected
    assert confidence_x == 1.0
    assert confidence_y == 1.0
    assert confidence == 1.0
