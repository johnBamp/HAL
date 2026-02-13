import random

from hal import (
    LanguageAgent,
    TrainingConfig,
    color_key,
    initialize_language,
    object_key,
    train_factored_language,
    train_language,
)


DEFAULT_COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
DEFAULT_OBJECTS = ["apple", "mushroom"]


def _run_factored(seed=7, steps=220):
    rng = random.Random(seed)
    cfg = TrainingConfig(learning_steps=steps)

    adam = LanguageAgent("Adam")
    eve = LanguageAgent("Eve")

    semantics = [color_key(c) for c in DEFAULT_COLORS] + [object_key(o) for o in DEFAULT_OBJECTS]
    initialize_language(adam, semantics, cfg, rng)
    initialize_language(eve, semantics, cfg, rng)

    return train_factored_language(adam, eve, DEFAULT_COLORS, DEFAULT_OBJECTS, cfg, rng)


def test_factored_default_target_unique_is_nine():
    result = _run_factored(seed=11)
    assert result.metrics["target_unique"] == 9


def test_factored_default_phrase_row_count_is_color_times_object():
    result = _run_factored(seed=11)
    assert len(result.phrase_rows) == len(DEFAULT_COLORS) * len(DEFAULT_OBJECTS)


def test_factored_fixed_seed_reaches_full_slot_and_phrase_consensus():
    result = _run_factored(seed=11)
    assert result.metrics["all_slot_consensus"] is True
    assert result.metrics["all_phrase_consensus"] is True


def test_factored_achieved_unique_matches_target_in_passing_run():
    result = _run_factored(seed=11)
    assert result.metrics["achieved_unique"] == result.metrics["target_unique"]


def test_holistic_single_concept_training_still_works():
    cfg = TrainingConfig(learning_steps=220)
    rng = random.Random(19)

    concept = "blue_mushroom"
    adam = LanguageAgent("Adam")
    eve = LanguageAgent("Eve")
    initialize_language(adam, [concept], cfg, rng)
    initialize_language(eve, [concept], cfg, rng)

    result = train_language(adam, eve, concept, cfg, rng)
    assert result.final_word != "NO_CONSENSUS"
