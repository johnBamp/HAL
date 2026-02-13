import random

from hal import (
    LanguageAgent,
    TrainingConfig,
    backtrace_factored_phrase,
    build_slot_id_maps,
    color_key,
    initialize_language,
    object_key,
    train_factored_language,
    train_language,
)
from main import is_backtrace_enabled


DEFAULT_COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
DEFAULT_OBJECTS = ["apple", "mushroom"]


def _run_factored(seed=7, steps=220, holdout_pairs=None):
    rng = random.Random(seed)
    cfg = TrainingConfig(learning_steps=steps)

    adam = LanguageAgent("Adam")
    eve = LanguageAgent("Eve")

    semantics = [color_key(c) for c in DEFAULT_COLORS] + [object_key(o) for o in DEFAULT_OBJECTS]
    initialize_language(adam, semantics, cfg, rng)
    initialize_language(eve, semantics, cfg, rng)

    result = train_factored_language(
        adam,
        eve,
        DEFAULT_COLORS,
        DEFAULT_OBJECTS,
        cfg,
        rng,
        holdout_pairs=holdout_pairs,
    )
    return adam, eve, result


def _find_non_exact_token(base_token, occupied, max_token):
    for delta in range(1, max_token):
        above = base_token + delta
        if above < max_token and above not in occupied:
            return above
        below = base_token - delta
        if below >= 0 and below not in occupied:
            return below
    raise AssertionError("could not find non-exact token for nearest fallback test")


def test_factored_default_target_unique_is_nine():
    _, _, result = _run_factored(seed=11)
    assert result.metrics["target_unique"] == 9


def test_factored_default_phrase_row_count_is_color_times_object():
    _, _, result = _run_factored(seed=11)
    assert len(result.phrase_rows) == len(DEFAULT_COLORS) * len(DEFAULT_OBJECTS)


def test_factored_fixed_seed_reaches_full_slot_and_phrase_consensus():
    _, _, result = _run_factored(seed=11)
    assert result.metrics["all_slot_consensus"] is True
    assert result.metrics["all_phrase_consensus"] is True


def test_factored_achieved_unique_matches_target_in_passing_run():
    _, _, result = _run_factored(seed=11)
    assert result.metrics["achieved_unique"] == result.metrics["target_unique"]


def test_factored_backtrace_exact_seen_phrase_decodes():
    adam, _, result = _run_factored(seed=11)
    slot_maps = build_slot_id_maps(adam, DEFAULT_COLORS, DEFAULT_OBJECTS)
    color_token = f"{slot_maps['color_label_to_token']['red']:04d}"
    object_token = f"{slot_maps['object_label_to_token']['apple']:04d}"

    decoded = backtrace_factored_phrase(
        adam,
        color_token=color_token,
        object_token=object_token,
        color_names=DEFAULT_COLORS,
        object_types=DEFAULT_OBJECTS,
        seen_pairs=set(result.seen_pairs),
    )
    assert decoded.decode_quality == "exact"
    assert decoded.decoded_concept_key == "red_apple"
    assert decoded.decoded_concept_label == "Red Apple"
    assert decoded.seen_in_training is True


def test_factored_backtrace_heldout_phrase_decodes_unseen_label():
    holdout_pairs = [("orange", "mushroom")]
    adam, _, result = _run_factored(seed=11, holdout_pairs=holdout_pairs)
    slot_maps = build_slot_id_maps(adam, DEFAULT_COLORS, DEFAULT_OBJECTS)
    color_token = f"{slot_maps['color_label_to_token']['orange']:04d}"
    object_token = f"{slot_maps['object_label_to_token']['mushroom']:04d}"

    decoded = backtrace_factored_phrase(
        adam,
        color_token=color_token,
        object_token=object_token,
        color_names=DEFAULT_COLORS,
        object_types=DEFAULT_OBJECTS,
        seen_pairs=set(result.seen_pairs),
    )
    assert decoded.decoded_concept_key == "orange_mushroom"
    assert decoded.seen_in_training is False
    assert "orange:mushroom" in result.heldout_pairs


def test_factored_backtrace_token_mode_nearest_fallback():
    adam, _, result = _run_factored(seed=11)
    slot_maps = build_slot_id_maps(adam, DEFAULT_COLORS, DEFAULT_OBJECTS)
    color_tokens = set(slot_maps["color_label_to_token"].values())
    base_color_token = slot_maps["color_label_to_token"]["blue"]
    nearby_color_token = _find_non_exact_token(
        base_color_token,
        color_tokens,
        max_token=TrainingConfig().word_space_size,
    )
    object_token = f"{slot_maps['object_label_to_token']['apple']:04d}"

    decoded = backtrace_factored_phrase(
        adam,
        color_token=f"{nearby_color_token:04d}",
        object_token=object_token,
        color_names=DEFAULT_COLORS,
        object_types=DEFAULT_OBJECTS,
        seen_pairs=set(result.seen_pairs),
    )
    assert decoded.decode_quality in {"nearest", "ambiguous"}
    assert decoded.decoded_color_label != "UNKNOWN"


def test_factored_holdout_removes_pairs_from_training_space():
    _, _, result = _run_factored(seed=11, holdout_pairs=[("orange", "mushroom")])
    assert "orange:mushroom" in result.heldout_pairs
    assert "orange:mushroom" not in result.seen_pairs

    heldout_rows = [row for row in result.phrase_rows if row["concept"] == "orange_mushroom"]
    assert len(heldout_rows) == 1
    assert heldout_rows[0]["was_trained"] is False


def test_holistic_backtrace_disabled_behavior():
    assert is_backtrace_enabled("factored") is True
    assert is_backtrace_enabled("holistic") is False


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
