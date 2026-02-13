"""Headless language training for HAL agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import random

import settings


@dataclass
class LanguageAgent:
    name: str
    q_table: dict[str, list[float]] = field(default_factory=dict)
    word_bias: dict[str, list[float]] = field(default_factory=dict)
    current_word: int | None = None


@dataclass
class TrainingConfig:
    word_space_size: int = settings.WORD_SPACE_SIZE
    learning_steps: int = settings.LEARNING_STEPS
    noise_scale: float = settings.NOISE_SCALE
    min_noise_scale: float = settings.MIN_NOISE_SCALE
    wta_gain: float = settings.WTA_GAIN
    wta_decay: float = settings.WTA_DECAY
    comm_lr: float = settings.COMM_LR
    social_lr: float = settings.SOCIAL_LR
    self_penalty_lr: float = settings.SELF_PENALTY_LR
    consensus_streak_target: int = settings.CONSENSUS_STREAK_TARGET


@dataclass
class TrainingResult:
    history_lines: list[str]
    debug_rows: list[dict]
    final_word: str
    converged_step: int | None


@dataclass
class FactoredTrainingResult:
    slot_rows: list[dict]
    phrase_rows: list[dict]
    metrics: dict
    debug_rows: list[dict]
    converged_step: int | None
    history_lines: list[str]


def color_key(color_name: str) -> str:
    return f"color:{color_name}"


def object_key(object_name: str) -> str:
    return f"object:{object_name}"


def format_word(word_int: int | None) -> str:
    if word_int is None:
        return "----"
    return f"{word_int:04d}"


def clamp_word(value: int, word_space_size: int) -> int:
    return max(0, min(word_space_size - 1, value))


def initialize_language(
    agent: LanguageAgent,
    object_types: list[str],
    cfg: TrainingConfig,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    for obj_type in object_types:
        agent.q_table[obj_type] = [0.0] * cfg.word_space_size
        agent.word_bias[obj_type] = [rng.uniform(-0.02, 0.02) for _ in range(cfg.word_space_size)]

        seed_word = rng.randint(0, cfg.word_space_size - 1)
        agent.q_table[obj_type][seed_word] = 0.5
        if agent.current_word is None:
            agent.current_word = seed_word


def _effective_noise(cfg: TrainingConfig, step: int) -> float:
    if cfg.learning_steps <= 1:
        return cfg.min_noise_scale
    anneal = 1.0 - (step / (cfg.learning_steps - 1))
    return max(cfg.min_noise_scale, cfg.noise_scale * anneal)


def recognize_word(
    agent: LanguageAgent,
    obj_type: str,
    noise_scale: float,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random
    q_values = agent.q_table[obj_type]
    biases = agent.word_bias[obj_type]

    best_idx = 0
    best_score = float("-inf")
    for idx in range(len(q_values)):
        score = q_values[idx] + biases[idx] + rng.uniform(-noise_scale, noise_scale)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def best_word(agent: LanguageAgent, obj_type: str) -> int:
    q_values = agent.q_table[obj_type]
    biases = agent.word_bias[obj_type]
    best_idx = 0
    best_score = float("-inf")
    for idx in range(len(q_values)):
        score = q_values[idx] + biases[idx]
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def apply_winner_takes_all(
    agent: LanguageAgent,
    obj_type: str,
    winner_idx: int,
    cfg: TrainingConfig,
) -> None:
    q_values = agent.q_table[obj_type]
    decay = 1.0 - cfg.wta_decay
    for idx in range(len(q_values)):
        q_values[idx] *= decay
    q_values[winner_idx] += cfg.wta_gain


def language_alignment_step(
    adam: LanguageAgent,
    eve: LanguageAgent,
    obj_type: str,
    cfg: TrainingConfig,
    step: int,
    rng: random.Random | None = None,
) -> tuple[int, int, float, int, float]:
    rng = rng or random
    noise = _effective_noise(cfg, step)

    adam_word = recognize_word(adam, obj_type, noise, rng)
    eve_word = recognize_word(eve, obj_type, noise, rng)

    apply_winner_takes_all(adam, obj_type, adam_word, cfg)
    apply_winner_takes_all(eve, obj_type, eve_word, cfg)

    distance = abs(adam_word - eve_word)
    closeness = 1.0 - (distance / (cfg.word_space_size - 1))
    reward = (2.0 * closeness) - 1.0

    adam_q = adam.q_table[obj_type]
    eve_q = eve.q_table[obj_type]

    adam_q[adam_word] += cfg.comm_lr * reward
    eve_q[eve_word] += cfg.comm_lr * reward

    if adam_word != eve_word:
        disagreement = 1.0 - closeness

        # On disagreement, pull both agents toward a shared midpoint token.
        midpoint = clamp_word((adam_word + eve_word) // 2, cfg.word_space_size)
        midpoint_boost = cfg.social_lr * (0.5 + disagreement)
        adam_q[midpoint] += midpoint_boost
        eve_q[midpoint] += midpoint_boost

        adam_q[adam_word] -= cfg.self_penalty_lr * disagreement
        eve_q[eve_word] -= cfg.self_penalty_lr * disagreement

        # Weak local smoothing around each agent's own current token.
        adam_q[clamp_word(adam_word + 1, cfg.word_space_size)] += 0.125 * disagreement
        adam_q[clamp_word(adam_word - 1, cfg.word_space_size)] += 0.125 * disagreement
        eve_q[clamp_word(eve_word + 1, cfg.word_space_size)] += 0.125 * disagreement
        eve_q[clamp_word(eve_word - 1, cfg.word_space_size)] += 0.125 * disagreement

    adam.current_word = adam_word
    eve.current_word = eve_word
    return adam_word, eve_word, reward, distance, noise


def train_language(
    adam: LanguageAgent,
    eve: LanguageAgent,
    obj_type: str,
    cfg: TrainingConfig,
    rng: random.Random | None = None,
) -> TrainingResult:
    rng = rng or random
    history_lines: list[str] = []
    debug_rows: list[dict] = []

    consensus_streak = 0
    converged_step: int | None = None

    for step in range(cfg.learning_steps):
        adam_word, eve_word, reward, distance, noise = language_alignment_step(
            adam,
            eve,
            obj_type,
            cfg,
            step,
            rng,
        )
        if distance == 0:
            consensus_streak += 1
        else:
            consensus_streak = 0

        history_lines.append(
            f"{step:03d}  Adam:{format_word(adam_word)}  Eve:{format_word(eve_word)}  "
            f"d={distance:4d}  r={reward:+.3f}  n={noise:.4f}  streak={consensus_streak:02d}"
        )
        debug_rows.append(
            {
                "step": step,
                "adam_word": adam_word,
                "eve_word": eve_word,
                "distance": distance,
                "reward": reward,
                "noise": noise,
                "consensus_streak": consensus_streak,
            }
        )

        if consensus_streak >= cfg.consensus_streak_target and step > 15:
            converged_step = step
            break

    # Final consensus check is deterministic (no noise/random tie-breaking).
    adam_best = best_word(adam, obj_type)
    eve_best = best_word(eve, obj_type)
    adam.current_word = adam_best
    eve.current_word = eve_best

    final_word = format_word(adam_best) if adam_best == eve_best else "NO_CONSENSUS"

    history_lines.append("-" * 72)
    history_lines.append(f"Final shared word for {obj_type}: {final_word}")
    if converged_step is not None:
        history_lines.append(f"Converged by step: {converged_step}")
    else:
        history_lines.append("Converged by step: NONE")

    return TrainingResult(
        history_lines=history_lines,
        debug_rows=debug_rows,
        final_word=final_word,
        converged_step=converged_step,
    )


def _find_nearest_free_index(preferred_idx: int, blocked: set[int], word_space_size: int) -> int:
    if preferred_idx not in blocked:
        return preferred_idx

    for radius in range(1, word_space_size):
        upper = preferred_idx + radius
        if upper < word_space_size and upper not in blocked:
            return upper

        lower = preferred_idx - radius
        if lower >= 0 and lower not in blocked:
            return lower

    return preferred_idx


def _push_semantic_index(
    agent: LanguageAgent,
    semantic_key: str,
    from_idx: int,
    to_idx: int,
    boost: float,
) -> None:
    q_values = agent.q_table[semantic_key]
    q_values[from_idx] -= 0.6 * boost
    q_values[to_idx] += boost


def _enforce_domain_uniqueness(
    agent: LanguageAgent,
    semantic_keys: list[str],
    cfg: TrainingConfig,
    strength: float,
) -> None:
    occupied: set[int] = set()
    for semantic in semantic_keys:
        idx = best_word(agent, semantic)
        if idx in occupied:
            new_idx = _find_nearest_free_index(idx, occupied, cfg.word_space_size)
            _push_semantic_index(agent, semantic, idx, new_idx, strength)
            idx = new_idx
        occupied.add(idx)


def _enforce_cross_domain_uniqueness(
    agent: LanguageAgent,
    primary_keys: list[str],
    secondary_keys: list[str],
    cfg: TrainingConfig,
    strength: float,
) -> None:
    occupied: set[int] = set(best_word(agent, key) for key in primary_keys)
    for semantic in secondary_keys:
        idx = best_word(agent, semantic)
        if idx in occupied:
            new_idx = _find_nearest_free_index(idx, occupied, cfg.word_space_size)
            _push_semantic_index(agent, semantic, idx, new_idx, strength)
            idx = new_idx
        occupied.add(idx)


def _all_slot_consensus(adam: LanguageAgent, eve: LanguageAgent, semantic_keys: list[str]) -> bool:
    return all(best_word(adam, key) == best_word(eve, key) for key in semantic_keys)


def _all_phrase_consensus(
    adam: LanguageAgent,
    eve: LanguageAgent,
    color_names: list[str],
    object_types: list[str],
) -> bool:
    for color_name in color_names:
        c_key = color_key(color_name)
        if best_word(adam, c_key) != best_word(eve, c_key):
            return False

    for object_type in object_types:
        o_key = object_key(object_type)
        if best_word(adam, o_key) != best_word(eve, o_key):
            return False

    return True


def train_factored_language(
    adam: LanguageAgent,
    eve: LanguageAgent,
    color_names: list[str],
    object_types: list[str],
    cfg: TrainingConfig,
    rng: random.Random | None = None,
) -> FactoredTrainingResult:
    rng = rng or random

    color_semantics = [color_key(color) for color in color_names]
    object_semantics = [object_key(obj) for obj in object_types]
    all_semantics = color_semantics + object_semantics

    concept_specs = [
        (color, obj, color_key(color), object_key(obj))
        for color in color_names
        for obj in object_types
    ]

    debug_rows: list[dict] = []
    history_lines: list[str] = []
    consensus_streak = 0
    converged_step: int | None = None

    for step in range(cfg.learning_steps):
        color_name, object_type, color_sem, object_sem = rng.choice(concept_specs)

        adam_color, eve_color, reward_color, distance_color, noise_color = language_alignment_step(
            adam, eve, color_sem, cfg, step, rng
        )
        adam_object, eve_object, reward_object, distance_object, noise_object = language_alignment_step(
            adam, eve, object_sem, cfg, step, rng
        )

        for agent in (adam, eve):
            _enforce_domain_uniqueness(agent, color_semantics, cfg, strength=1.35)
            _enforce_domain_uniqueness(agent, object_semantics, cfg, strength=1.35)
            _enforce_cross_domain_uniqueness(agent, color_semantics, object_semantics, cfg, strength=0.8)

        slot_consensus = _all_slot_consensus(adam, eve, all_semantics)
        phrase_consensus = _all_phrase_consensus(adam, eve, color_names, object_types)
        if slot_consensus and phrase_consensus:
            consensus_streak += 1
        else:
            consensus_streak = 0

        debug_rows.append(
            {
                "step": step,
                "color": color_name,
                "object": object_type,
                "adam_color_word": adam_color,
                "eve_color_word": eve_color,
                "adam_object_word": adam_object,
                "eve_object_word": eve_object,
                "color_distance": distance_color,
                "object_distance": distance_object,
                "color_reward": reward_color,
                "object_reward": reward_object,
                "color_noise": noise_color,
                "object_noise": noise_object,
                "slot_consensus": slot_consensus,
                "phrase_consensus": phrase_consensus,
                "consensus_streak": consensus_streak,
            }
        )

        history_lines.append(
            f"{step:03d}  {color_name}/{object_type}  "
            f"A:[{format_word(adam_color)},{format_word(adam_object)}]  "
            f"E:[{format_word(eve_color)},{format_word(eve_object)}]  "
            f"slot={int(slot_consensus)} phrase={int(phrase_consensus)} streak={consensus_streak:02d}"
        )

        if consensus_streak >= cfg.consensus_streak_target and step > 15:
            converged_step = step
            break

    slot_rows: list[dict] = []
    for color_name in color_names:
        semantic = color_key(color_name)
        adam_id = format_word(best_word(adam, semantic))
        eve_id = format_word(best_word(eve, semantic))
        shared = adam_id if adam_id == eve_id else "NO_CONSENSUS"
        slot_rows.append(
            {
                "slot_type": "color",
                "slot_label": color_name.title(),
                "semantic_key": semantic,
                "concept_label": f"Color:{color_name.title()}",
                "adam_word": adam_id,
                "eve_word": eve_id,
                "shared_word": shared,
                "converged_step": converged_step,
            }
        )

    for object_type in object_types:
        semantic = object_key(object_type)
        adam_id = format_word(best_word(adam, semantic))
        eve_id = format_word(best_word(eve, semantic))
        shared = adam_id if adam_id == eve_id else "NO_CONSENSUS"
        slot_rows.append(
            {
                "slot_type": "object",
                "slot_label": object_type.title(),
                "semantic_key": semantic,
                "concept_label": f"Object:{object_type.title()}",
                "adam_word": adam_id,
                "eve_word": eve_id,
                "shared_word": shared,
                "converged_step": converged_step,
            }
        )

    phrase_rows: list[dict] = []
    for color_name in color_names:
        c_sem = color_key(color_name)
        for object_type in object_types:
            o_sem = object_key(object_type)

            adam_c = format_word(best_word(adam, c_sem))
            adam_o = format_word(best_word(adam, o_sem))
            eve_c = format_word(best_word(eve, c_sem))
            eve_o = format_word(best_word(eve, o_sem))

            adam_phrase = f"[{adam_c}, {adam_o}]"
            eve_phrase = f"[{eve_c}, {eve_o}]"
            shared_phrase = adam_phrase if adam_c == eve_c and adam_o == eve_o else "NO_CONSENSUS"

            phrase_rows.append(
                {
                    "object_type": object_type,
                    "color_name": color_name,
                    "concept": f"{color_name}_{object_type}",
                    "concept_label": f"{color_name.title()} {object_type.title()}",
                    "syntax_order": "color_object",
                    "adam_phrase_ids": adam_phrase,
                    "eve_phrase_ids": eve_phrase,
                    "shared_phrase_ids": shared_phrase,
                    "converged_step": converged_step,
                }
            )

    shared_slot_words = [row["shared_word"] for row in slot_rows if row["shared_word"] != "NO_CONSENSUS"]
    target_unique = len(color_names) + len(object_types)
    achieved_unique = len(set(shared_slot_words))
    all_phrase_consensus = all(row["shared_phrase_ids"] != "NO_CONSENSUS" for row in phrase_rows)
    all_slot_consensus = all(row["shared_word"] != "NO_CONSENSUS" for row in slot_rows)

    metrics = {
        "target_unique": target_unique,
        "achieved_unique": achieved_unique,
        "all_phrase_consensus": all_phrase_consensus,
        "all_slot_consensus": all_slot_consensus,
        "exact_target": achieved_unique == target_unique,
        "converged_step": converged_step,
    }

    history_lines.append("-" * 72)
    history_lines.append(
        "Factored summary: "
        f"target_unique={target_unique} achieved_unique={achieved_unique} "
        f"phrase_consensus={all_phrase_consensus}"
    )

    return FactoredTrainingResult(
        slot_rows=slot_rows,
        phrase_rows=phrase_rows,
        metrics=metrics,
        debug_rows=debug_rows,
        converged_step=converged_step,
        history_lines=history_lines,
    )


def write_training_logs(
    result: TrainingResult,
    obj_type: str,
    seed: int | None,
    cfg: TrainingConfig,
    artifacts_dir: str = settings.ARTIFACTS_DIR,
) -> dict[str, str]:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repro_steps_path = out_dir / f"repro_{cfg.learning_steps}.json"
    repro_print_json_path = out_dir / "repro_print.json"
    repro_print_txt_path = out_dir / "repro_print.txt"

    payload = {
        "object_type": obj_type,
        "seed": seed,
        "config": {
            "word_space_size": cfg.word_space_size,
            "learning_steps": cfg.learning_steps,
            "noise_scale": cfg.noise_scale,
            "min_noise_scale": cfg.min_noise_scale,
            "wta_gain": cfg.wta_gain,
            "wta_decay": cfg.wta_decay,
            "comm_lr": cfg.comm_lr,
            "social_lr": cfg.social_lr,
            "self_penalty_lr": cfg.self_penalty_lr,
            "consensus_streak_target": cfg.consensus_streak_target,
        },
        "final_word": result.final_word,
        "converged_step": result.converged_step,
        "rows": result.debug_rows,
    }

    repro_steps_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    repro_print_json_path.write_text(
        json.dumps(
            {
                "object_type": obj_type,
                "final_word": result.final_word,
                "converged_step": result.converged_step,
                "lines": result.history_lines,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    repro_print_txt_path.write_text("\n".join(result.history_lines) + "\n", encoding="utf-8")

    return {
        "repro_steps_json": str(repro_steps_path),
        "repro_print_json": str(repro_print_json_path),
        "repro_print_txt": str(repro_print_txt_path),
    }


def write_lexicon_logs(
    results_by_object: dict[str, TrainingResult] | None,
    lexicon_rows: list[dict],
    seed: int | None,
    cfg: TrainingConfig,
    artifacts_dir: str = settings.ARTIFACTS_DIR,
    mode: str = "holistic",
    syntax_order: str = "color_object",
    slot_lexicon: list[dict] | None = None,
    phrase_lexicon: list[dict] | None = None,
    metrics: dict | None = None,
) -> dict[str, str]:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repro_steps_path = out_dir / f"repro_{cfg.learning_steps}.json"
    repro_print_json_path = out_dir / "repro_print.json"
    repro_print_txt_path = out_dir / "repro_print.txt"

    slot_lexicon = slot_lexicon or []
    phrase_lexicon = phrase_lexicon or []
    metrics = metrics or {}
    results_by_object = results_by_object or {}

    payload = {
        "seed": seed,
        "mode": mode,
        "syntax_order": syntax_order,
        "config": {
            "word_space_size": cfg.word_space_size,
            "learning_steps": cfg.learning_steps,
            "noise_scale": cfg.noise_scale,
            "min_noise_scale": cfg.min_noise_scale,
            "wta_gain": cfg.wta_gain,
            "wta_decay": cfg.wta_decay,
            "comm_lr": cfg.comm_lr,
            "social_lr": cfg.social_lr,
            "self_penalty_lr": cfg.self_penalty_lr,
            "consensus_streak_target": cfg.consensus_streak_target,
        },
        "metrics": metrics,
        "slot_lexicon": slot_lexicon,
        "phrase_lexicon": phrase_lexicon,
        # Backwards-compatible alias.
        "lexicon": lexicon_rows,
        "objects": {
            obj: {
                "final_word": result.final_word,
                "converged_step": result.converged_step,
                "rows": result.debug_rows,
            }
            for obj, result in results_by_object.items()
        },
    }
    repro_steps_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_lines = ["HAL Lexicon", "-" * 44, f"Mode: {mode}", f"Syntax: {syntax_order}"]
    if metrics:
        print_lines.append(
            "Metrics: "
            f"target_unique={metrics.get('target_unique', 'n/a')} "
            f"achieved_unique={metrics.get('achieved_unique', 'n/a')} "
            f"phrase_consensus={metrics.get('all_phrase_consensus', 'n/a')}"
        )

    if slot_lexicon:
        print_lines.append("Base Factored Lexicon")
        print_lines.append("-" * 44)
        for row in slot_lexicon:
            print_lines.append(
                f"{row.get('slot_type', '?'):<7} {row.get('slot_label', '?'):<12} "
                f"Adam:{row.get('adam_word', '----')} Eve:{row.get('eve_word', '----')} "
                f"Shared:{row.get('shared_word', 'NO_CONSENSUS')}"
            )

    section_rows = phrase_lexicon if phrase_lexicon else lexicon_rows
    if section_rows:
        print_lines.append("Composed Phrases")
        print_lines.append("-" * 44)
        for row in section_rows:
            concept_name = row.get("concept_label", row.get("object_type", "concept"))
            adam_text = row.get("adam_phrase_ids", row.get("adam_word", "----"))
            eve_text = row.get("eve_phrase_ids", row.get("eve_word", "----"))
            shared_text = row.get("shared_phrase_ids", row.get("shared_word", "NO_CONSENSUS"))
            print_lines.append(
                f"{concept_name:<20} Adam:{adam_text:<14} Eve:{eve_text:<14} Shared:{shared_text}"
            )

    repro_print_json_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "mode": mode,
                "syntax_order": syntax_order,
                "metrics": metrics,
                "slot_lexicon": slot_lexicon,
                "phrase_lexicon": phrase_lexicon,
                "lexicon": lexicon_rows,
                "lines": print_lines,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    repro_print_txt_path.write_text("\n".join(print_lines) + "\n", encoding="utf-8")

    return {
        "repro_steps_json": str(repro_steps_path),
        "repro_print_json": str(repro_print_json_path),
        "repro_print_txt": str(repro_print_txt_path),
    }


def sweep_social_pressures(
    object_type: str = "apple",
    seeds: int = 24,
    social_values: list[float] | None = None,
    cfg: TrainingConfig | None = None,
    artifacts_dir: str = settings.ARTIFACTS_DIR,
) -> str:
    cfg = cfg or TrainingConfig()
    social_values = social_values or [0.25, 0.45, 0.7, 0.95, 1.15, 1.4]

    summary = []
    for social_lr in social_values:
        converged = 0
        final_consensus = 0
        steps_to_converge: list[int] = []

        for seed in range(seeds):
            rng = random.Random(seed)
            run_cfg = TrainingConfig(**{**cfg.__dict__, "social_lr": social_lr})
            adam = LanguageAgent("Adam")
            eve = LanguageAgent("Eve")
            initialize_language(adam, [object_type], run_cfg, rng)
            initialize_language(eve, [object_type], run_cfg, rng)
            result = train_language(adam, eve, object_type, run_cfg, rng)

            if result.converged_step is not None:
                converged += 1
                steps_to_converge.append(result.converged_step)
            if result.final_word != "NO_CONSENSUS":
                final_consensus += 1

        avg_step = (sum(steps_to_converge) / len(steps_to_converge)) if steps_to_converge else None
        summary.append(
            {
                "social_lr": social_lr,
                "runs": seeds,
                "convergence_rate": converged / seeds,
                "final_consensus_rate": final_consensus / seeds,
                "avg_converged_step": avg_step,
            }
        )

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_soft_pressures_debug.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(out_path)
