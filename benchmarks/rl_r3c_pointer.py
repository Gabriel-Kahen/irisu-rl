#!/usr/bin/env python3
"""Short, non-canonical development benchmark for the R3c pointer policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, EventKind, IrisuEnv, find_library
from irisu_pointer.action import (
    PointerActionSpec,
    PointerActionTensor,
    decode_pointer_action,
)
from irisu_pointer.dataset import PointerDataset
from irisu_pointer.distill import PointerBCConfig, PointerBCMetrics, PointerBCTrainer
from irisu_pointer.experts import PointerExpertDecision, expert_anchors, matcher_anchor
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_pointer.search import (
    SpawnCensoredSearchTeacher,
    lower_expert_decision,
    ticks_before_next_spawn,
)
from irisu_rl.actions import ActionSpec, SemanticActionKind
from irisu_rl.encoding import EncodedBatch, TeacherStateEncoder
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728
TRUSTED_PORTABLE_LIBRARY = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
PRIOR_FAILED_RUN = {
    "path": "/tmp/rl-r3c-pointer-failed-20260728.json",
    "json_sha256": "3e9081d73139105bc81fe68e498c686c947b030f3c29ac00caa3209649562430",
    "heldout_kind_accuracy": 0.75,
    "heldout_target_accuracy": 0.0,
    "heldout_template_accuracy": 1.0,
    "heldout_actionable_recall": 1.0,
    "selected_target_hit_rate_among_predicted_shots": 0.0,
    "preserved_as_failed_evidence": True,
}
PRIOR_FAILED_BALANCED_RUN = {
    "path": "/tmp/rl-r3c-pointer-v2-20260728.json",
    "json_sha256": "04d197b62e62a8ba61dd60a118e40b419981adc7383309268d9fab9d24b3a235",
    "heldout_kind_accuracy": 0.9444444179534912,
    "heldout_target_accuracy": 0.375,
    "heldout_template_accuracy": 1.0,
    "heldout_actionable_recall": 1.0,
    "selected_target_hit_rate_among_predicted_shots": 0.5416666666666666,
    "matcher_oracle_selected_target_hit_rate": 0.8541666666666666,
    "preserved_as_failed_evidence": True,
}
PRIOR_FAILED_RELATION_RUN = {
    "path": "/tmp/rl-r3c-pointer-v2-relation-rerun.json",
    "json_sha256": "3ccac2f4cdbad8cc4e0ce82f348036b81f2c12cc8c8756bda36c770a38edf4c7",
    "heldout_kind_accuracy": 0.9166666865348816,
    "heldout_target_accuracy": 0.3958333432674408,
    "heldout_template_accuracy": 1.0,
    "heldout_actionable_recall": 1.0,
    "selected_target_hit_rate_among_predicted_shots": 0.8541666666666666,
    "matcher_oracle_selected_target_hit_rate": 0.8958333333333334,
    "preserved_as_failed_evidence": True,
}
PRIOR_FAILED_DUAL_PRIOR_RUN = {
    "path": "/tmp/rl-r3c-pointer-dual-prior-failed-20260728.json",
    "json_sha256": "b2e82523331bbb039ced71d7b8b24d8c252b535ce1430247ec48ec3f11cf8562",
    "controlled_heldout_target_accuracy": 0.875,
    "crowded_heldout_target_accuracy": 0.625,
    "controlled_selected_target_hit_rate": 0.875,
    "crowded_selected_target_hit_rate": 0.9375,
    "preserved_as_failed_evidence": True,
}


@dataclass(frozen=True, slots=True)
class Record:
    episode: str
    config_index: int
    expected_kind: int
    observation: dict[str, Any]
    snapshot: bytes
    decision: PointerExpertDecision
    label_source: str
    value_target: float


def _event_kind(event: dict[str, Any]) -> int | None:
    raw = event.get("kind")
    return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None


def _target_index(
    encoded: EncodedBatch,
    row: int,
    decision: PointerExpertDecision,
) -> int:
    if decision.target_body_id is None:
        return 0
    id_column = TEACHER_V1.body_features.index("id_scaled")
    visible = np.flatnonzero(encoded.body_mask[row])
    matching = [
        int(index)
        for index in visible
        if round(float(encoded.body_features[row, index, id_column]) * 2**32)
        == decision.target_body_id
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"target body {decision.target_body_id} did not map to one encoded row"
        )
    return matching[0]


def _labels(
    encoded: EncodedBatch,
    records: list[Record],
    spec: PointerActionSpec,
) -> PointerActionTensor:
    kinds: list[int] = []
    waits: list[int] = []
    targets: list[int] = []
    templates: list[int] = []
    for row, record in enumerate(records):
        decision = record.decision
        kinds.append(int(decision.kind))
        waits.append(spec.wait_choices.index(decision.wait_ticks))
        targets.append(_target_index(encoded, row, decision))
        templates.append(decision.template_index)
    return PointerActionTensor(
        torch.tensor(kinds, dtype=torch.long),
        torch.tensor(waits, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(templates, dtype=torch.long),
    )


def _build_dataset(
    records: list[Record],
    spec: PointerActionSpec,
) -> tuple[PointerDataset, list[tuple[int, ...]]]:
    encoded = TeacherStateEncoder().encode(
        [record.observation for record in records]
    )
    labels = _labels(encoded, records, spec)
    id_column = TEACHER_V1.body_features.index("id_scaled")
    body_id_rows = [
        tuple(
            round(float(encoded.body_features[row, index, id_column]) * 2**32)
            for index in range(encoded.schema.capacity)
        )
        for row in range(len(records))
    ]
    encoded.body_features[:, :, id_column] = 0.0
    if bool(np.any(encoded.body_features[:, :, id_column] != 0.0)):
        raise RuntimeError("ID-feature ablation failed")
    return (
        PointerDataset.from_encoded_batch(
            encoded,
            labels,
            [record.episode for record in records],
            [record.value_target for record in records],
            pointer_spec=spec,
        ),
        body_id_rows,
    )


def _episode_configs(
    episodes: int, *, actionable_falling_count: int
) -> tuple[dict[str, Any], ...]:
    configs: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        kind = episode_index % 3
        variation = episode_index // 3
        if kind == 0:
            falling_count = 0
            falling_y = 90.0 + 15.0 * (variation % 4)
        elif kind == 1:
            falling_count = actionable_falling_count
            falling_y = 255.0 + 10.0 * (variation % 4)
        else:
            falling_count = actionable_falling_count
            falling_y = 75.0 + 20.0 * (variation % 4)
        configs.append(
            {
                "gravity_y": 150.0 + 50.0 * (variation % 4),
                "piece_sizes": (4.0,) * 10,
                "initial_rotten_count": 4,
                "initial_falling_count": falling_count,
                "initial_falling_y": falling_y,
                "starting_colors": 3,
                "maximum_colors": 6,
                "spawn_interval_ticks": 300 + 25 * (variation % 3),
                "max_episode_ticks": 500,
            }
        )
    return tuple(configs)


def _matches_requested_class(
    observation: dict[str, Any],
    expected_kind: int,
    spec: PointerActionSpec,
    *,
    require_unambiguous_pair: bool,
) -> bool:
    if int(matcher_anchor(observation, spec).kind) != expected_kind:
        return False
    if not require_unambiguous_pair:
        return True
    pieces = [
        body
        for body in observation.get("bodies", ())
        if body.get("kind") == "piece"
        and body.get("lifecycle") in {"scripted_falling", "dynamic_fresh"}
    ]
    colors = Counter(int(body.get("color", -1)) for body in pieces)
    same_color_pairs = sum(count * (count - 1) // 2 for count in colors.values())
    expected_pairs = 0 if expected_kind == 0 else 1
    return (
        same_color_pairs == expected_pairs
    )


def _select_episode_seeds(
    library: Path,
    configs: tuple[dict[str, Any], ...],
    observations_per_episode: int,
    spec: PointerActionSpec,
    *,
    require_unambiguous_pair: bool,
) -> tuple[tuple[int, ...], int]:
    selected: list[int] = []
    candidate_seed = SEED
    candidates_examined = 0
    for episode_index, config in enumerate(configs):
        expected_kind = episode_index % 3
        for _ in range(512):
            seed = candidate_seed
            candidate_seed += 1
            candidates_examined += 1
            with IrisuEnv(
                library_path=library,
                physics_backend="portable",
                config=config,
            ) as env:
                observation, _ = env.reset(seed=seed)
                observation = env.step(
                    Action.wait(1 + (episode_index // 3) % 3)
                )[0]
                accepted = 0
                attempts = 0
                while accepted < observations_per_episode:
                    if observation["terminated"] or observation["truncated"]:
                        break
                    accepted += int(
                        _matches_requested_class(
                            observation,
                            expected_kind,
                            spec,
                            require_unambiguous_pair=require_unambiguous_pair,
                        )
                    )
                    attempts += 1
                    observation = env.step(
                        Action.wait(
                            (3, 5, 7, 11)[(attempts + episode_index) % 4]
                        )
                    )[0]
                    if attempts > observations_per_episode * 12:
                        break
            if accepted == observations_per_episode:
                selected.append(seed)
                break
        else:
            raise RuntimeError(
                f"could not find a deterministic seed for class {expected_kind}"
            )
    return tuple(selected), candidates_examined


def _collect(
    library: Path,
    *,
    episodes: int,
    observations_per_episode: int,
    search_audits: int,
    spec: PointerActionSpec,
    challenge_name: str,
    actionable_falling_count: int,
    require_unambiguous_pair: bool,
) -> tuple[
    list[Record],
    tuple[dict[str, Any], ...],
    list[int],
    dict[str, Any],
]:
    records: list[Record] = []
    configs = _episode_configs(
        episodes, actionable_falling_count=actionable_falling_count
    )
    episode_seeds, candidates_examined = _select_episode_seeds(
        library,
        configs,
        observations_per_episode,
        spec,
        require_unambiguous_pair=require_unambiguous_pair,
    )
    config_hashes: list[int] = []
    search_selected: Counter[str] = Counter()
    searched = 0
    kind_agreements = 0
    target_agreements = 0
    teacher = SpawnCensoredSearchTeacher(
        pointer_spec=spec,
        candidate_generator=lambda observation, pointer_spec: expert_anchors(
            observation, pointer_spec
        ),
    )
    remaining_search = search_audits
    for episode_index, episode_seed in enumerate(episode_seeds):
        expected_kind = episode_index % 3
        config = configs[episode_index]
        episode = f"{challenge_name}/class-{expected_kind}/seed-{episode_seed}"
        with IrisuEnv(
            library_path=library,
            physics_backend="portable",
            config=config,
        ) as env:
            observation, info = env.reset(seed=episode_seed)
            config_hashes.append(int(info["config_hash"]))
            observation = env.step(Action.wait(1 + (episode_index // 3) % 3))[0]
            accepted = 0
            attempts = 0
            while accepted < observations_per_episode:
                if observation["terminated"] or observation["truncated"]:
                    break
                decision = matcher_anchor(observation, spec)
                if _matches_requested_class(
                    observation,
                    expected_kind,
                    spec,
                    require_unambiguous_pair=require_unambiguous_pair,
                ):
                    snapshot = bytes(env.clone_state())
                    if (
                        remaining_search > 0
                        and ticks_before_next_spawn(observation) >= 2
                    ):
                        before = bytes(env.clone_state())
                        result = teacher.search(env, observation)
                        if bytes(env.clone_state()) != before:
                            raise RuntimeError("search teacher mutated its source state")
                        searched += 1
                        kind_agreements += int(
                            int(result.decision.kind) == int(decision.kind)
                        )
                        target_agreements += int(
                            result.decision.target_body_id
                            == decision.target_body_id
                        )
                        search_selected[result.selected_name] += 1
                        remaining_search -= 1
                    records.append(
                        Record(
                            episode,
                            episode_index,
                            expected_kind,
                            observation,
                            snapshot,
                            decision,
                            "matcher_anchor",
                            1.0 if decision.target_body_id is not None else 0.0,
                        )
                    )
                    accepted += 1
                attempts += 1
                observation = env.step(
                    Action.wait((3, 5, 7, 11)[(attempts + episode_index) % 4])
                )[0]
                if attempts > observations_per_episode * 12:
                    break
            if accepted != observations_per_episode:
                raise RuntimeError(
                    f"{episode} yielded {accepted}/{observations_per_episode} "
                    f"examples of requested class {expected_kind}"
                )
    return (
        records,
        configs,
        sorted(set(config_hashes)),
        {
            "requested": search_audits,
            "completed": searched,
            "kind_agreement_rate": kind_agreements / max(searched, 1),
            "target_agreement_rate": target_agreements / max(searched, 1),
            "selected_names": dict(sorted(search_selected.items())),
            "labels_used_for_training": False,
            "episode_seeds": list(episode_seeds),
            "seed_candidates_examined": candidates_examined,
            "selection_rule": (
                (
                    "zero same-color pairs for wait; exactly one same-color pair "
                    "for weak/strong, with matcher kind equal to the requested class"
                )
                if require_unambiguous_pair
                else "matcher kind equals the requested class; multiple competing "
                "same-color pairs are retained"
            ),
        },
    )


def _metric_summary(metrics: PointerBCMetrics) -> dict[str, float | int]:
    return {
        "examples": metrics.examples,
        "wait_examples": metrics.wait_examples,
        "actionable_examples": metrics.actionable_examples,
        "weak_examples": metrics.weak_examples,
        "strong_examples": metrics.strong_examples,
        "total_loss": metrics.total_loss,
        "kind_accuracy": metrics.kind_accuracy,
        "wait_accuracy": metrics.wait_accuracy,
        "target_accuracy": metrics.target_accuracy,
        "template_accuracy": metrics.template_accuracy,
        "weak_target_accuracy": metrics.weak_target_accuracy,
        "strong_target_accuracy": metrics.strong_target_accuracy,
        "weak_template_accuracy": metrics.weak_template_accuracy,
        "strong_template_accuracy": metrics.strong_template_accuracy,
        "weak_recall": metrics.weak_recall,
        "strong_recall": metrics.strong_recall,
        "actionable_recall": metrics.actionable_recall,
        "wait_only_rate": metrics.wait_only_rate,
    }


def _class_counts(dataset: PointerDataset) -> dict[str, int]:
    counts = Counter(int(example.label.kind.item()) for example in dataset)
    return {
        "wait": counts[0],
        "weak": counts[1],
        "strong": counts[2],
    }


def _stratified_episode_split(
    dataset: PointerDataset,
    records: list[Record],
) -> tuple[PointerDataset, PointerDataset]:
    episodes_by_kind: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for record in records:
        values = episodes_by_kind[record.expected_kind]
        if record.episode not in values:
            values.append(record.episode)
    validation_ids: set[str] = set()
    for kind, identities in episodes_by_kind.items():
        if len(identities) < 2:
            raise ValueError(f"class {kind} needs at least two episode identities")
        selected = identities[3::4] or identities[-1:]
        validation_ids.update(selected)
    training = PointerDataset(
        [
            example
            for example in dataset
            if example.episode_identity not in validation_ids
        ]
    )
    validation = PointerDataset(
        [
            example
            for example in dataset
            if example.episode_identity in validation_ids
        ]
    )
    if set(training.episode_identities) & set(validation.episode_identities):
        raise RuntimeError("episode identity leaked across the split")
    return training, validation


def _predict(
    model: EntityPointerActorCritic,
    dataset: PointerDataset,
) -> list[tuple[int, int, int, int]]:
    batch = dataset.as_tensors()
    model.eval()
    with torch.no_grad():
        output = model(
            batch.global_features.unsqueeze(0),
            batch.body_features.unsqueeze(0),
            batch.body_mask.unsqueeze(0),
            model.initial_state(batch.size),
            reset_before=torch.ones((1, batch.size), dtype=torch.bool),
        )
    kind = output.kind_logits[0].argmax(-1)
    wait = output.wait_logits[0].argmax(-1)
    rows = torch.arange(batch.size)
    branch = (kind - 1).clamp_min(0)
    target = output.target_logits[0, rows, branch].argmax(-1)
    template = output.template_logits[0, rows, branch, target].argmax(-1)
    return [
        tuple(int(value[index]) for value in (kind, wait, target, template))
        for index in range(batch.size)
    ]


def _execute_actions(
    env: IrisuEnv,
    initial_tick: int,
    actions: list[Action],
    horizon: int,
) -> tuple[Counter[int], bool, int, list[dict[str, Any]]]:
    counts: Counter[int] = Counter()
    events: list[dict[str, Any]] = []
    observation: dict[str, Any] | None = None
    terminal = False
    for action in actions:
        if terminal:
            break
        observation, _, terminated, truncated, info = env.step(action)
        events.extend(dict(event) for event in info.get("events", ()))
        counts.update(
            kind
            for event in info.get("events", ())
            if (kind := _event_kind(event)) is not None
        )
        terminal = bool(terminated or truncated)
    elapsed = (
        int(observation["tick"]) - initial_tick if observation is not None else 0
    )
    if not terminal and elapsed < horizon:
        observation, _, terminated, truncated, info = env.step(
            Action.wait(horizon - elapsed)
        )
        events.extend(dict(event) for event in info.get("events", ()))
        counts.update(
            kind
            for event in info.get("events", ())
            if (kind := _event_kind(event)) is not None
        )
        terminal = bool(terminated or truncated)
        elapsed = int(observation["tick"]) - initial_tick
    return counts, terminal, elapsed, events


def _causal_evaluation(
    library: Path,
    records: list[Record],
    configs: tuple[dict[str, Any], ...],
    body_id_rows: list[tuple[int, ...]],
    validation: PointerDataset,
    predictions: list[tuple[int, int, int, int]],
    *,
    horizon: int,
    spec: PointerActionSpec,
) -> dict[str, Any]:
    tensors = validation.as_tensors()
    heldout_ids = set(validation.episode_identities)
    heldout = [record for record in records if record.episode in heldout_ids]
    heldout_body_ids = [
        ids
        for record, ids in zip(records, body_id_rows)
        if record.episode in heldout_ids
    ]
    if len(heldout) != len(predictions) or len(heldout_body_ids) != len(heldout):
        raise RuntimeError("held-out records and predictions lost alignment")
    action_spec = ActionSpec()
    predicted_hits: Counter[int] = Counter()
    neutral_hits: Counter[int] = Counter()
    predicted_shot_hits: Counter[int] = Counter()
    neutral_shot_hits: Counter[int] = Counter()
    predicted_kind_counts: Counter[int] = Counter()
    predicted_shots = 0
    eligible = 0
    skipped_spawn_boundary = 0
    source_projectile_cases = 0
    invalid_actions = 0
    elapsed_mismatches = 0
    selected_target_hits = 0
    oracle_actionable = 0
    oracle_selected_target_hits = 0
    event_kinds = (
        int(EventKind.PROJECTILE_HIT),
        int(EventKind.CHAIN_JOINED),
        int(EventKind.CLEARED),
    )
    for index, (record, prediction) in enumerate(zip(heldout, predictions)):
        kind, wait_index, target_index, template_index = prediction
        predicted_kind_counts[kind] += 1
        safe_horizon = min(horizon, ticks_before_next_spawn(record.observation))
        if safe_horizon < (2 if kind else spec.wait_choices[wait_index]):
            skipped_spawn_boundary += 1
            continue
        eligible += 1
        if any(
            body.get("kind") == "projectile"
            for body in record.observation.get("bodies", ())
        ):
            source_projectile_cases += 1
        semantic = decode_pointer_action(
            kind=kind,
            wait_index=wait_index,
            template_index=template_index,
            selected_body_row=(
                None if kind == 0 else tensors.body_features[index, target_index]
            ),
            schema=TEACHER_V1,
            pointer_spec=spec,
            action_spec=action_spec,
        )
        primitive = [action_spec.press(semantic)]
        if semantic.kind is not SemanticActionKind.WAIT:
            predicted_shots += 1
            primitive.append(action_spec.release())
        config = configs[record.config_index]
        with IrisuEnv(
            library_path=library,
            physics_backend="portable",
            config=config,
        ) as env:
            env.reset(seed=SEED)
            env.restore_state(record.snapshot)
            predicted, _, predicted_elapsed, predicted_events = _execute_actions(
                env,
                int(record.observation["tick"]),
                primitive,
                safe_horizon,
            )
            invalid_actions += predicted[int(EventKind.INVALID_ACTION)]
            env.restore_state(record.snapshot)
            neutral, _, neutral_elapsed, _ = _execute_actions(
                env,
                int(record.observation["tick"]),
                [Action.wait(safe_horizon)],
                safe_horizon,
            )
            if record.decision.target_body_id is not None:
                oracle_actionable += 1
                env.restore_state(record.snapshot)
                _, _, _, oracle_events = _execute_actions(
                    env,
                    int(record.observation["tick"]),
                    list(
                        lower_expert_decision(
                            record.decision, record.observation, spec
                        )
                    ),
                    safe_horizon,
                )
                oracle_selected_target_hits += int(
                    any(
                        _event_kind(event) == int(EventKind.PROJECTILE_HIT)
                        and int(event.get("b", -1))
                        == record.decision.target_body_id
                        for event in oracle_events
                    )
                )
        elapsed_mismatches += int(predicted_elapsed != neutral_elapsed)
        if kind != 0:
            selected_body_id = heldout_body_ids[index][target_index]
            selected_target_hits += int(
                any(
                    _event_kind(event) == int(EventKind.PROJECTILE_HIT)
                    and int(event.get("b", -1)) == selected_body_id
                    for event in predicted_events
                )
            )
        for event_kind in event_kinds:
            predicted_hits[event_kind] += int(predicted[event_kind] > 0)
            neutral_hits[event_kind] += int(neutral[event_kind] > 0)
            if kind != 0:
                predicted_shot_hits[event_kind] += int(predicted[event_kind] > 0)
                neutral_shot_hits[event_kind] += int(neutral[event_kind] > 0)

    def rates(counts: Counter[int], denominator: int) -> dict[str, float]:
        denominator = max(denominator, 1)
        return {
            "projectile_hit_rate": counts[int(EventKind.PROJECTILE_HIT)] / denominator,
            "chain_joined_rate": counts[int(EventKind.CHAIN_JOINED)] / denominator,
            "cleared_rate": counts[int(EventKind.CLEARED)] / denominator,
        }

    predicted_rates = rates(predicted_hits, eligible)
    neutral_rates = rates(neutral_hits, eligible)
    return {
        "eligible_restored_states": eligible,
        "predicted_shots": predicted_shots,
        "selected_target_hit_count": selected_target_hits,
        "selected_target_hit_rate_among_all_states": (
            selected_target_hits / max(eligible, 1)
        ),
        "selected_target_hit_rate_among_predicted_shots": (
            selected_target_hits / max(predicted_shots, 1)
        ),
        "matcher_oracle_actionable_states": oracle_actionable,
        "matcher_oracle_selected_target_hit_rate": (
            oracle_selected_target_hits / max(oracle_actionable, 1)
        ),
        "predicted_kind_counts": {
            "wait": predicted_kind_counts[0],
            "weak": predicted_kind_counts[1],
            "strong": predicted_kind_counts[2],
        },
        "skipped_at_spawn_boundary": skipped_spawn_boundary,
        "source_states_with_existing_projectiles": source_projectile_cases,
        "invalid_actions": invalid_actions,
        "matched_horizon_elapsed_mismatches": elapsed_mismatches,
        "horizon_ticks": horizon,
        "spawn_censored": True,
        "predicted_decision": predicted_rates,
        "matched_neutral_wait": neutral_rates,
        "predicted_shot_subset": rates(predicted_shot_hits, predicted_shots),
        "matched_neutral_on_shot_subset": rates(
            neutral_shot_hits, predicted_shots
        ),
        "absolute_rate_delta": {
            key: predicted_rates[key] - neutral_rates[key]
            for key in predicted_rates
        },
        "metric_limits": [
            "Development-only portable-clone evidence; it is not canonical R3 evidence.",
            "Rates are per restored state and count whether an event occurred, not event multiplicity.",
            "The matched neutral branch controls deterministic state and horizon, but chain/clear events may reflect downstream physics rather than a direct hit.",
            "A zero event rate can mean the fixed spawn-censored horizon was too short, not necessarily that aiming failed.",
        ],
        "projectile_event_attribution_reliable": (
            eligible > 0
            and source_projectile_cases == 0
            and elapsed_mismatches == 0
            and invalid_actions == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--observations-per-episode", type=int, default=6)
    parser.add_argument("--search-audits", type=int, default=12)
    parser.add_argument("--training-steps", type=int, default=600)
    parser.add_argument("--causal-horizon", type=int, default=60)
    args = parser.parse_args()
    if (
        args.episodes < 6
        or args.observations_per_episode < 2
        or args.search_audits < 0
        or args.training_steps < 1
        or args.causal_horizon < 2
    ):
        parser.error("benchmark budgets are invalid")

    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.library is not None:
        library = Path(find_library(args.library)).resolve()
        library_selection = "explicit"
    elif TRUSTED_PORTABLE_LIBRARY.is_file():
        library = TRUSTED_PORTABLE_LIBRARY.resolve()
        library_selection = "trusted_r3_portable_artifact"
    else:
        library = Path(find_library()).resolve()
        library_selection = "development_build_fallback"
    spec = PointerActionSpec()
    controlled_records, controlled_configs, controlled_hashes, controlled_search = (
        _collect(
            library,
            episodes=args.episodes,
            observations_per_episode=args.observations_per_episode,
            search_audits=args.search_audits // 2,
            spec=spec,
            challenge_name="controlled-one-pair",
            actionable_falling_count=2,
            require_unambiguous_pair=True,
        )
    )
    crowded_records, crowded_configs, crowded_hashes, crowded_search = _collect(
        library,
        episodes=args.episodes,
        observations_per_episode=args.observations_per_episode,
        search_audits=args.search_audits - args.search_audits // 2,
        spec=spec,
        challenge_name="crowded-multi-pair",
        actionable_falling_count=5,
        require_unambiguous_pair=False,
    )
    controlled_dataset, controlled_body_ids = _build_dataset(
        controlled_records, spec
    )
    crowded_dataset, crowded_body_ids = _build_dataset(crowded_records, spec)
    controlled_train, controlled_validation = _stratified_episode_split(
        controlled_dataset, controlled_records
    )
    crowded_train, crowded_validation = _stratified_episode_split(
        crowded_dataset, crowded_records
    )
    training = PointerDataset([*controlled_train, *crowded_train])
    model_config = PointerModelConfig(
        global_hidden=48,
        body_hidden=48,
        attention_hidden=96,
        attention_heads=4,
        attention_layers=2,
        feedforward_hidden=192,
        recurrent_hidden=96,
        matcher_prior_scale=16.0,
    )
    model = EntityPointerActorCritic(
        TEACHER_V1, pointer_spec=spec, config=model_config
    )
    trainer = PointerBCTrainer(
        model,
        config=PointerBCConfig(
            learning_rate=1e-3,
            target_coefficient=1e-4,
            value_coefficient=0.01,
            max_gradient_norm=5.0,
        ),
    )
    initial_train = trainer.evaluate(training)
    trainer.fit(training, args.training_steps)
    train_metrics = trainer.evaluate(training)
    controlled_train_metrics = trainer.evaluate(controlled_train)
    controlled_heldout_metrics = trainer.evaluate(controlled_validation)
    crowded_train_metrics = trainer.evaluate(crowded_train)
    crowded_heldout_metrics = trainer.evaluate(crowded_validation)
    controlled_causal = _causal_evaluation(
        library,
        controlled_records,
        controlled_configs,
        controlled_body_ids,
        controlled_validation,
        _predict(model, controlled_validation),
        horizon=args.causal_horizon,
        spec=spec,
    )
    crowded_causal = _causal_evaluation(
        library,
        crowded_records,
        crowded_configs,
        crowded_body_ids,
        crowded_validation,
        _predict(model, crowded_validation),
        horizon=args.causal_horizon,
        spec=spec,
    )
    gates = {
        "default_has_at_least_48_episodes_per_challenge": args.episodes >= 48,
        "train_has_every_class": min(_class_counts(training).values()) > 0,
        "controlled_heldout_has_every_class": (
            min(_class_counts(controlled_validation).values()) > 0
        ),
        "crowded_heldout_has_every_class": (
            min(_class_counts(crowded_validation).values()) > 0
        ),
        "episode_splits_are_disjoint": (
            set(controlled_train.episode_identities).isdisjoint(
                controlled_validation.episode_identities
            )
            and set(crowded_train.episode_identities).isdisjoint(
                crowded_validation.episode_identities
            )
        ),
        "model_input_id_scaled_is_zero": True,
        "controlled_heldout_kind_accuracy_at_least_0_90": (
            controlled_heldout_metrics.kind_accuracy >= 0.90
        ),
        "crowded_heldout_kind_accuracy_at_least_0_90": (
            crowded_heldout_metrics.kind_accuracy >= 0.90
        ),
        "controlled_heldout_target_accuracy_at_least_0_90": (
            controlled_heldout_metrics.target_accuracy >= 0.90
        ),
        "crowded_heldout_target_accuracy_at_least_0_90": (
            crowded_heldout_metrics.target_accuracy >= 0.90
        ),
        "both_heldout_template_accuracies_at_least_0_90": (
            controlled_heldout_metrics.template_accuracy >= 0.90
            and crowded_heldout_metrics.template_accuracy >= 0.90
        ),
        "both_heldout_actionable_recalls_at_least_0_90": (
            controlled_heldout_metrics.actionable_recall >= 0.90
            and crowded_heldout_metrics.actionable_recall >= 0.90
        ),
        "both_selected_target_hit_rates_at_least_0_80": (
            controlled_causal["selected_target_hit_rate_among_predicted_shots"]
            >= 0.80
            and crowded_causal["selected_target_hit_rate_among_predicted_shots"]
            >= 0.80
        ),
        "both_projectile_event_attributions_reliable": (
            controlled_causal["projectile_event_attribution_reliable"]
            and crowded_causal["projectile_event_attribution_reliable"]
        ),
    }
    gates["all_passed"] = all(gates.values())
    elapsed = time.perf_counter() - started
    prior_path = Path(str(PRIOR_FAILED_RUN["path"]))
    prior_hash_matches = (
        prior_path.is_file()
        and hashlib.sha256(prior_path.read_bytes()).hexdigest()
        == PRIOR_FAILED_RUN["json_sha256"]
    )
    prior_balanced_path = Path(str(PRIOR_FAILED_BALANCED_RUN["path"]))
    prior_balanced_hash_matches = (
        prior_balanced_path.is_file()
        and hashlib.sha256(prior_balanced_path.read_bytes()).hexdigest()
        == PRIOR_FAILED_BALANCED_RUN["json_sha256"]
    )
    prior_relation_path = Path(str(PRIOR_FAILED_RELATION_RUN["path"]))
    prior_relation_hash_matches = (
        prior_relation_path.is_file()
        and hashlib.sha256(prior_relation_path.read_bytes()).hexdigest()
        == PRIOR_FAILED_RELATION_RUN["json_sha256"]
    )
    prior_dual_path = Path(str(PRIOR_FAILED_DUAL_PRIOR_RUN["path"]))
    prior_dual_hash_matches = (
        prior_dual_path.is_file()
        and hashlib.sha256(prior_dual_path.read_bytes()).hexdigest()
        == PRIOR_FAILED_DUAL_PRIOR_RUN["json_sha256"]
    )
    result = {
        "schema": "rl-r3c-pointer-development-benchmark-v2",
        "development_only": True,
        "canonical_r3_artifacts_touched": False,
        "prior_failed_default_run": {
            **PRIOR_FAILED_RUN,
            "artifact_still_present_and_hash_matches": prior_hash_matches,
        },
        "prior_failed_balanced_run": {
            **PRIOR_FAILED_BALANCED_RUN,
            "artifact_still_present_and_hash_matches": (
                prior_balanced_hash_matches
            ),
        },
        "prior_failed_relation_run": {
            **PRIOR_FAILED_RELATION_RUN,
            "artifact_still_present_and_hash_matches": (
                prior_relation_hash_matches
            ),
        },
        "prior_failed_dual_prior_run": {
            **PRIOR_FAILED_DUAL_PRIOR_RUN,
            "artifact_still_present_and_hash_matches": prior_dual_hash_matches,
        },
        "seed": SEED,
        "runtime": {
            "backend": "portable",
            "library": str(library),
            "library_selection": library_selection,
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "config_hashes": sorted(set(controlled_hashes + crowded_hashes)),
        },
        "identities": {
            "teacher_schema_sha256": TEACHER_V1.sha256,
            "pointer_action_sha256": spec.sha256,
        },
        "budgets": {
            "episodes_per_challenge": args.episodes,
            "total_episodes": args.episodes * 2,
            "observations_per_episode": args.observations_per_episode,
            "collected_observations": (
                len(controlled_records) + len(crowded_records)
            ),
            "search_teacher_audits_requested": args.search_audits,
            "training_steps": args.training_steps,
            "causal_horizon_ticks": args.causal_horizon,
        },
        "model_input_ablation": {
            "id_scaled": "zeroed_after external ID-to-row label binding",
            "labels_retain_body_ids": False,
            "snapshots_and_public_observations_retain_ids_for_execution_audit": True,
        },
        "model": model.manifest(),
        "optimizer": asdict(trainer.config),
        "metrics": {
            "combined_initial_train": _metric_summary(initial_train),
            "combined_train": _metric_summary(train_metrics),
        },
        "datasets": {
            "controlled_one_pair": {
                "configs": list(controlled_configs),
                "label_source": "matcher_anchor",
                "search_teacher_audit": controlled_search,
                "episode_identity_split": (
                    "deterministic_stratified_by_label_class"
                ),
                "train_episodes": len(set(controlled_train.episode_identities)),
                "heldout_episodes": len(
                    set(controlled_validation.episode_identities)
                ),
                "train_class_counts": _class_counts(controlled_train),
                "heldout_class_counts": _class_counts(controlled_validation),
                "metrics": {
                    "train": _metric_summary(controlled_train_metrics),
                    "heldout": _metric_summary(controlled_heldout_metrics),
                },
                "restored_state_causal_evaluation": controlled_causal,
            },
            "crowded_multi_pair": {
                "configs": list(crowded_configs),
                "label_source": "matcher_anchor",
                "search_teacher_audit": crowded_search,
                "episode_identity_split": (
                    "deterministic_stratified_by_label_class"
                ),
                "train_episodes": len(set(crowded_train.episode_identities)),
                "heldout_episodes": len(
                    set(crowded_validation.episode_identities)
                ),
                "train_class_counts": _class_counts(crowded_train),
                "heldout_class_counts": _class_counts(crowded_validation),
                "metrics": {
                    "train": _metric_summary(crowded_train_metrics),
                    "heldout": _metric_summary(crowded_heldout_metrics),
                },
                "restored_state_causal_evaluation": crowded_causal,
            },
        },
        "gates": gates,
        "scope_limits": [
            "Controlled and crowded challenges use separate episode identities "
            "but one jointly trained model.",
            "The crowded gate retains multiple competing same-color pairs and "
            "requires held-out target accuracy >=0.90.",
            "Earlier crowded failures remain preserved as negative evidence.",
            "Search-teacher decisions are audited but excluded from BC labels "
            "because its utility objective need not agree with matcher_anchor.",
        ],
        "elapsed_seconds": elapsed,
        "reproduction": (
            "uv run --extra training python benchmarks/rl_r3c_pointer.py"
        ),
        "interpretation": (
            "A fast portable-runtime development diagnostic for pointer "
            "representation, imitation, and restored-state action effects. "
            "It is not sealed, canonical, or evidence of original-game transfer."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
