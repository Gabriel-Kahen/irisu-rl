#!/usr/bin/env python3
"""Compare original-game and clone score-event timelines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCORE_EVENTS = {"score", "score_changed"}
CLEAR_EVENTS = {"qualifying_clear", "confirmed"}


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class ScoreCall:
    ordinal: int
    tick: int
    delta: int
    cumulative_before: int
    cumulative_after: int


@dataclass(frozen=True)
class ClearInput:
    ordinal: int
    tick: int
    kind: str
    chain: int | None
    group_num: int | None
    actors: dict[str, int]


@dataclass(frozen=True)
class Episode:
    tick: int
    calls: tuple[ScoreCall, ...]
    clears: tuple[ClearInput, ...]
    cumulative_before: int
    cumulative_after: int


@dataclass(frozen=True)
class Trace:
    label: str
    initial_score: int
    observed_through_tick: int
    calls: tuple[ScoreCall, ...]
    episodes: tuple[Episode, ...]
    clears: tuple[ClearInput, ...]


@dataclass(frozen=True)
class MismatchSegment:
    start_tick: int
    rejoin_tick: int | None
    score_before: tuple[int, int]


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComparisonError(f"{where} must be an integer")
    return value


def _event_name(record: dict[str, Any]) -> str:
    value = record.get("event", record.get("kind", record.get("kind_name", "")))
    return value if isinstance(value, str) else ""


def read_document(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError:
        rows: list[Any] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ComparisonError(
                        f"{path}:{line_number}: invalid JSON: {error}"
                    ) from error
        return rows


def _document_events(document: Any) -> tuple[list[dict[str, Any]], int]:
    horizon = 0
    if isinstance(document, list):
        events = document
    elif isinstance(document, dict):
        events = None
        for key in ("causal_events", "events"):
            candidate = document.get(key)
            if isinstance(candidate, list):
                events = candidate
                break
        if events is None and isinstance(document.get("score_curve"), list):
            events = list(document["score_curve"])
            episodes = document.get("score_episodes", {})
            if isinstance(episodes, dict):
                for value in episodes.values():
                    if isinstance(value, dict) and isinstance(
                        value.get("confirmed"), list
                    ):
                        events.extend(value["confirmed"])
        if events is None and _event_name(document):
            events = [document]
        if events is None:
            raise ComparisonError("JSON object contains no full event timeline")
        for parent, child in (
            ("final", "tick"),
            ("terminal", "tick"),
        ):
            value = document.get(parent)
            if isinstance(value, dict) and isinstance(value.get(child), int):
                horizon = max(horizon, value[child])
        outcome = document.get("outcome")
        if isinstance(outcome, dict):
            clone = outcome.get("clone")
            if isinstance(clone, dict) and isinstance(clone.get("tick"), int):
                horizon = max(horizon, clone["tick"])
    else:
        raise ComparisonError("input must be a JSON object, array, or JSONL stream")

    if not all(isinstance(event, dict) for event in events):
        raise ComparisonError("event timeline must contain JSON objects")
    for index, event in enumerate(events):
        if "tick" in event:
            horizon = max(horizon, _integer(event["tick"], f"event {index}.tick"))
    return events, horizon


def _event_order(record: dict[str, Any], fallback: int) -> tuple[int, int]:
    sequence = record.get("sequence")
    return (sequence if isinstance(sequence, int) else fallback, fallback)


def _score_delta(record: dict[str, Any], index: int) -> int:
    value = record.get("delta", record.get("value"))
    return _integer(value, f"score event {index}.delta")


def _initial_score(score_rows: list[tuple[int, dict[str, Any]]]) -> int:
    if not score_rows:
        return 0
    first_tick = _integer(score_rows[0][1].get("tick"), "first score tick")
    same_tick = [
        (index, row) for index, row in score_rows if row["tick"] == first_tick
    ]
    first_index, first = same_tick[0]
    delta = _score_delta(first, first_index)
    for key in ("cumulative_after", "score"):
        if key in first:
            return _integer(first[key], f"score event {first_index}.{key}") - delta
    finals = {
        _integer(row["score_after_tick"], f"score event {index}.score_after_tick")
        for index, row in same_tick
        if "score_after_tick" in row
    }
    if len(finals) > 1:
        raise ComparisonError("first score episode has conflicting final scores")
    if finals:
        return next(iter(finals)) - sum(
            _score_delta(row, index) for index, row in same_tick
        )
    return 0


def normalize_trace(document: Any, label: str) -> Trace:
    records, horizon = _document_events(document)
    indexed = list(enumerate(records))
    score_rows = [
        (index, row) for index, row in indexed if _event_name(row) in SCORE_EVENTS
    ]
    score_rows.sort(
        key=lambda item: (
            _integer(item[1].get("tick"), f"score event {item[0]}.tick"),
            _event_order(item[1], item[0]),
        )
    )

    clear_rows = [
        (index, row) for index, row in indexed if _event_name(row) in CLEAR_EVENTS
    ]
    clear_rows.sort(
        key=lambda item: (
            _integer(item[1].get("tick"), f"clear event {item[0]}.tick"),
            _event_order(item[1], item[0]),
        )
    )
    clears: list[ClearInput] = []
    next_clear = 1
    for index, row in clear_rows:
        tick = _integer(row.get("tick"), f"clear event {index}.tick")
        explicit = row.get("clears", row.get("clear_ordinal"))
        ordinal = (
            _integer(explicit, f"clear event {index}.ordinal")
            if explicit is not None
            else next_clear
        )
        next_clear = max(next_clear + 1, ordinal + 1)
        chain = row.get("chain")
        group_num = row.get("group_num")
        if group_num is None and _event_name(row) == "confirmed":
            group_num = row.get("value")
        chain = (
            None
            if chain is None
            else _integer(chain, f"clear event {index}.chain")
        )
        group_num = (
            None
            if group_num is None
            else _integer(group_num, f"clear event {index}.group_num")
        )
        actors: dict[str, int] = {}
        actor_fields = (
            ("target", "target"),
            ("source", "source"),
            ("a", "target"),
            ("b", "source"),
        )
        for source, target in actor_fields:
            if source in row and target not in actors:
                actors[target] = _integer(row[source], f"clear event {index}.{source}")
        clears.append(
            ClearInput(ordinal, tick, _event_name(row), chain, group_num, actors)
        )

    clears_by_tick: dict[int, list[ClearInput]] = {}
    for clear in clears:
        clears_by_tick.setdefault(clear.tick, []).append(clear)

    initial = _initial_score(score_rows)
    running = initial
    calls: list[ScoreCall] = []
    episodes: list[Episode] = []
    cursor = 0
    while cursor < len(score_rows):
        tick = _integer(score_rows[cursor][1].get("tick"), "score tick")
        end = cursor + 1
        while end < len(score_rows) and score_rows[end][1]["tick"] == tick:
            end += 1
        before = running
        episode_calls: list[ScoreCall] = []
        final_candidates: set[int] = set()
        for index, row in score_rows[cursor:end]:
            delta = _score_delta(row, index)
            call_before = running
            running += delta
            direct = row.get("cumulative_after")
            if direct is None and _event_name(row) == "score":
                direct = row.get("score")
            if direct is not None and _integer(
                direct, f"score event {index}.cumulative"
            ) != running:
                raise ComparisonError(
                    f"score event {index} has inconsistent cumulative score"
                )
            if "score_after_tick" in row:
                final_candidates.add(
                    _integer(
                        row["score_after_tick"],
                        f"score event {index}.score_after_tick",
                    )
                )
            call = ScoreCall(len(calls) + 1, tick, delta, call_before, running)
            calls.append(call)
            episode_calls.append(call)
        if len(final_candidates) > 1 or (
            final_candidates and next(iter(final_candidates)) != running
        ):
            raise ComparisonError(
                f"score episode at tick {tick} has inconsistent final score"
            )
        episodes.append(
            Episode(
                tick,
                tuple(episode_calls),
                tuple(clears_by_tick.get(tick, ())),
                before,
                running,
            )
        )
        cursor = end

    return Trace(label, initial, horizon, tuple(calls), tuple(episodes), tuple(clears))


def _clear_summary(clear: ClearInput) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ordinal": clear.ordinal,
        "kind": clear.kind,
    }
    if clear.chain is not None:
        result["chain"] = clear.chain
    if clear.group_num is not None:
        result["group_num"] = clear.group_num
    if clear.actors:
        result["actors"] = clear.actors
    return result


def _call_summary(call: ScoreCall | None) -> dict[str, Any] | None:
    if call is None:
        return None
    return {
        "ordinal": call.ordinal,
        "tick": call.tick,
        "delta": call.delta,
        "cumulative_after": call.cumulative_after,
    }


def _episode_summary(episode: Episode | None) -> dict[str, Any] | None:
    if episode is None:
        return None
    return {
        "tick": episode.tick,
        "deltas": [call.delta for call in episode.calls],
        "cumulative_before": episode.cumulative_before,
        "cumulative_after": episode.cumulative_after,
        "score_call_ordinals": [episode.calls[0].ordinal, episode.calls[-1].ordinal],
        "clear_inputs": [_clear_summary(clear) for clear in episode.clears],
    }


def _score_at(trace: Trace, tick: int) -> int:
    score = trace.initial_score
    for episode in trace.episodes:
        if episode.tick > tick:
            break
        score = episode.cumulative_after
    return score


def _segments(original: Trace, clone: Trace, horizon: int) -> list[MismatchSegment]:
    ticks = sorted(
        {episode.tick for episode in original.episodes if episode.tick <= horizon}
        | {episode.tick for episode in clone.episodes if episode.tick <= horizon}
    )
    original_by_tick = {episode.tick: episode for episode in original.episodes}
    clone_by_tick = {episode.tick: episode for episode in clone.episodes}
    left = original.initial_score
    right = clone.initial_score
    active: MismatchSegment | None = None
    result: list[MismatchSegment] = []
    if left != right:
        active = MismatchSegment(0, None, (left, right))
    for tick in ticks:
        before = (left, right)
        if tick in original_by_tick:
            left = original_by_tick[tick].cumulative_after
        if tick in clone_by_tick:
            right = clone_by_tick[tick].cumulative_after
        if left != right and active is None:
            active = MismatchSegment(tick, None, before)
        elif left == right and active is not None:
            result.append(
                MismatchSegment(active.start_tick, tick, active.score_before)
            )
            active = None
    if active is not None:
        result.append(active)
    return result


def _episodes_in(trace: Trace, start: int, end: int | None) -> list[Episode]:
    return [
        episode
        for episode in trace.episodes
        if episode.tick >= start and (end is None or episode.tick <= end)
    ]


def _timing_signature(episodes: list[Episode]) -> list[tuple[int, int]]:
    return [
        (call.delta, call.cumulative_after)
        for episode in episodes
        for call in episode.calls
    ]


def _first_episode_at_or_after(
    trace: Trace, tick: int, through_tick: int
) -> Episode | None:
    return next(
        (
            episode
            for episode in trace.episodes
            if tick <= episode.tick <= through_tick
        ),
        None,
    )


def compare_traces(original: Trace, clone: Trace) -> dict[str, Any]:
    horizon = min(original.observed_through_tick, clone.observed_through_tick)
    prefix = 0
    strict_prefix = 0
    for left, right in zip(original.calls, clone.calls):
        if (left.delta, left.cumulative_after) != (
            right.delta,
            right.cumulative_after,
        ):
            break
        prefix += 1
    for left, right in zip(original.calls, clone.calls):
        if (left.tick, left.delta, left.cumulative_after) != (
            right.tick,
            right.delta,
            right.cumulative_after,
        ):
            break
        strict_prefix += 1

    def complete_episode_count(trace: Trace, call_count: int) -> int:
        return sum(
            episode.calls[-1].ordinal <= call_count for episode in trace.episodes
        )

    next_left = original.calls[prefix] if prefix < len(original.calls) else None
    next_right = clone.calls[prefix] if prefix < len(clone.calls) else None
    prefix_score = (
        original.calls[prefix - 1].cumulative_after
        if prefix
        else original.initial_score
    )
    exact_prefix = {
        "basis": "score-call delta and cumulative total; timing is reported separately",
        "score_calls": prefix,
        "score_episodes": min(
            complete_episode_count(original, prefix),
            complete_episode_count(clone, prefix),
        ),
        "cumulative_score": prefix_score,
        "score_calls_including_tick": strict_prefix,
        "last_matching_original_call": _call_summary(
            original.calls[prefix - 1] if prefix else None
        ),
        "last_matching_clone_call": _call_summary(
            clone.calls[prefix - 1] if prefix else None
        ),
    }

    call_mismatch: dict[str, Any] | None = None
    if next_left is not None or next_right is not None:
        differing = []
        if next_left is None or next_right is None:
            differing.append("missing_call")
        else:
            if next_left.delta != next_right.delta:
                differing.append("delta")
            if next_left.cumulative_after != next_right.cumulative_after:
                differing.append("cumulative_after")
        call_mismatch = {
            "ordinal": prefix + 1,
            "differing_fields": differing,
            "original": _call_summary(next_left),
            "clone": _call_summary(next_right),
        }

    segments = _segments(original, clone, horizon)
    transient: dict[str, Any] | None = None
    for segment in segments:
        if segment.rejoin_tick is None:
            continue
        left_episodes = _episodes_in(original, segment.start_tick, segment.rejoin_tick)
        right_episodes = _episodes_in(clone, segment.start_tick, segment.rejoin_tick)
        if _timing_signature(left_episodes) != _timing_signature(right_episodes):
            continue
        transient = {
            "start_tick": segment.start_tick,
            "rejoined_tick": segment.rejoin_tick,
            "duration_ticks": segment.rejoin_tick - segment.start_tick,
            "score_before": segment.score_before[0],
            "score_after": _score_at(original, segment.rejoin_tick),
            "first_tick_offset_clone_minus_original": (
                right_episodes[0].tick - left_episodes[0].tick
                if left_episodes and right_episodes
                else None
            ),
            "original_episodes": [_episode_summary(value) for value in left_episodes],
            "clone_episodes": [_episode_summary(value) for value in right_episodes],
        }
        break

    lasting: dict[str, Any] | None = None
    open_segment = next(
        (segment for segment in segments if segment.rejoin_tick is None), None
    )
    if open_segment is not None:
        left_at_start = next(
            (
                episode
                for episode in original.episodes
                if episode.tick == open_segment.start_tick
            ),
            None,
        )
        right_at_start = next(
            (
                episode
                for episode in clone.episodes
                if episode.tick == open_segment.start_tick
            ),
            None,
        )
        lasting = {
            "status": "unrejoined_within_observed_overlap",
            "start_tick": open_segment.start_tick,
            "observed_through_tick": horizon,
            "score_before": {
                "original": open_segment.score_before[0],
                "clone": open_segment.score_before[1],
            },
            "score_at_start": {
                "original": _score_at(original, open_segment.start_tick),
                "clone": _score_at(clone, open_segment.start_tick),
            },
            "score_at_overlap_end": {
                "original": _score_at(original, horizon),
                "clone": _score_at(clone, horizon),
            },
            "original_at_start": _episode_summary(left_at_start),
            "clone_at_start": _episode_summary(right_at_start),
            "first_original_episode": _episode_summary(
                _first_episode_at_or_after(
                    original, open_segment.start_tick, horizon
                )
            ),
            "first_clone_episode": _episode_summary(
                _first_episode_at_or_after(clone, open_segment.start_tick, horizon)
            ),
        }

    return {
        "schema_version": 1,
        "inputs": {
            "original": {
                "label": original.label,
                "observed_through_tick": original.observed_through_tick,
                "score_calls": len(original.calls),
                "score_episodes": len(original.episodes),
                "qualifying_clears": len(original.clears),
                "initial_score": original.initial_score,
            },
            "clone": {
                "label": clone.label,
                "observed_through_tick": clone.observed_through_tick,
                "score_calls": len(clone.calls),
                "score_episodes": len(clone.episodes),
                "qualifying_clears": len(clone.clears),
                "initial_score": clone.initial_score,
            },
        },
        "observed_overlap_through_tick": horizon,
        "exact_prefix": exact_prefix,
        "first_transient_timing_mismatch": transient,
        "first_score_call_value_mismatch": call_mismatch,
        "first_lasting_cumulative_mismatch": lasting,
    }


def compare_documents(original: Any, clone: Any) -> dict[str, Any]:
    return compare_traces(
        normalize_trace(original, "original"),
        normalize_trace(clone, "clone"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_events", type=Path)
    parser.add_argument("clone_timeline", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    try:
        report = compare_traces(
            normalize_trace(read_document(args.original_events), str(args.original_events)),
            normalize_trace(read_document(args.clone_timeline), str(args.clone_timeline)),
        )
    except (OSError, ComparisonError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
