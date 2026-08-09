"""Deterministic, identity-bound archive for public strategic cells."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .strategic import StrategicFeatures, extract_strategic_features


@dataclass(frozen=True, slots=True)
class ArchiveCellConfig:
    """Discretization that is independent of transient body and chain IDs."""

    level_band_width: int = 5
    score_boundaries: tuple[int, ...] = (
        0,
        16,
        64,
        256,
        1_000,
        4_000,
        16_000,
        64_000,
        100_000,
        250_000,
    )
    gauge_bins: int = 8
    spawn_phase_bins: int = 4
    group_size_cap: int = 6
    hazard_cap: int = 4
    hit_budget_cap: int = 8

    def __post_init__(self) -> None:
        for name, value in (
            ("level_band_width", self.level_band_width),
            ("gauge_bins", self.gauge_bins),
            ("spawn_phase_bins", self.spawn_phase_bins),
            ("group_size_cap", self.group_size_cap),
            ("hazard_cap", self.hazard_cap),
            ("hit_budget_cap", self.hit_budget_cap),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not self.score_boundaries
            or self.score_boundaries[0] != 0
            or any(
                type(value) is not int or value < 0
                for value in self.score_boundaries
            )
            or tuple(sorted(set(self.score_boundaries))) != self.score_boundaries
        ):
            raise ValueError(
                "score_boundaries must start at zero and increase uniquely"
            )


@dataclass(frozen=True, order=True, slots=True)
class ArchiveCellKey:
    """A color-permutation-invariant cell key derived from public features."""

    level_band: int
    score_bin: int
    gauge_bin: int
    active_colors: int
    highest_chain: int
    color_group_profiles: tuple[tuple[int, ...], ...]
    viable_anchors: int
    safe_hit_budget: int
    floor_hazards: int
    rot_hazards: int
    cadence_interval: int
    spawn_phase_bin: int

    def manifest(self) -> dict[str, object]:
        return {
            "level_band": self.level_band,
            "score_bin": self.score_bin,
            "gauge_bin": self.gauge_bin,
            "active_colors": self.active_colors,
            "highest_chain": self.highest_chain,
            "color_group_profiles": [
                list(profile) for profile in self.color_group_profiles
            ],
            "viable_anchors": self.viable_anchors,
            "safe_hit_budget": self.safe_hit_budget,
            "floor_hazards": self.floor_hazards,
            "rot_hazards": self.rot_hazards,
            "cadence_interval": self.cadence_interval,
            "spawn_phase_bin": self.spawn_phase_bin,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def archive_cell_key(
    features: StrategicFeatures,
    config: ArchiveCellConfig | None = None,
) -> ArchiveCellKey:
    """Discretize public strategic features without consulting future RNG."""

    resolved = ArchiveCellConfig() if config is None else config
    profiles = tuple(
        sorted(
            (
                tuple(
                    sorted(
                        (
                            min(group.members, resolved.group_size_cap)
                            for group in color.groups
                        ),
                        reverse=True,
                    )
                )
                + (
                    min(color.ungrouped_fresh, resolved.group_size_cap),
                    min(color.ungrouped_rotten, resolved.hazard_cap),
                )
            )
            for color in features.colors
        )
    )
    gauge_bin = min(
        resolved.gauge_bins - 1,
        int(features.gauge_fraction * resolved.gauge_bins),
    )
    phase_bin = min(
        resolved.spawn_phase_bins - 1,
        int(features.spawn_phase * resolved.spawn_phase_bins),
    )
    return ArchiveCellKey(
        level_band=features.level // resolved.level_band_width,
        score_bin=bisect.bisect_right(
            resolved.score_boundaries, features.raw_score
        )
        - 1,
        gauge_bin=gauge_bin,
        active_colors=features.active_colors,
        highest_chain=min(features.highest_chain, resolved.group_size_cap),
        color_group_profiles=profiles,
        viable_anchors=min(features.viable_anchor_count, resolved.hazard_cap),
        safe_hit_budget=min(
            features.total_safe_direct_hit_budget, resolved.hit_budget_cap
        ),
        floor_hazards=min(features.floor_hazard_count, resolved.hazard_cap),
        rot_hazards=min(features.rotten_piece_count, resolved.hazard_cap),
        cadence_interval=features.spawn_interval_ticks,
        spawn_phase_bin=phase_bin,
    )


@dataclass(frozen=True, slots=True)
class ArchiveElite:
    """Opaque snapshot plus public evidence used for deterministic replacement."""

    cell: ArchiveCellKey
    snapshot: bytes
    snapshot_sha256: str
    source_identity: str
    trajectory_identity: str
    raw_score: int
    survival_ticks: int
    alive: bool
    qualifying_clears: int
    highest_chain: int
    gauge: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, bytes) or not self.snapshot:
            raise ValueError("archive snapshot must be nonempty bytes")
        expected = hashlib.sha256(self.snapshot).hexdigest()
        if self.snapshot_sha256 != expected:
            raise ValueError("archive snapshot identity does not match its bytes")
        for name, value in (
            ("source_identity", self.source_identity),
            ("trajectory_identity", self.trajectory_identity),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for name, value in (
            ("raw_score", self.raw_score),
            ("survival_ticks", self.survival_ticks),
            ("qualifying_clears", self.qualifying_clears),
            ("highest_chain", self.highest_chain),
            ("gauge", self.gauge),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.alive) is not bool:
            raise TypeError("alive must be a bool")

    @classmethod
    def create(
        cls,
        observation: Mapping[str, Any],
        snapshot: bytes,
        *,
        source_identity: str,
        trajectory_identity: str,
        cell_config: ArchiveCellConfig | None = None,
    ) -> ArchiveElite:
        features = extract_strategic_features(observation)
        return cls(
            cell=archive_cell_key(features, cell_config),
            snapshot=snapshot,
            snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
            source_identity=source_identity,
            trajectory_identity=trajectory_identity,
            raw_score=features.raw_score,
            survival_ticks=features.tick,
            alive=features.alive,
            qualifying_clears=features.qualifying_clears,
            highest_chain=features.highest_chain,
            gauge=features.gauge,
        )

    @property
    def objective(self) -> tuple[int, int, int]:
        """Scientific objective: raw score, survival, then nonterminal state."""

        return self.raw_score, self.survival_ticks, int(self.alive)

    def better_than(self, other: ArchiveElite) -> bool:
        """Compare objectives, then use evidence identities only as tie-breaks."""

        if self.objective != other.objective:
            return self.objective > other.objective
        return (
            self.snapshot_sha256,
            self.trajectory_identity,
            self.source_identity,
        ) < (
            other.snapshot_sha256,
            other.trajectory_identity,
            other.source_identity,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "cell": self.cell.manifest(),
            "snapshot_sha256": self.snapshot_sha256,
            "source_identity": self.source_identity,
            "trajectory_identity": self.trajectory_identity,
            "raw_score": self.raw_score,
            "survival_ticks": self.survival_ticks,
            "alive": self.alive,
            "qualifying_clears": self.qualifying_clears,
            "highest_chain": self.highest_chain,
            "gauge": self.gauge,
        }


class StrategicArchive:
    """One deterministic raw-score/survival elite per strategic cell."""

    def __init__(
        self,
        *,
        source_identity: str,
        cell_config: ArchiveCellConfig | None = None,
    ) -> None:
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError("source_identity must be a nonempty string")
        self.source_identity = source_identity
        self.cell_config = ArchiveCellConfig() if cell_config is None else cell_config
        self._elites: dict[ArchiveCellKey, ArchiveElite] = {}

    def __len__(self) -> int:
        return len(self._elites)

    def __iter__(self) -> Iterator[ArchiveElite]:
        for cell in sorted(self._elites):
            yield self._elites[cell]

    def get(self, cell: ArchiveCellKey) -> ArchiveElite | None:
        return self._elites.get(cell)

    def consider(self, elite: ArchiveElite) -> bool:
        """Insert or replace an elite; reject mixed-source evidence."""

        if elite.source_identity != self.source_identity:
            raise ValueError("archive elite source identity mismatch")
        existing = self._elites.get(elite.cell)
        if existing is not None and not elite.better_than(existing):
            return False
        self._elites[elite.cell] = elite
        return True

    def capture(
        self,
        observation: Mapping[str, Any],
        snapshot: bytes,
        *,
        trajectory_identity: str,
    ) -> tuple[ArchiveElite, bool]:
        elite = ArchiveElite.create(
            observation,
            snapshot,
            source_identity=self.source_identity,
            trajectory_identity=trajectory_identity,
            cell_config=self.cell_config,
        )
        return elite, self.consider(elite)

    def manifest(self) -> dict[str, object]:
        return {
            "source_identity": self.source_identity,
            "cell_config": {
                "level_band_width": self.cell_config.level_band_width,
                "score_boundaries": list(self.cell_config.score_boundaries),
                "gauge_bins": self.cell_config.gauge_bins,
                "spawn_phase_bins": self.cell_config.spawn_phase_bins,
                "group_size_cap": self.cell_config.group_size_cap,
                "hazard_cap": self.cell_config.hazard_cap,
                "hit_budget_cap": self.cell_config.hit_budget_cap,
            },
            "elites": [elite.manifest() for elite in self],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ArchiveCellConfig",
    "ArchiveCellKey",
    "ArchiveElite",
    "StrategicArchive",
    "archive_cell_key",
]
