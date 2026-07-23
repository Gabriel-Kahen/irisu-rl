"""Atomic, replay-verifiable snapshot bundle generation for R3."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .actions import ActionSpec, SemanticAction, SemanticActionKind
from .curriculum import (
    SnapshotBlobStore,
    SnapshotLibrary,
    SnapshotRecipe,
    replay_snapshot_recipe,
)
from .runtime_identity import attest_simulator_runtime

if TYPE_CHECKING:
    from .r3b_evaluation import CrossBackendEvaluationManifest


GENERATOR_VERSION = "r3b-full-game-generator-v1"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SPLITS = {"train", "validation", "calibration", "test"}
_SPLIT_ORDER = ("train", "calibration", "validation", "test")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or _SAFE_IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a safe nonempty ASCII identifier")
    return value


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_canonical_json(path: Path, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    if payload != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must use canonical JSON encoding")
    return value


@dataclass(frozen=True, slots=True)
class SnapshotIntent:
    """Backend-neutral authority for reaching one snapshot boundary."""

    snapshot_id: str
    stage_id: str
    split: str
    scenario_family: str
    environment_pool: str
    reset_seed: int
    semantic_actions_hex: tuple[str, ...]
    version: str = "r3b-snapshot-intent-v1"

    def __post_init__(self) -> None:
        if self.version != "r3b-snapshot-intent-v1":
            raise ValueError("unknown snapshot intent version")
        for label in (
            "snapshot_id",
            "stage_id",
            "scenario_family",
            "environment_pool",
        ):
            _safe_identifier(getattr(self, label), label)
        if self.split not in _SPLITS:
            raise ValueError("snapshot intent split is invalid")
        if (
            isinstance(self.reset_seed, bool)
            or not isinstance(self.reset_seed, Integral)
            or not 0 <= self.reset_seed < 2**32
        ):
            raise ValueError("snapshot reset seed must fit uint32")
        if not isinstance(self.semantic_actions_hex, tuple):
            raise TypeError("semantic trace must be an immutable tuple")
        action_spec = ActionSpec()
        for payload in self.semantic_actions_hex:
            if not isinstance(payload, str):
                raise TypeError("semantic trace entries must be strings")
            try:
                decoded = bytes.fromhex(payload)
                semantic = action_spec.deserialize(decoded)
            except (TypeError, ValueError) as exc:
                raise ValueError("semantic trace contains an invalid action") from exc
            if action_spec.serialize(semantic).hex() != payload:
                raise ValueError("semantic actions must use canonical lowercase hex")

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> SnapshotIntent:
        expected = {
            "snapshot_id",
            "stage_id",
            "split",
            "scenario_family",
            "environment_pool",
            "reset_seed",
            "semantic_actions_hex",
            "version",
        }
        if set(value) != expected:
            raise ValueError("snapshot intent manifest keys differ")
        trace = value["semantic_actions_hex"]
        if not isinstance(trace, list):
            raise ValueError("snapshot intent trace must be an array")
        arguments = dict(value)
        arguments["semantic_actions_hex"] = tuple(trace)
        try:
            return cls(**arguments)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("snapshot intent manifest types are malformed") from exc

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["semantic_actions_hex"] = list(self.semantic_actions_hex)
        return value


@dataclass(frozen=True, slots=True)
class SnapshotSourceManifest:
    """Reviewed, versioned snapshot inputs containing no observed outcomes."""

    source_id: str
    action_spec_sha256: str
    intents: tuple[SnapshotIntent, ...]
    generator_version: str = GENERATOR_VERSION
    version: str = "r3b-snapshot-source-v1"

    def __post_init__(self) -> None:
        _safe_identifier(self.source_id, "source_id")
        if (
            self.version != "r3b-snapshot-source-v1"
            or self.generator_version != GENERATOR_VERSION
            or self.action_spec_sha256 != ActionSpec().sha256
            or not isinstance(self.intents, tuple)
            or not self.intents
            or any(not isinstance(value, SnapshotIntent) for value in self.intents)
            or len({value.snapshot_id for value in self.intents}) != len(self.intents)
        ):
            raise ValueError("snapshot source manifest is invalid")

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> SnapshotSourceManifest:
        expected = {
            "source_id",
            "action_spec_sha256",
            "intents",
            "generator_version",
            "version",
        }
        if set(value) != expected:
            raise ValueError("snapshot source manifest keys differ")
        intents = value["intents"]
        if not isinstance(intents, list) or any(
            not isinstance(item, Mapping) for item in intents
        ):
            raise ValueError("snapshot source intents must be an array of objects")
        arguments = dict(value)
        arguments["intents"] = tuple(
            SnapshotIntent.from_manifest(item) for item in intents
        )
        try:
            source = cls(**arguments)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("snapshot source manifest types are malformed") from exc
        if source.manifest() != dict(value):
            raise ValueError("snapshot source manifest is not canonical")
        return source

    @classmethod
    def from_json(cls, path: str | Path) -> SnapshotSourceManifest:
        return cls.from_manifest(
            _read_canonical_json(Path(path), "snapshot source manifest")
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_id": self.source_id,
            "generator_version": self.generator_version,
            "action_spec_sha256": self.action_spec_sha256,
            "intents": [value.manifest() for value in self.intents],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class SnapshotPlanSplit:
    count: int
    trace_actions_min: int
    trace_actions_max: int
    scenario_family_namespace: str

    def __post_init__(self) -> None:
        values = (self.count, self.trace_actions_min, self.trace_actions_max)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in values
        ):
            raise ValueError("snapshot plan split counts must be positive integers")
        if self.trace_actions_min > self.trace_actions_max:
            raise ValueError("snapshot trace range is inverted")
        if self.count > 1_000_000 or self.trace_actions_max > 1_000_000:
            raise ValueError("snapshot plan split exceeds safe materialization limits")
        _safe_identifier(self.scenario_family_namespace, "scenario_family_namespace")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SnapshotPlanSplit:
        expected = {
            "count",
            "trace_actions_min",
            "trace_actions_max",
            "scenario_family_namespace",
        }
        if set(value) != expected:
            raise ValueError("snapshot plan split keys differ")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("snapshot plan split types are malformed") from exc

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SnapshotSourcePlan:
    """Deterministically materialize reviewed inputs without observing outcomes."""

    plan_id: str
    master_seed: int
    stage_id: str
    portable_environment_pool: str
    exact_environment_pool: str
    max_wait_ticks: int
    splits: tuple[tuple[str, SnapshotPlanSplit], ...]
    action_spec_sha256: str
    version: str = "r3b-snapshot-source-plan-v1"

    def __post_init__(self) -> None:
        for label in (
            "plan_id",
            "stage_id",
            "portable_environment_pool",
            "exact_environment_pool",
        ):
            _safe_identifier(getattr(self, label), label)
        if (
            self.version != "r3b-snapshot-source-plan-v1"
            or isinstance(self.master_seed, bool)
            or not isinstance(self.master_seed, Integral)
            or not 0 <= self.master_seed < 2**64
            or isinstance(self.max_wait_ticks, bool)
            or not isinstance(self.max_wait_ticks, Integral)
            or self.max_wait_ticks <= 0
            or self.action_spec_sha256 != ActionSpec().sha256
            or not isinstance(self.splits, tuple)
            or tuple(name for name, _value in self.splits) != _SPLIT_ORDER
            or any(
                not isinstance(value, SnapshotPlanSplit) for _name, value in self.splits
            )
            or self.portable_environment_pool == self.exact_environment_pool
        ):
            raise ValueError("snapshot source plan is invalid")
        if not any(wait <= self.max_wait_ticks for wait in ActionSpec().wait_choices):
            raise ValueError("snapshot plan permits no declared wait action")
        if sum(value.count for _name, value in self.splits) > 1_000_000:
            raise ValueError("snapshot plan has too many reset seeds")
        namespaces = tuple(
            value.scenario_family_namespace for _name, value in self.splits
        )
        if len(set(namespaces)) != len(namespaces):
            raise ValueError("snapshot split family namespaces must be disjoint")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SnapshotSourcePlan:
        expected = {
            "version",
            "plan_id",
            "master_seed",
            "stage_id",
            "max_wait_ticks",
            "action_spec_sha256",
            "backends",
            "splits",
        }
        if set(value) != expected:
            raise ValueError("snapshot source plan keys differ")
        backends = value["backends"]
        splits = value["splits"]
        if (
            not isinstance(backends, Mapping)
            or set(backends)
            != {
                "portable_environment_pool",
                "exact_environment_pool",
            }
            or not isinstance(splits, Mapping)
            or set(splits) != set(_SPLIT_ORDER)
            or any(not isinstance(splits[name], Mapping) for name in _SPLIT_ORDER)
        ):
            raise ValueError("snapshot source plan tables differ")
        arguments = {
            key: item
            for key, item in value.items()
            if key not in {"backends", "splits"}
        }
        arguments.update(backends)
        arguments["splits"] = tuple(
            (name, SnapshotPlanSplit.from_mapping(splits[name]))
            for name in _SPLIT_ORDER
        )
        try:
            return cls(**arguments)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("snapshot source plan types are malformed") from exc

    @classmethod
    def from_toml(cls, path: str | Path) -> SnapshotSourcePlan:
        supplied = Path(path)
        if supplied.is_symlink() or not supplied.is_file():
            raise ValueError("snapshot source plan must be a regular TOML file")
        try:
            value = tomllib.loads(supplied.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError("snapshot source plan TOML is malformed") from exc
        return cls.from_mapping(value)

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_id": self.plan_id,
            "master_seed": int(self.master_seed),
            "stage_id": self.stage_id,
            "max_wait_ticks": int(self.max_wait_ticks),
            "action_spec_sha256": self.action_spec_sha256,
            "backends": {
                "portable_environment_pool": self.portable_environment_pool,
                "exact_environment_pool": self.exact_environment_pool,
            },
            "splits": {name: value.manifest() for name, value in self.splits},
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    def materialize(self, backend: str) -> SnapshotSourceManifest:
        if backend not in {"portable", "exact"}:
            raise ValueError("snapshot source backend must be portable or exact")
        environment_pool = (
            self.portable_environment_pool
            if backend == "portable"
            else self.exact_environment_pool
        )
        seed_material = hashlib.sha256(
            f"r3b-reset-seed-v1:{self.master_seed}".encode()
        ).digest()
        multiplier = int.from_bytes(seed_material[:4], "big") | 1
        offset = int.from_bytes(seed_material[4:8], "big")
        waits = tuple(
            value for value in ActionSpec().wait_choices if value <= self.max_wait_ticks
        )
        action_spec = ActionSpec()
        intents: list[SnapshotIntent] = []
        ordinal = 0
        for split, split_plan in self.splits:
            span = split_plan.trace_actions_max - split_plan.trace_actions_min + 1
            for index in range(split_plan.count):
                domain = f"r3b-snapshot-trace-v1:{self.master_seed}:{split}:{index}"
                length_digest = hashlib.sha256(f"{domain}:length".encode()).digest()
                length = split_plan.trace_actions_min + (
                    int.from_bytes(length_digest[:8], "big") % span
                )
                trace: list[str] = []
                for action_index in range(length):
                    digest = hashlib.sha256(
                        f"{domain}:action:{action_index}".encode()
                    ).digest()
                    kind = digest[0] % 3
                    if kind == 0:
                        action = SemanticAction.wait(
                            waits[int.from_bytes(digest[1:5], "big") % len(waits)]
                        )
                    else:
                        denominator = float(2**53 - 1)
                        x = (int.from_bytes(digest[8:16], "big") >> 11) / denominator
                        y = (int.from_bytes(digest[16:24], "big") >> 11) / denominator
                        constructor = (
                            SemanticAction.weak if kind == 1 else SemanticAction.strong
                        )
                        action = constructor(x, y)
                    trace.append(action_spec.serialize(action).hex())
                reset_seed = (multiplier * ordinal + offset) & 0xFFFFFFFF
                intents.append(
                    SnapshotIntent(
                        f"{self.plan_id}-{backend}-{split}-{index:04d}",
                        self.stage_id,
                        split,
                        f"{split_plan.scenario_family_namespace}-{index:04d}",
                        environment_pool,
                        reset_seed,
                        tuple(trace),
                    )
                )
                ordinal += 1
        return SnapshotSourceManifest(
            f"{self.plan_id}-{backend}",
            self.action_spec_sha256,
            tuple(intents),
        )


@dataclass(frozen=True, slots=True)
class SnapshotBundle:
    source: SnapshotSourceManifest
    library: SnapshotLibrary
    store: SnapshotBlobStore
    runtime_backend: str
    runtime_identity_sha256: str
    version: str = "r3b-snapshot-bundle-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-snapshot-bundle-v1"
            or not self.runtime_backend
            or self.store.library.sha256 != self.library.sha256
            or {value.snapshot_id for value in self.source.intents}
            != {value.snapshot_id for value in self.library.recipes}
        ):
            raise ValueError("snapshot bundle identities disagree")
        intents = {value.snapshot_id: value for value in self.source.intents}
        for recipe in self.library.recipes:
            intent = intents[recipe.snapshot_id]
            if (
                recipe.stage_id != intent.stage_id
                or recipe.split != intent.split
                or recipe.scenario_family != intent.scenario_family
                or recipe.environment_pool != intent.environment_pool
                or recipe.reset_seed != intent.reset_seed
                or recipe.semantic_actions_hex != intent.semantic_actions_hex
                or recipe.action_spec_sha256 != self.source.action_spec_sha256
                or recipe.runtime_identity_sha256 != self.runtime_identity_sha256
                or recipe.generator_version != self.source.generator_version
            ):
                raise ValueError("snapshot recipe differs from its source intent")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_sha256": self.source.sha256,
            "library_sha256": self.library.sha256,
            "store_sha256": self.store.sha256,
            "runtime_backend": self.runtime_backend,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "action_spec_sha256": self.source.action_spec_sha256,
            "generator_version": self.source.generator_version,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def _observation_value(observation: object, key: str) -> object:
    if isinstance(observation, Mapping):
        return observation[key]
    return getattr(observation, key)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _generate_recipe(
    simulator: Any,
    intent: SnapshotIntent,
    runtime_sha256: str,
) -> tuple[SnapshotRecipe, bytes]:
    action_spec = ActionSpec()
    reset = simulator.reset(seed=int(intent.reset_seed))
    observation = reset[0] if isinstance(reset, tuple) else reset
    done = False
    for payload in intent.semantic_actions_hex:
        semantic = action_spec.deserialize(bytes.fromhex(payload))
        primitives = [action_spec.press(semantic)]
        if semantic.kind is not SemanticActionKind.WAIT:
            primitives.append(action_spec.release())
        for primitive in primitives:
            result = simulator.step(primitive)
            if not isinstance(result, tuple) or len(result) != 5:
                raise TypeError("snapshot simulator returned a malformed step")
            observation, _reward, terminated, truncated, info = result
            if not isinstance(info, Mapping):
                raise TypeError("snapshot simulator returned malformed step info")
            if bool(info.get("invalid_action", False)):
                raise ValueError("snapshot trace produced an invalid native action")
            done = bool(terminated) or bool(truncated)
            if done:
                break
        if done:
            break
    if (
        done
        or bool(_observation_value(observation, "terminated"))
        or bool(_observation_value(observation, "truncated"))
    ):
        raise ValueError("snapshot trace ends outside a live decision boundary")
    config = simulator.config
    config = config() if callable(config) else config
    if not isinstance(config, Mapping):
        raise TypeError("snapshot simulator config must be a mapping")
    config_sha256 = _sha256(config)
    snapshot = bytes(simulator.clone_state())
    if not snapshot:
        raise ValueError("snapshot simulator returned an empty snapshot")
    recipe = SnapshotRecipe(
        intent.snapshot_id,
        intent.stage_id,
        intent.split,
        intent.scenario_family,
        intent.environment_pool,
        config_sha256,
        _integer(simulator.config_hash(), "simulator config hash"),
        int(intent.reset_seed),
        action_spec.sha256,
        intent.semantic_actions_hex,
        _integer(_observation_value(observation, "tick"), "snapshot tick"),
        _integer(_observation_value(observation, "score"), "snapshot score"),
        _integer(simulator.state_hash(), "simulator state hash"),
        hashlib.sha256(snapshot).hexdigest(),
        runtime_sha256,
        GENERATOR_VERSION,
    )
    return recipe, snapshot


def _write_canonical_json(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_output(path: str | Path) -> tuple[Path, bool]:
    supplied = Path(path).absolute()
    if supplied.name in {"", ".", ".."}:
        raise ValueError("snapshot bundle output path is invalid")
    if any(value.is_symlink() for value in (supplied.parent, *supplied.parents)):
        raise ValueError("snapshot bundle output path must not traverse symlinks")
    parent = supplied.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("snapshot bundle parent must be a real directory")
    output = parent / supplied.name
    existed = output.exists()
    if output.is_symlink():
        raise ValueError("snapshot bundle output must not be a symlink")
    if existed and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("snapshot bundle output must be absent or empty")
    return output, existed


def generate_snapshot_bundle(
    simulator: Any,
    source: SnapshotSourceManifest,
    output_directory: str | Path,
) -> SnapshotBundle:
    """Generate, replay-check, and atomically publish one immutable bundle."""

    if not isinstance(source, SnapshotSourceManifest):
        raise TypeError("source must be a typed snapshot source manifest")
    output, _existed = _prepare_output(output_directory)
    runtime = attest_simulator_runtime(simulator)
    recipes: list[SnapshotRecipe] = []
    blobs: dict[str, bytes] = {}
    for intent in source.intents:
        recipe, blob = _generate_recipe(simulator, intent, runtime.sha256)
        recipes.append(recipe)
        blobs[intent.snapshot_id] = blob
    library = SnapshotLibrary(tuple(recipes))
    store = SnapshotBlobStore(library, blobs)
    bundle = SnapshotBundle(source, library, store, runtime.backend, runtime.sha256)
    for recipe in library.recipes:
        if replay_snapshot_recipe(simulator, recipe) != store[recipe.snapshot_id]:
            raise RuntimeError("snapshot replay returned different bytes")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshots = staging / "snapshots"
        snapshots.mkdir()
        _write_canonical_json(staging / "source.json", source.manifest())
        _write_canonical_json(staging / "library.json", library.manifest())
        _write_canonical_json(staging / "bundle.json", bundle.manifest())
        for recipe in library.recipes:
            path = snapshots / f"{recipe.snapshot_id}.snapshot"
            with path.open("xb") as handle:
                handle.write(store[recipe.snapshot_id])
                handle.flush()
                os.fsync(handle.fileno())
        loaded = load_snapshot_bundle(staging, simulator)
        if loaded.sha256 != bundle.sha256:
            raise RuntimeError("staged snapshot bundle identity changed")
        snapshots_fd = os.open(snapshots, os.O_RDONLY)
        try:
            os.fsync(snapshots_fd)
        finally:
            os.close(snapshots_fd)
        staging_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return bundle


def load_snapshot_bundle(
    directory: str | Path,
    simulator: Any,
) -> SnapshotBundle:
    """Load all bundle artifacts, verify hashes, then replay every recipe."""

    supplied = Path(directory).absolute()
    if any(value.is_symlink() for value in (supplied, *supplied.parents)):
        raise ValueError("snapshot bundle root must not traverse symlinks")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot bundle root must be a directory")
    entries = tuple(root.iterdir())
    if (
        any(value.is_symlink() for value in entries)
        or {value.name for value in entries}
        != {"source.json", "library.json", "bundle.json", "snapshots"}
        or not (root / "snapshots").is_dir()
    ):
        raise ValueError("snapshot bundle layout is unsafe or incomplete")
    source = SnapshotSourceManifest.from_manifest(
        _read_canonical_json(root / "source.json", "snapshot source manifest")
    )
    library = SnapshotLibrary.from_manifest(
        _read_canonical_json(root / "library.json", "snapshot library")
    )
    store = SnapshotBlobStore.from_directory(library, root / "snapshots")
    runtime = attest_simulator_runtime(simulator)
    bundle = SnapshotBundle(source, library, store, runtime.backend, runtime.sha256)
    declared = _read_canonical_json(root / "bundle.json", "snapshot bundle manifest")
    if set(declared) != set(bundle.manifest()) or dict(declared) != bundle.manifest():
        raise ValueError("snapshot bundle manifest identity mismatch")
    for recipe in library.recipes:
        if replay_snapshot_recipe(simulator, recipe) != store[recipe.snapshot_id]:
            raise ValueError("snapshot bundle replay bytes differ")
    return bundle


def pair_snapshot_bundles(
    portable: SnapshotBundle,
    exact: SnapshotBundle,
) -> dict[str, CrossBackendEvaluationManifest]:
    """Pair backend-specific recipes by construction provenance for evaluation."""

    from .r3b_evaluation import (
        CrossBackendCellPair,
        CrossBackendEvaluationManifest,
        LogicalEvaluationCell,
    )

    if (
        not isinstance(portable, SnapshotBundle)
        or not isinstance(exact, SnapshotBundle)
        or portable.runtime_backend != "portable"
        or exact.runtime_backend != "exact"
        or portable.source.action_spec_sha256 != exact.source.action_spec_sha256
    ):
        raise ValueError("snapshot pairing requires portable and exact bundles")

    result: dict[str, CrossBackendEvaluationManifest] = {}
    for split in ("calibration", "validation", "test"):
        portable_recipes = tuple(
            recipe for recipe in portable.library.recipes if recipe.split == split
        )
        exact_recipes = tuple(
            recipe for recipe in exact.library.recipes if recipe.split == split
        )

        def indexed(recipes: tuple[SnapshotRecipe, ...]) -> dict[str, SnapshotRecipe]:
            values = {
                LogicalEvaluationCell.from_recipe(recipe).sha256: recipe
                for recipe in recipes
            }
            if len(values) != len(recipes):
                raise ValueError("snapshot bundle repeats a logical evaluation cell")
            return values

        portable_by_cell = indexed(portable_recipes)
        exact_by_cell = indexed(exact_recipes)
        if not portable_by_cell or set(portable_by_cell) != set(exact_by_cell):
            raise ValueError(f"{split} snapshot bundles lack identical logical cells")
        manifest = CrossBackendEvaluationManifest(
            tuple(
                CrossBackendCellPair.from_recipes(
                    portable_by_cell[cell], exact_by_cell[cell]
                )
                for cell in sorted(portable_by_cell)
            )
        )
        result[split] = manifest
    return result
