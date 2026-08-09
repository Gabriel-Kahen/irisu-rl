"""Fail-closed, repository-independent evidence helpers for the R3I/G3 campaign.

This module deliberately depends only on the Python standard library.  It does
not discover repository state or import campaign/model code.  Callers must
provide every identity, seed set, path, and gate report explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CANONICAL_JSON_ENCODING = "utf-8"
SEED_MANIFEST_SCHEMA = "irisu.r3i.g3.seed-manifest.v1"
COLLECTION_CLOSURE_SCHEMA = "irisu.r3i.g3.collection-closure.v1"
STAGE_RECEIPT_SCHEMA = "irisu.r3i.g3.stage-receipt.v1"
PARTITION_SCHEMA = "irisu.r3i.g3.whole-seed-8-fold.v1"
OOF_CLOSURE_SCHEMA = "irisu.r3i.g3.oof-closure.v1"
THRESHOLD_RULE_SCHEMA = "irisu.r3i.g3.threshold-rule.v1"
THRESHOLD_REPORT_SCHEMA = "irisu.r3i.g3.threshold-calibration.v1"
AUTHORIZATION_SCHEMA = "irisu.r3i.g3.authorization.v1"
SLOT_PLAN_SCHEMA = "irisu.r3i.g3.two-slot-plan.v1"
AFFINITY_REPORT_SCHEMA = "irisu.r3i.g3.affinity-report.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_SAFE_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_THREAD_LIMIT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


class EvidenceError(ValueError):
    """Evidence is incomplete, malformed, ambiguous, or no longer live."""


class AuthorizationError(EvidenceError):
    """A downstream stage is not authorized."""


def _require_exact(value: Any, expected: type, field: str) -> None:
    if type(value) is not expected:
        raise EvidenceError(f"{field} must be exact {expected.__name__}")


def _require_sha256(value: Any, field: str) -> str:
    _require_exact(value, str, field)
    if _SHA256_RE.fullmatch(value) is None:
        raise EvidenceError(f"{field} must be a lowercase SHA-256")
    return value


def _require_token(value: Any, field: str) -> str:
    _require_exact(value, str, field)
    if _TOKEN_RE.fullmatch(value) is None:
        raise EvidenceError(f"{field} must be a lowercase campaign token")
    return value


def _validate_json_value(value: Any, field: str = "$") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        if value_type is str:
            try:
                value.encode(CANONICAL_JSON_ENCODING)
            except UnicodeEncodeError as exc:
                raise EvidenceError(f"{field} is not valid UTF-8 text") from exc
        return
    if value_type is float:
        if not math.isfinite(value):
            raise EvidenceError(f"{field} contains a non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvidenceError(f"{field} contains a non-string key")
            _validate_json_value(key, f"{field}.<key>")
            _validate_json_value(item, f"{field}.{key}")
        return
    raise EvidenceError(f"{field} contains unsupported exact type {value_type.__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted UTF-8 JSON encoding (without a trailing newline)."""

    _validate_json_value(value)
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return rendered.encode(CANONICAL_JSON_ENCODING)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EvidenceError("value cannot be canonically encoded") from exc


def sha256_bytes(value: bytes) -> str:
    _require_exact(value, bytes, "value")
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def seal_record(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    _require_exact(payload, dict, "payload")
    _require_exact(digest_field, str, "digest_field")
    if digest_field in payload:
        raise EvidenceError(f"payload already contains {digest_field}")
    _validate_json_value(payload)
    result = dict(payload)
    result[digest_field] = sha256_json(payload)
    return result


def verify_sealed_record(record: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    _require_exact(record, dict, "record")
    _require_exact(digest_field, str, "digest_field")
    if digest_field not in record:
        raise EvidenceError(f"record is missing {digest_field}")
    claimed = _require_sha256(record[digest_field], digest_field)
    payload = dict(record)
    del payload[digest_field]
    if sha256_json(payload) != claimed:
        raise EvidenceError(f"{digest_field} does not bind the record")
    return dict(record)


def _absolute_lexical(path: os.PathLike[str] | str) -> Path:
    if isinstance(path, (str, os.PathLike)):
        return Path(os.path.abspath(os.fspath(path)))
    raise EvidenceError("path must be path-like")


def _validate_directory(path: Path, field: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"{field} is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EvidenceError(f"{field} must be a real directory: {path}")
    return info


def _validate_parent_chain(root: Path, target: Path) -> Path:
    root = _absolute_lexical(root)
    target = _absolute_lexical(target)
    _validate_directory(root, "root")
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise EvidenceError("target escapes the supplied root") from exc
    if not relative.parts:
        raise EvidenceError("target must be below the supplied root")
    cursor = root
    for component in relative.parts[:-1]:
        if component in ("", ".", ".."):
            raise EvidenceError("target has an unsafe path component")
        cursor = cursor / component
        _validate_directory(cursor, "target parent")
    return relative


def validate_regular_file(
    path: os.PathLike[str] | str,
    *,
    root: os.PathLike[str] | str | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> dict[str, Any]:
    """Bind a live, non-symlink, single-link file and detect concurrent mutation."""

    target = _absolute_lexical(path)
    binding_root = target.parent if root is None else _absolute_lexical(root)
    relative = _validate_parent_chain(binding_root, target)
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
    if expected_size is not None:
        _require_exact(expected_size, int, "expected_size")
        if expected_size < 0:
            raise EvidenceError("expected_size must be non-negative")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise EvidenceError(f"file cannot be opened without following links: {target}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise EvidenceError("file must be regular and have exactly one hard link")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise EvidenceError("file identity changed while it was hashed")
    try:
        live = target.lstat()
    except OSError as exc:
        raise EvidenceError("file disappeared after hashing") from exc
    if (
        live.st_dev != after.st_dev
        or live.st_ino != after.st_ino
        or live.st_nlink != 1
        or stat.S_ISLNK(live.st_mode)
        or not stat.S_ISREG(live.st_mode)
    ):
        raise EvidenceError("path no longer names the hashed single-link file")
    actual_sha256 = digest.hexdigest()
    if size != after.st_size:
        raise EvidenceError("hashed byte count does not match file size")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise EvidenceError("file SHA-256 mismatch")
    if expected_size is not None and size != expected_size:
        raise EvidenceError("file size mismatch")

    return {
        "schema": "irisu.r3i.g3.file-binding.v1",
        "path": relative.as_posix(),
        "size": size,
        "sha256": actual_sha256,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": stat.S_IMODE(after.st_mode),
        "mtime_ns": after.st_mtime_ns,
    }


def validate_file_binding(
    binding: Mapping[str, Any],
    *,
    root: os.PathLike[str] | str,
) -> dict[str, Any]:
    _require_exact(binding, dict, "binding")
    if binding.get("schema") != "irisu.r3i.g3.file-binding.v1":
        raise EvidenceError("unknown file-binding schema")
    required = {
        "schema",
        "path",
        "size",
        "sha256",
        "device",
        "inode",
        "mode",
        "mtime_ns",
    }
    if set(binding) != required:
        raise EvidenceError("file binding has missing or extra fields")
    _require_exact(binding["path"], str, "binding.path")
    relative = Path(binding["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise EvidenceError("binding.path is not a safe relative path")
    for field in ("size", "device", "inode", "mode", "mtime_ns"):
        _require_exact(binding[field], int, f"binding.{field}")
    _require_sha256(binding["sha256"], "binding.sha256")
    live = validate_regular_file(
        _absolute_lexical(root) / relative,
        root=root,
        expected_sha256=binding["sha256"],
        expected_size=binding["size"],
    )
    if live != binding:
        raise EvidenceError("live file identity no longer matches its binding")
    return live


def write_once_atomic_json(
    path: os.PathLike[str] | str,
    value: Any,
    *,
    mode: int = 0o444,
) -> dict[str, Any]:
    """Publish canonical JSON atomically without any overwrite-capable operation."""

    target = _absolute_lexical(path)
    parent = target.parent
    _validate_directory(parent, "target parent")
    _require_exact(mode, int, "mode")
    if mode < 0 or mode > 0o777:
        raise EvidenceError("mode must be a permission mask")
    if not _SAFE_FILENAME_RE.fullmatch(target.name):
        raise EvidenceError("target filename is unsafe")
    data = canonical_json_bytes(value) + b"\n"

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as exc:
        raise EvidenceError("target parent cannot be opened safely") from exc
    temporary_name = f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    temporary_created = False
    try:
        try:
            os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("write-once target already exists")

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        temporary_created = True
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise EvidenceError("short write while publishing JSON")
                view = view[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise EvidenceError("write-once target appeared during publication") from exc
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_created = False
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)

    return validate_regular_file(
        target,
        root=parent,
        expected_sha256=sha256_bytes(data),
        expected_size=len(data),
    )


def _derive_seed(
    namespace: str,
    master_seed: int,
    split: str,
    index: int,
    seed_bits: int,
) -> int:
    payload = {
        "namespace": namespace,
        "master_seed": master_seed,
        "split": split,
        "index": index,
        "seed_bits": seed_bits,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big") % (
        1 << seed_bits
    )


def build_seed_manifest(
    namespace: str,
    master_seed: int,
    split_sizes: Mapping[str, int],
    *,
    seed_bits: int = 31,
) -> dict[str, Any]:
    _require_token(namespace, "namespace")
    _require_exact(master_seed, int, "master_seed")
    if master_seed < 0:
        raise EvidenceError("master_seed must be non-negative")
    _require_exact(split_sizes, dict, "split_sizes")
    _require_exact(seed_bits, int, "seed_bits")
    if not 16 <= seed_bits <= 63:
        raise EvidenceError("seed_bits must be in [16, 63]")
    if not split_sizes:
        raise EvidenceError("at least one split is required")

    seen: set[int] = set()
    splits: list[dict[str, Any]] = []
    for split in sorted(split_sizes):
        _require_token(split, "split name")
        size = split_sizes[split]
        _require_exact(size, int, f"split_sizes.{split}")
        if size <= 0:
            raise EvidenceError("split sizes must be positive")
        seeds = []
        for index in range(size):
            seed = _derive_seed(namespace, master_seed, split, index, seed_bits)
            if seed in seen:
                raise EvidenceError("derived seed collision; choose another master_seed")
            seen.add(seed)
            seeds.append(seed)
        splits.append({"name": split, "seeds": seeds})
    return seal_record(
        {
            "schema": SEED_MANIFEST_SCHEMA,
            "namespace": namespace,
            "master_seed": master_seed,
            "seed_bits": seed_bits,
            "splits": splits,
        },
        "manifest_sha256",
    )


def validate_seed_manifest(
    manifest: Mapping[str, Any],
    *,
    required_splits: Sequence[str] | None = None,
) -> dict[str, tuple[int, ...]]:
    verified = verify_sealed_record(manifest, "manifest_sha256")
    if verified.get("schema") != SEED_MANIFEST_SCHEMA:
        raise EvidenceError("unknown seed-manifest schema")
    required_fields = {
        "schema",
        "namespace",
        "master_seed",
        "seed_bits",
        "splits",
        "manifest_sha256",
    }
    if set(verified) != required_fields:
        raise EvidenceError("seed manifest has missing or extra fields")
    namespace = _require_token(verified["namespace"], "namespace")
    master_seed = verified["master_seed"]
    seed_bits = verified["seed_bits"]
    _require_exact(master_seed, int, "master_seed")
    _require_exact(seed_bits, int, "seed_bits")
    if master_seed < 0 or not 16 <= seed_bits <= 63:
        raise EvidenceError("invalid seed derivation parameters")
    _require_exact(verified["splits"], list, "splits")
    if not verified["splits"]:
        raise EvidenceError("seed manifest has no splits")

    result: dict[str, tuple[int, ...]] = {}
    all_seeds: set[int] = set()
    names: list[str] = []
    for split_record in verified["splits"]:
        _require_exact(split_record, dict, "split record")
        if set(split_record) != {"name", "seeds"}:
            raise EvidenceError("split record has missing or extra fields")
        name = _require_token(split_record["name"], "split name")
        _require_exact(split_record["seeds"], list, f"{name}.seeds")
        if not split_record["seeds"]:
            raise EvidenceError("every split must contain a seed")
        seeds: list[int] = []
        for index, seed in enumerate(split_record["seeds"]):
            _require_exact(seed, int, f"{name}.seeds[{index}]")
            expected = _derive_seed(namespace, master_seed, name, index, seed_bits)
            if seed != expected:
                raise EvidenceError("seed does not match deterministic derivation")
            if seed in all_seeds:
                raise EvidenceError("seed splits are not disjoint")
            all_seeds.add(seed)
            seeds.append(seed)
        if name in result:
            raise EvidenceError("duplicate split name")
        result[name] = tuple(seeds)
        names.append(name)
    if names != sorted(names):
        raise EvidenceError("split records are not in canonical name order")

    if required_splits is not None:
        _require_exact(required_splits, list, "required_splits")
        expected_names = []
        for name in required_splits:
            expected_names.append(_require_token(name, "required split"))
        if len(set(expected_names)) != len(expected_names):
            raise EvidenceError("required_splits contains duplicates")
        if set(result) != set(expected_names):
            raise EvidenceError("seed manifest does not have the exact required splits")
    return result


def assert_split_disjointness(splits: Mapping[str, Sequence[int]]) -> None:
    _require_exact(splits, dict, "splits")
    seen: set[int] = set()
    for name in sorted(splits):
        _require_token(name, "split name")
        values = splits[name]
        if type(values) not in (list, tuple):
            raise EvidenceError("split seeds must be an exact list or tuple")
        local: set[int] = set()
        for seed in values:
            _require_exact(seed, int, "seed")
            if seed in local or seed in seen:
                raise EvidenceError("seed splits are not disjoint")
            local.add(seed)
            seen.add(seed)


def _validate_seed_sequence(seeds: Sequence[int], field: str) -> tuple[int, ...]:
    if type(seeds) not in (list, tuple):
        raise EvidenceError(f"{field} must be an exact list or tuple")
    result = []
    seen: set[int] = set()
    for seed in seeds:
        _require_exact(seed, int, field)
        if seed < 0 or seed in seen:
            raise EvidenceError(f"{field} must contain unique non-negative seeds")
        seen.add(seed)
        result.append(seed)
    if not result:
        raise EvidenceError(f"{field} cannot be empty")
    return tuple(result)


def validate_collection_closure(
    root: os.PathLike[str] | str,
    expected_seeds: Sequence[int],
    *,
    layout: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Require the exact direct-file Cartesian product of seeds and roles."""

    collection_root = _absolute_lexical(root)
    _validate_directory(collection_root, "collection root")
    seeds = tuple(sorted(_validate_seed_sequence(expected_seeds, "expected_seeds")))
    if layout is None:
        role_layout: dict[str, str] = {
            "episode": "{seed}.episode.json",
            "queries": "{seed}.queries.jsonl",
        }
    else:
        _require_exact(layout, dict, "layout")
        role_layout = dict(layout)
    if not role_layout:
        raise EvidenceError("collection layout cannot be empty")

    expected_names: dict[str, tuple[int, str]] = {}
    for role in sorted(role_layout):
        _require_token(role, "collection role")
        template = role_layout[role]
        _require_exact(template, str, f"layout.{role}")
        if template.count("{seed}") != 1:
            raise EvidenceError("each layout template must contain exactly one {seed}")
        for seed in seeds:
            name = template.format(seed=seed)
            if (
                not _SAFE_FILENAME_RE.fullmatch(name)
                or "/" in name
                or "\\" in name
                or name in expected_names
            ):
                raise EvidenceError("layout produces an unsafe or duplicate filename")
            expected_names[name] = (seed, role)

    try:
        actual_names = {entry.name for entry in os.scandir(collection_root)}
    except OSError as exc:
        raise EvidenceError("collection root cannot be listed") from exc
    if actual_names != set(expected_names):
        missing = sorted(set(expected_names) - actual_names)
        extra = sorted(actual_names - set(expected_names))
        raise EvidenceError(f"collection is not closed (missing={missing}, extra={extra})")

    files = []
    for name in sorted(expected_names):
        seed, role = expected_names[name]
        binding = validate_regular_file(collection_root / name, root=collection_root)
        files.append({"seed": seed, "role": role, "binding": binding})
    return seal_record(
        {
            "schema": COLLECTION_CLOSURE_SCHEMA,
            "seeds": list(seeds),
            "roles": sorted(role_layout),
            "files": files,
        },
        "closure_sha256",
    )


def build_stage_receipt(
    *,
    campaign_sha256: str,
    stage: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    prerequisite_receipt_sha256s: Sequence[str] = (),
    passed: bool,
    terminal: bool,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    campaign_sha256 = _require_sha256(campaign_sha256, "campaign_sha256")
    stage = _require_token(stage, "stage")
    _require_exact(inputs, dict, "inputs")
    _require_exact(outputs, dict, "outputs")
    if type(prerequisite_receipt_sha256s) not in (list, tuple):
        raise EvidenceError("prerequisite receipts must be an exact list or tuple")
    prerequisites = [
        _require_sha256(item, "prerequisite receipt")
        for item in prerequisite_receipt_sha256s
    ]
    if len(prerequisites) != len(set(prerequisites)):
        raise EvidenceError("prerequisite receipts contain duplicates")
    if prerequisites != sorted(prerequisites):
        raise EvidenceError("prerequisite receipts must be in SHA order")
    _require_exact(passed, bool, "passed")
    _require_exact(terminal, bool, "terminal")
    if not passed and not terminal:
        raise EvidenceError("a failed stage receipt must be terminal")
    if metadata is None:
        metadata_value: dict[str, Any] = {}
    else:
        _require_exact(metadata, dict, "metadata")
        metadata_value = dict(metadata)
    _validate_json_value(inputs, "inputs")
    _validate_json_value(outputs, "outputs")
    _validate_json_value(metadata_value, "metadata")
    return seal_record(
        {
            "schema": STAGE_RECEIPT_SCHEMA,
            "campaign_sha256": campaign_sha256,
            "stage": stage,
            "passed": passed,
            "terminal": terminal,
            "prerequisite_receipt_sha256s": prerequisites,
            "inputs": dict(inputs),
            "outputs": dict(outputs),
            "metadata": metadata_value,
        },
        "receipt_sha256",
    )


def validate_stage_receipt(
    receipt: Mapping[str, Any],
    *,
    campaign_sha256: str | None = None,
    stage: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    prerequisite_receipt_sha256s: Sequence[str] | None = None,
) -> dict[str, Any]:
    verified = verify_sealed_record(receipt, "receipt_sha256")
    required = {
        "schema",
        "campaign_sha256",
        "stage",
        "passed",
        "terminal",
        "prerequisite_receipt_sha256s",
        "inputs",
        "outputs",
        "metadata",
        "receipt_sha256",
    }
    if set(verified) != required or verified.get("schema") != STAGE_RECEIPT_SCHEMA:
        raise EvidenceError("stage receipt has an unknown schema or field set")
    rebuilt = build_stage_receipt(
        campaign_sha256=verified["campaign_sha256"],
        stage=verified["stage"],
        inputs=verified["inputs"],
        outputs=verified["outputs"],
        prerequisite_receipt_sha256s=verified["prerequisite_receipt_sha256s"],
        passed=verified["passed"],
        terminal=verified["terminal"],
        metadata=verified["metadata"],
    )
    if rebuilt != verified:
        raise EvidenceError("stage receipt is not canonical")
    if campaign_sha256 is not None and verified["campaign_sha256"] != _require_sha256(
        campaign_sha256, "campaign_sha256"
    ):
        raise EvidenceError("stage receipt belongs to another campaign")
    if stage is not None and verified["stage"] != _require_token(stage, "stage"):
        raise EvidenceError("stage receipt has the wrong stage")
    if inputs is not None and verified["inputs"] != inputs:
        raise EvidenceError("stage receipt input binding mismatch")
    if outputs is not None and verified["outputs"] != outputs:
        raise EvidenceError("stage receipt output binding mismatch")
    if prerequisite_receipt_sha256s is not None:
        if type(prerequisite_receipt_sha256s) not in (list, tuple):
            raise EvidenceError("prerequisite receipts must be an exact list or tuple")
        expected = list(prerequisite_receipt_sha256s)
        if verified["prerequisite_receipt_sha256s"] != expected:
            raise EvidenceError("stage receipt prerequisite mismatch")
    return verified


def build_whole_seed_partition(
    seeds: Sequence[int],
    *,
    namespace: str,
    fold_count: int = 8,
) -> dict[str, Any]:
    """Create a deterministic partition where each seed is held out once."""

    seed_values = _validate_seed_sequence(seeds, "seeds")
    namespace = _require_token(namespace, "namespace")
    _require_exact(fold_count, int, "fold_count")
    if fold_count != 8:
        raise EvidenceError("the G3 campaign requires exactly eight folds")
    if len(seed_values) < fold_count:
        raise EvidenceError("eight-fold partition requires at least eight seeds")

    ordered = sorted(
        seed_values,
        key=lambda seed: (
            hashlib.sha256(
                canonical_json_bytes({"namespace": namespace, "seed": seed})
            ).digest(),
            seed,
        ),
    )
    heldout: list[list[int]] = [[] for _ in range(fold_count)]
    for index, seed in enumerate(ordered):
        heldout[index % fold_count].append(seed)
    folds = [
        {"fold": fold, "heldout_seeds": sorted(fold_seeds)}
        for fold, fold_seeds in enumerate(heldout)
    ]
    return seal_record(
        {
            "schema": PARTITION_SCHEMA,
            "namespace": namespace,
            "fold_count": fold_count,
            "seeds": sorted(seed_values),
            "folds": folds,
        },
        "partition_sha256",
    )


def validate_whole_seed_partition(
    partition: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> dict[int, int]:
    verified = verify_sealed_record(partition, "partition_sha256")
    required = {
        "schema",
        "namespace",
        "fold_count",
        "seeds",
        "folds",
        "partition_sha256",
    }
    if set(verified) != required or verified.get("schema") != PARTITION_SCHEMA:
        raise EvidenceError("whole-seed partition has an unknown schema or field set")
    rebuilt = build_whole_seed_partition(
        verified["seeds"],
        namespace=verified["namespace"],
        fold_count=verified["fold_count"],
    )
    if rebuilt != verified:
        raise EvidenceError("whole-seed partition is not the deterministic partition")
    if expected_seeds is not None:
        expected = sorted(_validate_seed_sequence(expected_seeds, "expected_seeds"))
        if verified["seeds"] != expected:
            raise EvidenceError("partition does not close over the expected seeds")
    seed_to_fold: dict[int, int] = {}
    for fold_record in verified["folds"]:
        for seed in fold_record["heldout_seeds"]:
            if seed in seed_to_fold:
                raise EvidenceError("a seed is held out in more than one fold")
            seed_to_fold[seed] = fold_record["fold"]
    if set(seed_to_fold) != set(verified["seeds"]):
        raise EvidenceError("not every seed is held out exactly once")
    return seed_to_fold


def validate_oof_closure(
    partition: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    expected_items: Mapping[int, Sequence[str]] | None = None,
    seed_key: str = "seed",
    fold_key: str = "fold",
    item_key: str = "item_id",
) -> dict[str, Any]:
    seed_to_fold = validate_whole_seed_partition(partition)
    _require_exact(predictions, list, "predictions")
    for key, name in (
        (seed_key, "seed_key"),
        (fold_key, "fold_key"),
        (item_key, "item_key"),
    ):
        _require_exact(key, str, name)

    if expected_items is None:
        expected_pairs = {(seed, None) for seed in seed_to_fold}
        require_item = False
    else:
        _require_exact(expected_items, dict, "expected_items")
        if set(expected_items) != set(seed_to_fold):
            raise EvidenceError("expected_items must cover the exact partition seed set")
        expected_pairs: set[tuple[int, str | None]] = set()
        for seed in sorted(expected_items):
            _require_exact(seed, int, "expected_items seed")
            items = expected_items[seed]
            if type(items) not in (list, tuple) or not items:
                raise EvidenceError("each seed must have a non-empty exact item list")
            for item in items:
                _require_exact(item, str, "item id")
                pair = (seed, item)
                if pair in expected_pairs:
                    raise EvidenceError("expected item ids contain duplicates")
                expected_pairs.add(pair)
        require_item = True

    seen: set[tuple[int, str | None]] = set()
    for index, row in enumerate(predictions):
        _require_exact(row, dict, f"predictions[{index}]")
        _validate_json_value(row, f"predictions[{index}]")
        if seed_key not in row or fold_key not in row:
            raise EvidenceError("OOF prediction is missing seed or fold")
        seed = row[seed_key]
        fold = row[fold_key]
        _require_exact(seed, int, "prediction seed")
        _require_exact(fold, int, "prediction fold")
        if seed not in seed_to_fold or fold != seed_to_fold[seed]:
            raise EvidenceError("prediction was not made by its seed's heldout fold")
        if require_item:
            if item_key not in row:
                raise EvidenceError("OOF prediction is missing item id")
            item = row[item_key]
            _require_exact(item, str, "prediction item id")
        else:
            item = None
        pair = (seed, item)
        if pair in seen:
            raise EvidenceError("duplicate OOF prediction")
        seen.add(pair)
    if seen != expected_pairs:
        raise EvidenceError("OOF predictions do not exactly close over expected items")
    return seal_record(
        {
            "schema": OOF_CLOSURE_SCHEMA,
            "partition_sha256": partition["partition_sha256"],
            "prediction_count": len(predictions),
            "prediction_sha256": sha256_json(predictions),
            "seed_count": len(seed_to_fold),
        },
        "closure_sha256",
    )


def build_threshold_rule(
    *,
    max_false_positives: int = 0,
    minimum_selected: int = 1,
) -> dict[str, Any]:
    _require_exact(max_false_positives, int, "max_false_positives")
    _require_exact(minimum_selected, int, "minimum_selected")
    if max_false_positives < 0 or minimum_selected <= 0:
        raise EvidenceError("threshold rule counts are out of range")
    return seal_record(
        {
            "schema": THRESHOLD_RULE_SCHEMA,
            "comparator": ">=",
            "candidate_set": "all-exact-float-tie-blocks",
            "objective": "maximum-selected-then-highest-threshold",
            "max_false_positives": max_false_positives,
            "minimum_selected": minimum_selected,
        },
        "rule_sha256",
    )


def _validate_threshold_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    verified = verify_sealed_record(rule, "rule_sha256")
    required = {
        "schema",
        "comparator",
        "candidate_set",
        "objective",
        "max_false_positives",
        "minimum_selected",
        "rule_sha256",
    }
    if set(verified) != required:
        raise EvidenceError("threshold rule has missing or extra fields")
    rebuilt = build_threshold_rule(
        max_false_positives=verified["max_false_positives"],
        minimum_selected=verified["minimum_selected"],
    )
    if verified != rebuilt:
        raise EvidenceError("threshold rule is not the fixed G3 rule")
    return verified


def calibrate_threshold(
    observations: Sequence[Mapping[str, Any]],
    *,
    rule: Mapping[str, Any],
    score_key: str = "score",
    label_key: str = "acceptable",
) -> dict[str, Any]:
    """Evaluate every exact observed score boundary; never split an equal-score tie."""

    _require_exact(observations, list, "observations")
    fixed_rule = _validate_threshold_rule(rule)
    _require_exact(score_key, str, "score_key")
    _require_exact(label_key, str, "label_key")
    if not observations:
        raise EvidenceError("threshold calibration needs observations")

    pairs: list[tuple[float, bool]] = []
    for index, row in enumerate(observations):
        _require_exact(row, dict, f"observations[{index}]")
        _validate_json_value(row, f"observations[{index}]")
        if score_key not in row or label_key not in row:
            raise EvidenceError("threshold observation is missing score or label")
        score = row[score_key]
        label = row[label_key]
        _require_exact(score, float, "threshold score")
        _require_exact(label, bool, "threshold label")
        if not math.isfinite(score):
            raise EvidenceError("threshold score must be finite")
        pairs.append((score, label))

    groups: dict[float, list[bool]] = {}
    for score, label in pairs:
        groups.setdefault(score, []).append(label)
    selected_count = 0
    true_count = 0
    false_count = 0
    best: tuple[float, int, int, int, int] | None = None
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        selected_count += len(labels)
        true_count += sum(labels)
        false_count += len(labels) - sum(labels)
        if false_count <= fixed_rule["max_false_positives"]:
            best = (score, selected_count, true_count, false_count, len(labels))

    if best is None:
        threshold: float | None = None
        selected_count = true_count = false_count = tie_block_size = 0
    else:
        threshold, selected_count, true_count, false_count, tie_block_size = best
    passed = best is not None and selected_count >= fixed_rule["minimum_selected"]
    return seal_record(
        {
            "schema": THRESHOLD_REPORT_SCHEMA,
            "passed": passed,
            "terminal": not passed,
            "rule": fixed_rule,
            "input_sha256": sha256_json(observations),
            "observation_count": len(observations),
            "unique_score_count": len(groups),
            "threshold": threshold,
            "comparator": ">=",
            "selected_count": selected_count,
            "acceptable_count": true_count,
            "false_positive_count": false_count,
            "threshold_tie_block_size": tie_block_size,
        },
        "calibration_sha256",
    )


def audit_threshold_calibration(
    observations: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    fixed_rule: Mapping[str, Any],
    score_key: str = "score",
    label_key: str = "acceptable",
) -> dict[str, Any]:
    verify_sealed_record(report, "calibration_sha256")
    expected = calibrate_threshold(
        observations,
        rule=fixed_rule,
        score_key=score_key,
        label_key=label_key,
    )
    if report != expected:
        raise EvidenceError("threshold report does not reproduce under the fixed rule")
    return dict(report)


def _validate_gate_report(report: Mapping[str, Any]) -> None:
    _require_exact(report, dict, "gate report")
    _validate_json_value(report, "gate report")
    _require_exact(report.get("passed"), bool, "gate report passed")
    _require_exact(report.get("terminal"), bool, "gate report terminal")
    if report["passed"] is False and report["terminal"] is not True:
        raise EvidenceError("a failed gate report must be terminal")


def build_authorization(
    *,
    campaign_sha256: str,
    required_stages: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    gate_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Authorize only an exact all-passing receipt set with no terminal failure."""

    campaign_sha256 = _require_sha256(campaign_sha256, "campaign_sha256")
    if type(required_stages) not in (list, tuple) or not required_stages:
        raise EvidenceError("required_stages must be a non-empty exact sequence")
    stages = [_require_token(stage, "required stage") for stage in required_stages]
    if len(stages) != len(set(stages)):
        raise EvidenceError("required_stages contains duplicates")
    if type(receipts) not in (list, tuple):
        raise EvidenceError("receipts must be an exact list or tuple")
    if type(gate_reports) not in (list, tuple):
        raise EvidenceError("gate_reports must be an exact list or tuple")

    by_stage: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        verified = validate_stage_receipt(receipt, campaign_sha256=campaign_sha256)
        stage = verified["stage"]
        if stage in by_stage:
            raise AuthorizationError("duplicate stage receipt")
        by_stage[stage] = verified
    if set(by_stage) != set(stages):
        raise AuthorizationError("receipt set does not match the exact required stages")
    for stage in stages:
        receipt = by_stage[stage]
        if receipt["passed"] is not True:
            raise AuthorizationError(f"stage {stage} is terminal-failed")

    gate_bindings = []
    for report in gate_reports:
        _validate_gate_report(report)
        if report["passed"] is not True:
            raise AuthorizationError("terminal passed:false gate blocks authorization")
        gate_bindings.append(sha256_json(report))
    if len(gate_bindings) != len(set(gate_bindings)):
        raise EvidenceError("duplicate gate report")

    return seal_record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "campaign_sha256": campaign_sha256,
            "passed": True,
            "required_stages": stages,
            "receipts": [
                {"stage": stage, "receipt_sha256": by_stage[stage]["receipt_sha256"]}
                for stage in stages
            ],
            "gate_report_sha256s": gate_bindings,
        },
        "authorization_sha256",
    )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    receipts: Sequence[Mapping[str, Any]],
    gate_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    verified = verify_sealed_record(authorization, "authorization_sha256")
    required = {
        "schema",
        "campaign_sha256",
        "passed",
        "required_stages",
        "receipts",
        "gate_report_sha256s",
        "authorization_sha256",
    }
    if (
        set(verified) != required
        or verified.get("schema") != AUTHORIZATION_SCHEMA
        or verified.get("passed") is not True
    ):
        raise AuthorizationError("authorization has an invalid schema or status")
    rebuilt = build_authorization(
        campaign_sha256=verified["campaign_sha256"],
        required_stages=verified["required_stages"],
        receipts=receipts,
        gate_reports=gate_reports,
    )
    if rebuilt != verified:
        raise AuthorizationError("authorization does not bind the supplied evidence")
    return verified


def build_two_slot_plan(cpu_ids: Sequence[int]) -> dict[str, Any]:
    if type(cpu_ids) not in (list, tuple) or len(cpu_ids) != 2:
        raise EvidenceError("exactly two CPU ids are required")
    cpus = []
    for cpu in cpu_ids:
        _require_exact(cpu, int, "cpu id")
        if cpu < 0:
            raise EvidenceError("CPU ids must be non-negative")
        cpus.append(cpu)
    if len(set(cpus)) != 2:
        raise EvidenceError("the two slots must use distinct CPUs")
    return seal_record(
        {
            "schema": SLOT_PLAN_SCHEMA,
            "max_concurrent": 2,
            "slots": [
                {"slot": index, "cpu": cpu} for index, cpu in enumerate(cpus)
            ],
            "thread_limits": {key: "1" for key in _THREAD_LIMIT_KEYS},
        },
        "plan_sha256",
    )


def validate_two_slot_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    verified = verify_sealed_record(plan, "plan_sha256")
    required = {
        "schema",
        "max_concurrent",
        "slots",
        "thread_limits",
        "plan_sha256",
    }
    if set(verified) != required or verified.get("schema") != SLOT_PLAN_SCHEMA:
        raise EvidenceError("two-slot plan has an unknown schema or field set")
    _require_exact(verified["slots"], list, "slots")
    if len(verified["slots"]) != 2:
        raise EvidenceError("two-slot plan must have exactly two slots")
    cpus = []
    for index, slot in enumerate(verified["slots"]):
        _require_exact(slot, dict, "slot")
        if set(slot) != {"slot", "cpu"} or slot.get("slot") != index:
            raise EvidenceError("slot records are not exact and ordered")
        _require_exact(slot["cpu"], int, "slot cpu")
        cpus.append(slot["cpu"])
    rebuilt = build_two_slot_plan(cpus)
    if rebuilt != verified:
        raise EvidenceError("two-slot plan is not canonical")
    return verified


def build_affinity_report(
    plan: Mapping[str, Any],
    *,
    slot: int,
    observed_cpus: Sequence[int],
    pid: int,
    observed_thread_limits: Mapping[str, str],
) -> dict[str, Any]:
    verified_plan = validate_two_slot_plan(plan)
    _require_exact(slot, int, "slot")
    if slot not in (0, 1):
        raise EvidenceError("slot must be 0 or 1")
    if type(observed_cpus) not in (list, tuple):
        raise EvidenceError("observed_cpus must be an exact list or tuple")
    cpus = []
    for cpu in observed_cpus:
        _require_exact(cpu, int, "observed cpu")
        cpus.append(cpu)
    expected_cpu = verified_plan["slots"][slot]["cpu"]
    if cpus != [expected_cpu]:
        raise EvidenceError("worker affinity is not the planned singleton CPU")
    _require_exact(pid, int, "pid")
    if pid <= 0:
        raise EvidenceError("pid must be positive")
    _require_exact(observed_thread_limits, dict, "observed_thread_limits")
    if observed_thread_limits != verified_plan["thread_limits"]:
        raise EvidenceError("worker thread limits do not match the plan")
    return seal_record(
        {
            "schema": AFFINITY_REPORT_SCHEMA,
            "plan_sha256": verified_plan["plan_sha256"],
            "slot": slot,
            "cpu": expected_cpu,
            "observed_cpus": cpus,
            "pid": pid,
            "thread_limits": dict(observed_thread_limits),
        },
        "report_sha256",
    )


def validate_affinity_reports(
    plan: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    verified_plan = validate_two_slot_plan(plan)
    if type(reports) not in (list, tuple) or len(reports) != 2:
        raise EvidenceError("exactly two affinity reports are required")
    result: dict[int, dict[str, Any]] = {}
    for report in reports:
        verified = verify_sealed_record(report, "report_sha256")
        required = {
            "schema",
            "plan_sha256",
            "slot",
            "cpu",
            "observed_cpus",
            "pid",
            "thread_limits",
            "report_sha256",
        }
        if (
            set(verified) != required
            or verified.get("schema") != AFFINITY_REPORT_SCHEMA
            or verified.get("plan_sha256") != verified_plan["plan_sha256"]
        ):
            raise EvidenceError("affinity report has an invalid schema or plan binding")
        slot = verified["slot"]
        if type(slot) is not int or slot not in (0, 1) or slot in result:
            raise EvidenceError("affinity reports do not cover each slot exactly once")
        rebuilt = build_affinity_report(
            verified_plan,
            slot=slot,
            observed_cpus=verified["observed_cpus"],
            pid=verified["pid"],
            observed_thread_limits=verified["thread_limits"],
        )
        if rebuilt != verified:
            raise EvidenceError("affinity report is not canonical")
        result[slot] = verified
    if set(result) != {0, 1}:
        raise EvidenceError("affinity reports do not cover both slots")
    return result


__all__ = [
    "AFFINITY_REPORT_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "AuthorizationError",
    "COLLECTION_CLOSURE_SCHEMA",
    "EvidenceError",
    "OOF_CLOSURE_SCHEMA",
    "PARTITION_SCHEMA",
    "SEED_MANIFEST_SCHEMA",
    "SLOT_PLAN_SCHEMA",
    "STAGE_RECEIPT_SCHEMA",
    "THRESHOLD_REPORT_SCHEMA",
    "THRESHOLD_RULE_SCHEMA",
    "assert_split_disjointness",
    "audit_threshold_calibration",
    "build_affinity_report",
    "build_authorization",
    "build_seed_manifest",
    "build_stage_receipt",
    "build_threshold_rule",
    "build_two_slot_plan",
    "build_whole_seed_partition",
    "calibrate_threshold",
    "canonical_json_bytes",
    "seal_record",
    "sha256_bytes",
    "sha256_json",
    "validate_affinity_reports",
    "validate_authorization",
    "validate_collection_closure",
    "validate_file_binding",
    "validate_oof_closure",
    "validate_regular_file",
    "validate_seed_manifest",
    "validate_stage_receipt",
    "validate_two_slot_plan",
    "validate_whole_seed_partition",
    "verify_sealed_record",
    "write_once_atomic_json",
]
