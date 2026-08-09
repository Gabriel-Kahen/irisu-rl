"""Corrective, fail-closed evidence primitives for the R3I/G3 campaign.

Version 2 is append-only relative to :mod:`r3i_g3_evidence`.  It reuses only
that module's canonical JSON, hashing, sealed-record, file-validation, stage
receipt, and write-once publication primitives, and pins the exact source bytes
of that dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import r3i_g3_evidence as _v1


EvidenceError = _v1.EvidenceError
AuthorizationError = _v1.AuthorizationError
canonical_json_bytes = _v1.canonical_json_bytes
seal_record = _v1.seal_record
sha256_bytes = _v1.sha256_bytes
sha256_json = _v1.sha256_json
verify_sealed_record = _v1.verify_sealed_record
write_once_atomic_json = _v1.write_once_atomic_json

V1_EVIDENCE_SOURCE_SHA256 = (
    "8f09f572f25f79bec4855a1742af1fe17966f62b61deb8dff287a0e45091c5b9"
)
DEPENDENCY_MANIFEST_SCHEMA = "irisu.r3i.g3.evidence-dependency.v2"
AUTHORIZATION_PREREGISTRATION_SCHEMA = (
    "irisu.r3i.g3.authorization-preregistration.v2"
)
AUTHORIZATION_SCHEMA = "irisu.r3i.g3.authorization.v2"
GATE_REPORT_SCHEMA = "irisu.r3i.g3.gate-report.v2"
COLLECTION_CLOSURE_SCHEMA = "irisu.r3i.g3.collection-closure.v2"
FILE_BINDING_SCHEMA = "irisu.r3i.g3.file-binding.v2"
CHECKPOINT_ENVELOPE_SCHEMA = "irisu.r3i.g3.checkpoint-envelope.v2"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_SAFE_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_DIRECTORY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_FILE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

_DEPENDENCY_MANIFEST = seal_record(
    {
        "schema": DEPENDENCY_MANIFEST_SCHEMA,
        "module": "irisu_pointer.r3i_g3_evidence",
        "source_sha256": V1_EVIDENCE_SOURCE_SHA256,
    },
    "manifest_sha256",
)
V1_DEPENDENCY_MANIFEST_SHA256 = _DEPENDENCY_MANIFEST["manifest_sha256"]


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


def _ordered_tokens(value: Any, field: str, *, nonempty: bool = True) -> list[str]:
    if type(value) not in (list, tuple):
        raise EvidenceError(f"{field} must be an exact list or tuple")
    result = [_require_token(item, field) for item in value]
    if nonempty and not result:
        raise EvidenceError(f"{field} cannot be empty")
    if len(result) != len(set(result)):
        raise EvidenceError(f"{field} contains duplicates")
    return result


def _require_exact_string_keys(value: dict[Any, Any], field: str) -> None:
    if any(type(key) is not str for key in value):
        raise EvidenceError(f"{field} keys must be exact strings")


def _canonical_dict_copy(value: Any, field: str) -> dict[str, Any]:
    _require_exact(value, dict, field)
    encoded = canonical_json_bytes(value)
    copied = json.loads(encoded.decode("utf-8"))
    _require_exact(copied, dict, field)
    return copied


def dependency_manifest() -> dict[str, Any]:
    """Return the stable manifest after revalidating the live v1 source file."""

    source = Path(os.path.abspath(os.fspath(_v1.__file__)))
    _v1.validate_regular_file(
        source,
        root=source.parent,
        expected_sha256=V1_EVIDENCE_SOURCE_SHA256,
    )
    return dict(_DEPENDENCY_MANIFEST)


def _bind_dependency() -> str:
    manifest = dependency_manifest()
    verify_sealed_record(manifest, "manifest_sha256")
    return manifest["manifest_sha256"]


def build_authorization_preregistration(
    *,
    campaign_sha256: str,
    receipt_stage_inventory: Sequence[str],
    gate_name_inventory: Sequence[str],
    prerequisite_topology: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Freeze exact ordered stage/gate inventories and an acyclic receipt DAG."""

    dependency_sha256 = _bind_dependency()
    campaign_sha256 = _require_sha256(campaign_sha256, "campaign_sha256")
    stages = _ordered_tokens(receipt_stage_inventory, "receipt_stage_inventory")
    gates = _ordered_tokens(gate_name_inventory, "gate_name_inventory")
    _require_exact(prerequisite_topology, dict, "prerequisite_topology")
    _require_exact_string_keys(prerequisite_topology, "prerequisite_topology")
    if set(prerequisite_topology) != set(stages):
        raise EvidenceError("prerequisite topology must have the exact stage set")

    stage_index = {stage: index for index, stage in enumerate(stages)}
    topology: list[dict[str, Any]] = []
    for stage in stages:
        predecessors = _ordered_tokens(
            prerequisite_topology[stage],
            f"prerequisite_topology.{stage}",
            nonempty=False,
        )
        if any(
            predecessor not in stage_index
            or stage_index[predecessor] >= stage_index[stage]
            for predecessor in predecessors
        ):
            raise EvidenceError(
                "every prerequisite must be a known earlier stage in the inventory"
            )
        canonical_predecessors = sorted(predecessors, key=stage_index.__getitem__)
        if predecessors != canonical_predecessors:
            raise EvidenceError("prerequisites must follow receipt-stage order")
        topology.append(
            {"stage": stage, "predecessor_stages": canonical_predecessors}
        )

    return seal_record(
        {
            "schema": AUTHORIZATION_PREREGISTRATION_SCHEMA,
            "v1_dependency_manifest_sha256": dependency_sha256,
            "campaign_sha256": campaign_sha256,
            "receipt_stage_inventory": stages,
            "gate_name_inventory": gates,
            "prerequisite_topology": topology,
        },
        "preregistration_sha256",
    )


def validate_authorization_preregistration(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    _bind_dependency()
    verified = verify_sealed_record(preregistration, "preregistration_sha256")
    required = {
        "schema",
        "v1_dependency_manifest_sha256",
        "campaign_sha256",
        "receipt_stage_inventory",
        "gate_name_inventory",
        "prerequisite_topology",
        "preregistration_sha256",
    }
    if (
        set(verified) != required
        or verified.get("schema") != AUTHORIZATION_PREREGISTRATION_SCHEMA
        or verified.get("v1_dependency_manifest_sha256")
        != V1_DEPENDENCY_MANIFEST_SHA256
    ):
        raise EvidenceError("authorization preregistration has an invalid schema")
    _require_exact(
        verified["prerequisite_topology"], list, "prerequisite_topology"
    )
    topology: dict[str, list[str]] = {}
    for item in verified["prerequisite_topology"]:
        _require_exact(item, dict, "prerequisite topology item")
        if set(item) != {"stage", "predecessor_stages"}:
            raise EvidenceError("prerequisite topology item has the wrong fields")
        stage = _require_token(item["stage"], "topology stage")
        if stage in topology:
            raise EvidenceError("duplicate topology stage")
        topology[stage] = item["predecessor_stages"]
    rebuilt = build_authorization_preregistration(
        campaign_sha256=verified["campaign_sha256"],
        receipt_stage_inventory=verified["receipt_stage_inventory"],
        gate_name_inventory=verified["gate_name_inventory"],
        prerequisite_topology=topology,
    )
    if rebuilt != verified:
        raise EvidenceError("authorization preregistration is not canonical")
    return verified


def build_gate_report(
    *,
    gate_name: str,
    campaign_sha256: str,
    preregistration_sha256: str,
    passed: bool,
    terminal: bool,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one named gate result against its campaign and preregistration."""

    _bind_dependency()
    gate_name = _require_token(gate_name, "gate_name")
    campaign_sha256 = _require_sha256(campaign_sha256, "campaign_sha256")
    preregistration_sha256 = _require_sha256(
        preregistration_sha256, "preregistration_sha256"
    )
    _require_exact(passed, bool, "passed")
    _require_exact(terminal, bool, "terminal")
    if passed is False and terminal is not True:
        raise EvidenceError("a failed gate report must be terminal")
    evidence_value = _canonical_dict_copy(evidence, "evidence")
    return seal_record(
        {
            "schema": GATE_REPORT_SCHEMA,
            "v1_dependency_manifest_sha256": V1_DEPENDENCY_MANIFEST_SHA256,
            "gate_name": gate_name,
            "campaign_sha256": campaign_sha256,
            "preregistration_sha256": preregistration_sha256,
            "passed": passed,
            "terminal": terminal,
            "evidence": evidence_value,
        },
        "gate_report_sha256",
    )


def _validate_gate_report(
    report: Any,
    *,
    mapping_name: str,
    campaign_sha256: str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    _require_exact(report, dict, f"gate_reports.{mapping_name}")
    verified = verify_sealed_record(report, "gate_report_sha256")
    required = {
        "schema",
        "v1_dependency_manifest_sha256",
        "gate_name",
        "campaign_sha256",
        "preregistration_sha256",
        "passed",
        "terminal",
        "evidence",
        "gate_report_sha256",
    }
    if (
        set(verified) != required
        or verified.get("schema") != GATE_REPORT_SCHEMA
        or verified.get("v1_dependency_manifest_sha256")
        != V1_DEPENDENCY_MANIFEST_SHA256
        or verified.get("gate_name") != mapping_name
        or verified.get("campaign_sha256") != campaign_sha256
        or verified.get("preregistration_sha256") != preregistration_sha256
    ):
        raise AuthorizationError(
            f"gate report {mapping_name} has a foreign or relabeled identity"
        )
    rebuilt = build_gate_report(
        gate_name=verified["gate_name"],
        campaign_sha256=verified["campaign_sha256"],
        preregistration_sha256=verified["preregistration_sha256"],
        passed=verified["passed"],
        terminal=verified["terminal"],
        evidence=verified["evidence"],
    )
    if rebuilt != verified:
        raise EvidenceError(f"gate report {mapping_name} is not canonical")
    return verified


def _authorization_inputs(
    preregistration: Mapping[str, Any],
    receipts_by_stage: Mapping[str, Mapping[str, Any]],
    gate_reports: Mapping[str, Mapping[str, Any]],
    *,
    expected_preregistration_sha256: str,
    expected_receipt_stage_inventory: Sequence[str],
    expected_gate_name_inventory: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prereg = validate_authorization_preregistration(preregistration)
    expected_preregistration_sha256 = _require_sha256(
        expected_preregistration_sha256, "expected_preregistration_sha256"
    )
    expected_stages = _ordered_tokens(
        expected_receipt_stage_inventory, "expected_receipt_stage_inventory"
    )
    expected_gates = _ordered_tokens(
        expected_gate_name_inventory, "expected_gate_name_inventory"
    )
    if prereg["preregistration_sha256"] != expected_preregistration_sha256:
        raise AuthorizationError("preregistration identity is not externally authorized")
    if prereg["receipt_stage_inventory"] != expected_stages:
        raise AuthorizationError("receipt-stage inventory differs from expectation")
    if prereg["gate_name_inventory"] != expected_gates:
        raise AuthorizationError("gate-name inventory differs from expectation")
    _require_exact(receipts_by_stage, dict, "receipts_by_stage")
    _require_exact(gate_reports, dict, "gate_reports")
    _require_exact_string_keys(receipts_by_stage, "receipts_by_stage")
    _require_exact_string_keys(gate_reports, "gate_reports")
    stages = prereg["receipt_stage_inventory"]
    gates = prereg["gate_name_inventory"]
    if set(receipts_by_stage) != set(stages):
        raise AuthorizationError(
            "receipt mapping does not close the preregistered stage inventory"
        )
    if set(gate_reports) != set(gates):
        raise AuthorizationError(
            "gate mapping does not close the preregistered gate inventory"
        )

    verified_receipts: dict[str, dict[str, Any]] = {}
    for stage in stages:
        receipt = receipts_by_stage[stage]
        _require_exact(receipt, dict, f"receipts_by_stage.{stage}")
        verified = _v1.validate_stage_receipt(
            receipt,
            campaign_sha256=prereg["campaign_sha256"],
            stage=stage,
        )
        if verified["passed"] is not True:
            raise AuthorizationError(f"stage {stage} is terminal-failed")
        verified_receipts[stage] = verified

    topology = {
        item["stage"]: item["predecessor_stages"]
        for item in prereg["prerequisite_topology"]
    }
    receipt_bindings: list[dict[str, Any]] = []
    for stage in stages:
        expected = sorted(
            verified_receipts[predecessor]["receipt_sha256"]
            for predecessor in topology[stage]
        )
        actual = verified_receipts[stage]["prerequisite_receipt_sha256s"]
        if actual != expected:
            raise AuthorizationError(
                f"stage {stage} has missing, extra, or stale prerequisite receipts"
            )
        receipt_bindings.append(
            {
                "stage": stage,
                "receipt_sha256": verified_receipts[stage]["receipt_sha256"],
                "predecessor_receipt_sha256s": expected,
            }
        )

    gate_bindings: list[dict[str, Any]] = []
    gate_identities: set[str] = set()
    for name in gates:
        report = _validate_gate_report(
            gate_reports[name],
            mapping_name=name,
            campaign_sha256=prereg["campaign_sha256"],
            preregistration_sha256=prereg["preregistration_sha256"],
        )
        if report["passed"] is not True:
            raise AuthorizationError(f"gate {name} is terminal-failed")
        identity = report["gate_report_sha256"]
        if identity in gate_identities:
            raise AuthorizationError("duplicate gate report identity")
        gate_identities.add(identity)
        gate_bindings.append(
            {"name": name, "gate_report_sha256": identity}
        )
    return prereg, receipt_bindings, gate_bindings


def build_authorization(
    *,
    preregistration: Mapping[str, Any],
    expected_preregistration_sha256: str,
    expected_receipt_stage_inventory: Sequence[str],
    expected_gate_name_inventory: Sequence[str],
    receipts_by_stage: Mapping[str, Mapping[str, Any]],
    gate_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Authorize only the exact preregistered passing evidence closure."""

    prereg, receipt_bindings, gate_bindings = _authorization_inputs(
        preregistration,
        receipts_by_stage,
        gate_reports,
        expected_preregistration_sha256=expected_preregistration_sha256,
        expected_receipt_stage_inventory=expected_receipt_stage_inventory,
        expected_gate_name_inventory=expected_gate_name_inventory,
    )
    return seal_record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "v1_dependency_manifest_sha256": V1_DEPENDENCY_MANIFEST_SHA256,
            "campaign_sha256": prereg["campaign_sha256"],
            "preregistration_sha256": prereg["preregistration_sha256"],
            "passed": True,
            "receipts": receipt_bindings,
            "gates": gate_bindings,
        },
        "authorization_sha256",
    )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    expected_preregistration_sha256: str,
    expected_receipt_stage_inventory: Sequence[str],
    expected_gate_name_inventory: Sequence[str],
    receipts_by_stage: Mapping[str, Mapping[str, Any]],
    gate_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    verified = verify_sealed_record(authorization, "authorization_sha256")
    required = {
        "schema",
        "v1_dependency_manifest_sha256",
        "campaign_sha256",
        "preregistration_sha256",
        "passed",
        "receipts",
        "gates",
        "authorization_sha256",
    }
    if (
        set(verified) != required
        or verified.get("schema") != AUTHORIZATION_SCHEMA
        or verified.get("v1_dependency_manifest_sha256")
        != V1_DEPENDENCY_MANIFEST_SHA256
        or verified.get("passed") is not True
    ):
        raise AuthorizationError("authorization has an invalid schema or status")
    rebuilt = build_authorization(
        preregistration=preregistration,
        expected_preregistration_sha256=expected_preregistration_sha256,
        expected_receipt_stage_inventory=expected_receipt_stage_inventory,
        expected_gate_name_inventory=expected_gate_name_inventory,
        receipts_by_stage=receipts_by_stage,
        gate_reports=gate_reports,
    )
    if rebuilt != verified:
        raise AuthorizationError("authorization does not bind the supplied closure")
    return verified


def _absolute(path: os.PathLike[str] | str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except TypeError as exc:
        raise EvidenceError("path must be path-like") from exc


def _stat_record(info: os.stat_result, fields: tuple[str, ...]) -> dict[str, int]:
    return {field[3:]: int(getattr(info, field)) for field in fields}


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"directory cannot be opened safely: {path}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise EvidenceError("opened path is not a directory")
    try:
        live = path.lstat()
    except OSError as exc:
        os.close(descriptor)
        raise EvidenceError("directory path is no longer live") from exc
    if (
        stat.S_ISLNK(live.st_mode)
        or not stat.S_ISDIR(live.st_mode)
        or (live.st_dev, live.st_ino) != (info.st_dev, info.st_ino)
    ):
        os.close(descriptor)
        raise EvidenceError("directory path does not name the opened directory")
    return descriptor, info


def _scan_directory(descriptor: int) -> dict[str, os.stat_result]:
    result: dict[str, os.stat_result] = {}
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if type(entry.name) is not str or entry.name in result:
                    raise EvidenceError("directory contains an invalid duplicate name")
                result[entry.name] = entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise EvidenceError("directory cannot be scanned safely") from exc
    return result


def _hash_direct_file(
    directory_fd: int,
    name: str,
    *,
    include_bytes: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    if (
        type(name) is not str
        or _SAFE_FILENAME_RE.fullmatch(name) is None
        or "/" in name
        or "\\" in name
    ):
        raise EvidenceError("file name is not a safe direct-child name")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise EvidenceError(f"file cannot be opened safely: {name}") from exc
    chunks: list[bytes] | None = [] if include_bytes else None
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise EvidenceError("file must be regular and have exactly one hard link")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if any(
        getattr(before, field) != getattr(after, field) for field in _FILE_FIELDS
    ):
        raise EvidenceError("file identity changed while it was read")
    try:
        live = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceError("file path disappeared after reading") from exc
    if (
        not stat.S_ISREG(live.st_mode)
        or live.st_nlink != 1
        or any(
            getattr(live, field) != getattr(after, field)
            for field in _FILE_FIELDS
        )
    ):
        raise EvidenceError("file path no longer names the bytes that were read")
    if size != after.st_size:
        raise EvidenceError("read byte count does not match file size")
    binding = {
        "schema": FILE_BINDING_SCHEMA,
        "path": name,
        "sha256": digest.hexdigest(),
        **_stat_record(after, _FILE_FIELDS),
    }
    return binding, None if chunks is None else b"".join(chunks)


def _validate_seed_sequence(seeds: Any) -> list[int]:
    if type(seeds) not in (list, tuple) or not seeds:
        raise EvidenceError("expected_seeds must be a non-empty exact sequence")
    result: list[int] = []
    for seed in seeds:
        _require_exact(seed, int, "seed")
        if seed < 0 or seed in result:
            raise EvidenceError("seeds must be unique non-negative integers")
        result.append(seed)
    return sorted(result)


def _collection_layout(
    seeds: list[int], layout: Mapping[str, str] | None
) -> tuple[list[str], dict[str, tuple[int, str]]]:
    if layout is None:
        role_layout = {
            "episode": "{seed}.episode.json",
            "queries": "{seed}.queries.jsonl",
        }
    else:
        _require_exact(layout, dict, "layout")
        role_layout = dict(layout)
    if not role_layout:
        raise EvidenceError("layout cannot be empty")
    expected: dict[str, tuple[int, str]] = {}
    for role in sorted(role_layout):
        _require_token(role, "collection role")
        template = role_layout[role]
        _require_exact(template, str, f"layout.{role}")
        if template.count("{seed}") != 1:
            raise EvidenceError("layout templates need exactly one {seed}")
        for seed in seeds:
            name = template.format(seed=seed)
            if (
                _SAFE_FILENAME_RE.fullmatch(name) is None
                or "/" in name
                or "\\" in name
                or name in expected
            ):
                raise EvidenceError("layout produces an unsafe or duplicate name")
            expected[name] = (seed, role)
    return sorted(role_layout), expected


def build_collection_closure(
    root: os.PathLike[str] | str,
    expected_seeds: Sequence[int],
    *,
    layout: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Hash an exact collection and re-scan its directory before sealing."""

    dependency_sha256 = _bind_dependency()
    collection_root = _absolute(root)
    seeds = _validate_seed_sequence(expected_seeds)
    roles, expected = _collection_layout(seeds, layout)
    directory_fd, directory_before = _open_directory(collection_root)
    try:
        initial = _scan_directory(directory_fd)
        if set(initial) != set(expected):
            raise EvidenceError("initial collection names do not exactly close")
        for name in expected:
            info = initial[name]
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise EvidenceError("every expected entry must be a single-link file")

        files: list[dict[str, Any]] = []
        by_name: dict[str, dict[str, Any]] = {}
        for name in sorted(expected):
            seed, role = expected[name]
            binding, _ = _hash_direct_file(directory_fd, name)
            by_name[name] = binding
            files.append({"seed": seed, "role": role, "binding": binding})

        # Resolve the lexical path before the final descriptor-relative boundary.
        # No path lookup is permitted after the scan/stat closure below.
        try:
            live = collection_root.lstat()
        except OSError as exc:
            raise EvidenceError("collection path disappeared during closure") from exc
        if (
            stat.S_ISLNK(live.st_mode)
            or not stat.S_ISDIR(live.st_mode)
            or (live.st_dev, live.st_ino)
            != (directory_before.st_dev, directory_before.st_ino)
        ):
            raise EvidenceError("collection path was swapped during closure")

        final = _scan_directory(directory_fd)
        if set(final) != set(expected):
            raise EvidenceError("final collection names do not exactly close")
        for name in sorted(expected):
            info = final[name]
            binding = by_name[name]
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or any(
                    int(getattr(info, field)) != binding[field[3:]]
                    for field in _FILE_FIELDS
                )
            ):
                raise EvidenceError("final collection entry replaced or changed")
        directory_after = os.fstat(directory_fd)
        if any(
            getattr(directory_before, field) != getattr(directory_after, field)
            for field in _DIRECTORY_FIELDS
        ):
            raise EvidenceError("collection directory changed during closure")
    finally:
        os.close(directory_fd)

    directory_before_binding = _stat_record(directory_before, _DIRECTORY_FIELDS)
    directory_after_binding = _stat_record(directory_after, _DIRECTORY_FIELDS)
    return seal_record(
        {
            "schema": COLLECTION_CLOSURE_SCHEMA,
            "v1_dependency_manifest_sha256": dependency_sha256,
            "directory": {
                "path": str(collection_root),
                "before": directory_before_binding,
                "after": directory_after_binding,
            },
            "seeds": seeds,
            "roles": roles,
            "files": files,
        },
        "closure_sha256",
    )


def validate_collection_closure(
    closure: Mapping[str, Any],
    *,
    root: os.PathLike[str] | str,
    expected_seeds: Sequence[int],
    layout: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verified = verify_sealed_record(closure, "closure_sha256")
    required = {
        "schema",
        "v1_dependency_manifest_sha256",
        "directory",
        "seeds",
        "roles",
        "files",
        "closure_sha256",
    }
    if (
        set(verified) != required
        or verified.get("schema") != COLLECTION_CLOSURE_SCHEMA
        or verified.get("v1_dependency_manifest_sha256")
        != V1_DEPENDENCY_MANIFEST_SHA256
    ):
        raise EvidenceError("collection closure has an invalid schema")
    rebuilt = build_collection_closure(root, expected_seeds, layout=layout)
    if rebuilt != verified:
        raise EvidenceError("collection closure does not match the live collection")
    return verified


def _direct_file_binding(
    path: os.PathLike[str] | str,
    *,
    root: os.PathLike[str] | str,
    include_bytes: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    target = _absolute(path)
    binding_root = _absolute(root)
    if target.parent != binding_root:
        raise EvidenceError("bound file must be a direct child of its supplied root")
    directory_fd, directory_before = _open_directory(binding_root)
    try:
        binding, data = _hash_direct_file(
            directory_fd, target.name, include_bytes=include_bytes
        )
        # Check the lexical root before the last descriptor-relative target and
        # directory stats.  This makes a swap during lstat observable below.
        live_root = binding_root.lstat()
        if (
            stat.S_ISLNK(live_root.st_mode)
            or not stat.S_ISDIR(live_root.st_mode)
            or (live_root.st_dev, live_root.st_ino)
            != (directory_before.st_dev, directory_before.st_ino)
        ):
            raise EvidenceError("file root path was swapped")
        try:
            final_file = os.stat(
                target.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except OSError as exc:
            raise EvidenceError("bound file path disappeared") from exc
        if any(
            int(getattr(final_file, field)) != binding[field[3:]]
            for field in _FILE_FIELDS
        ):
            raise EvidenceError("bound file path was replaced after reading")
        directory_after = os.fstat(directory_fd)
        if any(
            getattr(directory_before, field) != getattr(directory_after, field)
            for field in _DIRECTORY_FIELDS
        ):
            raise EvidenceError("file root changed during validation")
    finally:
        os.close(directory_fd)
    return binding, data


def _final_checkpoint_bindings(
    *,
    envelope_path: os.PathLike[str] | str,
    evidence_root: os.PathLike[str] | str,
    model_path: os.PathLike[str] | str,
    model_root: os.PathLike[str] | str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Rebind both paths, then end on checked descriptor-relative stats."""

    envelope_target = _absolute(envelope_path)
    evidence_directory = _absolute(evidence_root)
    model_target = _absolute(model_path)
    model_directory = _absolute(model_root)
    if (
        envelope_target.parent != evidence_directory
        or model_target.parent != model_directory
    ):
        raise EvidenceError("checkpoint files must be direct children of their roots")

    evidence_fd, evidence_before = _open_directory(evidence_directory)
    try:
        model_fd, model_before = _open_directory(model_directory)
        try:
            envelope_binding, envelope_data = _direct_file_binding(
                envelope_target, root=evidence_directory, include_bytes=True
            )
            model_binding, _ = _direct_file_binding(
                model_target, root=model_directory
            )
            assert envelope_data is not None

            # Every lexical check above precedes this final descriptor-relative
            # pair closure.  Nothing path-based is consulted after these stats.
            try:
                envelope_final = os.stat(
                    envelope_target.name,
                    dir_fd=evidence_fd,
                    follow_symlinks=False,
                )
                model_final = os.stat(
                    model_target.name,
                    dir_fd=model_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise EvidenceError(
                    "checkpoint path disappeared at the final boundary"
                ) from exc
            if any(
                int(getattr(envelope_final, field))
                != envelope_binding[field[3:]]
                for field in _FILE_FIELDS
            ):
                raise EvidenceError("envelope changed at the final load boundary")
            if any(
                int(getattr(model_final, field)) != model_binding[field[3:]]
                for field in _FILE_FIELDS
            ):
                raise EvidenceError("model changed at the final load boundary")
            evidence_after = os.fstat(evidence_fd)
            model_after = os.fstat(model_fd)
            if any(
                getattr(evidence_before, field) != getattr(evidence_after, field)
                for field in _DIRECTORY_FIELDS
            ):
                raise EvidenceError("evidence root changed at the final boundary")
            if any(
                getattr(model_before, field) != getattr(model_after, field)
                for field in _DIRECTORY_FIELDS
            ):
                raise EvidenceError("model root changed at the final boundary")
        finally:
            os.close(model_fd)
    finally:
        os.close(evidence_fd)
    return envelope_binding, envelope_data, model_binding


def build_checkpoint_envelope(
    *,
    model_path: os.PathLike[str] | str,
    model_root: os.PathLike[str] | str,
    metadata: Mapping[str, Any],
    partition_sha256: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    dependency_sha256 = _bind_dependency()
    metadata_value = _canonical_dict_copy(metadata, "metadata")
    partition_sha256 = _require_sha256(partition_sha256, "partition_sha256")
    dataset_sha256 = _require_sha256(dataset_sha256, "dataset_sha256")
    model_binding, _ = _direct_file_binding(model_path, root=model_root)
    return seal_record(
        {
            "schema": CHECKPOINT_ENVELOPE_SCHEMA,
            "v1_dependency_manifest_sha256": dependency_sha256,
            "model": model_binding,
            "model_sha256": model_binding["sha256"],
            "metadata": metadata_value,
            "metadata_sha256": sha256_json(metadata_value),
            "partition_sha256": partition_sha256,
            "dataset_sha256": dataset_sha256,
        },
        "envelope_sha256",
    )


def _decode_canonical_record(data: bytes) -> dict[str, Any]:
    if not data.endswith(b"\n"):
        raise EvidenceError("evidence JSON must have one canonical trailing newline")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError("evidence JSON contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvidenceError(f"evidence JSON contains non-finite constant {value}")

    try:
        decoded = json.loads(
            data[:-1].decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("evidence file is not canonical UTF-8 JSON") from exc
    _require_exact(decoded, dict, "checkpoint envelope")
    if canonical_json_bytes(decoded) + b"\n" != data:
        raise EvidenceError("evidence file does not use canonical JSON bytes")
    return decoded


def load_checkpoint_envelope(
    envelope_path: os.PathLike[str] | str,
    *,
    evidence_root: os.PathLike[str] | str,
    model_root: os.PathLike[str] | str,
    expected_envelope_sha256: str,
    expected_model_sha256: str,
    expected_metadata: Mapping[str, Any],
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
) -> dict[str, Any]:
    """Load an exact checkpoint envelope and rebind its live model file."""

    _bind_dependency()
    expected_envelope_sha256 = _require_sha256(
        expected_envelope_sha256, "expected_envelope_sha256"
    )
    expected_model_sha256 = _require_sha256(
        expected_model_sha256, "expected_model_sha256"
    )
    expected_partition_sha256 = _require_sha256(
        expected_partition_sha256, "expected_partition_sha256"
    )
    expected_dataset_sha256 = _require_sha256(
        expected_dataset_sha256, "expected_dataset_sha256"
    )
    expected_metadata_value = _canonical_dict_copy(
        expected_metadata, "expected_metadata"
    )

    envelope_binding, data = _direct_file_binding(
        envelope_path, root=evidence_root, include_bytes=True
    )
    assert data is not None
    verified = verify_sealed_record(
        _decode_canonical_record(data), "envelope_sha256"
    )
    required = {
        "schema",
        "v1_dependency_manifest_sha256",
        "model",
        "model_sha256",
        "metadata",
        "metadata_sha256",
        "partition_sha256",
        "dataset_sha256",
        "envelope_sha256",
    }
    if (
        set(verified) != required
        or verified.get("schema") != CHECKPOINT_ENVELOPE_SCHEMA
        or verified.get("v1_dependency_manifest_sha256")
        != V1_DEPENDENCY_MANIFEST_SHA256
    ):
        raise EvidenceError("checkpoint envelope has an invalid schema")
    for field in (
        "model_sha256",
        "metadata_sha256",
        "partition_sha256",
        "dataset_sha256",
        "envelope_sha256",
    ):
        _require_sha256(verified[field], field)
    if verified["envelope_sha256"] != expected_envelope_sha256:
        raise EvidenceError("checkpoint envelope identity mismatch")
    if verified["model_sha256"] != expected_model_sha256:
        raise EvidenceError("checkpoint model identity mismatch")
    if verified["partition_sha256"] != expected_partition_sha256:
        raise EvidenceError("checkpoint partition identity mismatch")
    if verified["dataset_sha256"] != expected_dataset_sha256:
        raise EvidenceError("checkpoint dataset identity mismatch")
    _require_exact(verified["metadata"], dict, "metadata")
    if (
        sha256_json(verified["metadata"]) != verified["metadata_sha256"]
        or canonical_json_bytes(verified["metadata"])
        != canonical_json_bytes(expected_metadata_value)
    ):
        raise EvidenceError("checkpoint metadata identity mismatch")

    _require_exact(verified["model"], dict, "model")
    required_binding = {
        "schema",
        "path",
        "sha256",
        "dev",
        "ino",
        "mode",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    if (
        set(verified["model"]) != required_binding
        or verified["model"].get("schema") != FILE_BINDING_SCHEMA
    ):
        raise EvidenceError("checkpoint model binding has an invalid schema")
    name = verified["model"].get("path")
    if (
        type(name) is not str
        or _SAFE_FILENAME_RE.fullmatch(name) is None
        or "/" in name
        or "\\" in name
    ):
        raise EvidenceError("checkpoint model path is unsafe")
    _require_sha256(verified["model"]["sha256"], "model.sha256")
    for field in (
        "dev",
        "ino",
        "mode",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    ):
        _require_exact(verified["model"][field], int, f"model.{field}")
        if verified["model"][field] < 0:
            raise EvidenceError(f"model.{field} must be non-negative")
    # Rebind the envelope after all semantic validation, then bind the model as
    # the final filesystem boundary.  No path operation occurs before return.
    final_envelope_binding, final_data, live_binding = _final_checkpoint_bindings(
        envelope_path=envelope_path,
        evidence_root=evidence_root,
        model_path=_absolute(model_root) / name,
        model_root=model_root,
    )
    if (
        final_envelope_binding != envelope_binding
        or final_data != data
    ):
        raise EvidenceError("checkpoint envelope path changed during load")
    if live_binding != verified["model"]:
        raise EvidenceError("live checkpoint file no longer matches its envelope")
    if live_binding["sha256"] != verified["model_sha256"]:
        raise EvidenceError("model binding and model hash disagree")
    return verified


def write_checkpoint_envelope(
    envelope_path: os.PathLike[str] | str,
    *,
    evidence_root: os.PathLike[str] | str,
    model_path: os.PathLike[str] | str,
    model_root: os.PathLike[str] | str,
    metadata: Mapping[str, Any],
    partition_sha256: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Build and atomically publish a checkpoint envelope exactly once."""

    target = _absolute(envelope_path)
    if target.parent != _absolute(evidence_root):
        raise EvidenceError("envelope must be a direct child of evidence_root")
    envelope = build_checkpoint_envelope(
        model_path=model_path,
        model_root=model_root,
        metadata=metadata,
        partition_sha256=partition_sha256,
        dataset_sha256=dataset_sha256,
    )
    publication = write_once_atomic_json(target, envelope)
    published_binding, published_data = _direct_file_binding(
        target, root=evidence_root, include_bytes=True
    )
    if (
        published_data != canonical_json_bytes(envelope) + b"\n"
        or publication.get("sha256") != published_binding["sha256"]
        or publication.get("size") != published_binding["size"]
        or publication.get("device") != published_binding["dev"]
        or publication.get("inode") != published_binding["ino"]
        or publication.get("mtime_ns") != published_binding["mtime_ns"]
        or publication.get("mode") != stat.S_IMODE(published_binding["mode"])
    ):
        raise EvidenceError("published envelope path no longer names its publication")
    rebound = load_checkpoint_envelope(
        target,
        evidence_root=evidence_root,
        model_root=model_root,
        expected_envelope_sha256=envelope["envelope_sha256"],
        expected_model_sha256=envelope["model_sha256"],
        expected_metadata=envelope["metadata"],
        expected_partition_sha256=envelope["partition_sha256"],
        expected_dataset_sha256=envelope["dataset_sha256"],
    )
    if rebound != envelope:
        raise EvidenceError("published checkpoint envelope failed final rebinding")
    return {"envelope": rebound, "publication": publication}


__all__ = [
    "AUTHORIZATION_PREREGISTRATION_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "AuthorizationError",
    "CHECKPOINT_ENVELOPE_SCHEMA",
    "COLLECTION_CLOSURE_SCHEMA",
    "DEPENDENCY_MANIFEST_SCHEMA",
    "EvidenceError",
    "FILE_BINDING_SCHEMA",
    "GATE_REPORT_SCHEMA",
    "V1_DEPENDENCY_MANIFEST_SHA256",
    "V1_EVIDENCE_SOURCE_SHA256",
    "build_authorization",
    "build_authorization_preregistration",
    "build_checkpoint_envelope",
    "build_collection_closure",
    "build_gate_report",
    "dependency_manifest",
    "load_checkpoint_envelope",
    "validate_authorization",
    "validate_authorization_preregistration",
    "validate_collection_closure",
    "write_checkpoint_envelope",
]
