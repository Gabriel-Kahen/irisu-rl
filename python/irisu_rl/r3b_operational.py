"""Durable, fail-closed orchestration state for long-running R3b trials."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time_ns
from typing import Mapping, Sequence

from .r3b_experiments import (
    R3BExperimentPlan,
    SealedTestRunAuthorization,
    TrialJob,
    ValidationRunAuthorization,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_ZERO = "0" * 64
CANONICAL_PLAN_SHA256 = (
    "68860ef26686c954960c176afe67a44da34e2ffab03dd02ba5aa7c1fc193baf8"
)
CANONICAL_OPERATIONAL_CONFIG_SHA256 = (
    "b59828dfcf0bf933ba940ad8f219765784e8328cde8a5ca39b09411d2a4d275c"
)
CANONICAL_EXACT_SNAPSHOT_BUNDLE_SHA256 = (
    "2371129c883ade88b309e509c2d8a7399a85944dc5dd7e841fc14d977527eb7a"
)
CANONICAL_PORTABLE_SNAPSHOT_BUNDLE_SHA256 = (
    "e8ee8ae85037a9db86e8a60cc5274e4be35414958317fce1b09dd36bed713354"
)
CANONICAL_PAIRING_SHA256S = {
    "calibration": "bc669a86d398d1f94228de29477b59b2e3228f7e8b60ea093ebbe0ac9001b7cf",
    "validation": "3cff35fa8cda87cffa04cf5945f1af2470459b8d764295eb62a1d2963229d0fb",
    "test": "058040f869e445487f558e115eb51881ae21fc88b8577f28f97d0434f41b712e",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == _SHA256_ZERO
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _require_keys(
    value: object, expected: set[str], *, label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed table")
    if set(value) != expected:
        raise ValueError(f"{label} keys differ from the frozen operational schema")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class R3BOperationalConfig:
    """Choices that the experiment plan intentionally does not contain."""

    lanes: int
    workers: int
    torch_threads: int
    max_consecutive_skips: int
    collector_max_decisions: int
    collector_lambda_tick: float
    model_global_hidden: int
    model_body_hidden: int
    model_fused_hidden: int
    model_recurrent_hidden: int
    model_recurrent_layers: int
    ppo_epochs: int
    ppo_lane_minibatch_size: int
    ppo_clip_ratio: float
    ppo_value_clip: float
    ppo_value_coefficient: float
    ppo_entropy_coefficient: float
    ppo_max_gradient_norm: float
    ppo_target_kl: float
    curve_snapshots: int
    evaluation_shards: int
    calibration_repetitions: int
    validation_repetitions: int
    test_repetitions: int
    evaluation_max_decisions: int
    evaluation_max_simulated_ticks: int
    snapshot_generator_version: str
    minimum_train_snapshots: int
    minimum_calibration_snapshots: int
    minimum_validation_snapshots: int
    minimum_test_snapshots: int
    checkpoint_retention: str
    primary_backend: str
    transfer_eligible: bool
    version: str = "r3b-operational-config-v1"

    def __post_init__(self) -> None:
        positive = (
            self.lanes,
            self.workers,
            self.torch_threads,
            self.max_consecutive_skips,
            self.collector_max_decisions,
            self.model_global_hidden,
            self.model_body_hidden,
            self.model_fused_hidden,
            self.model_recurrent_hidden,
            self.model_recurrent_layers,
            self.ppo_epochs,
            self.ppo_lane_minibatch_size,
            self.curve_snapshots,
            self.evaluation_shards,
            self.calibration_repetitions,
            self.validation_repetitions,
            self.test_repetitions,
            self.evaluation_max_decisions,
            self.evaluation_max_simulated_ticks,
            self.minimum_train_snapshots,
            self.minimum_calibration_snapshots,
            self.minimum_validation_snapshots,
            self.minimum_test_snapshots,
        )
        finite_positive = (
            self.collector_lambda_tick,
            self.ppo_clip_ratio,
            self.ppo_value_clip,
            self.ppo_value_coefficient,
            self.ppo_max_gradient_norm,
            self.ppo_target_kl,
        )
        if (
            self.version != "r3b-operational-config-v1"
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in positive
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) < float("inf")
                for value in finite_positive
            )
            or isinstance(self.ppo_entropy_coefficient, bool)
            or not isinstance(self.ppo_entropy_coefficient, (int, float))
            or not 0 <= float(self.ppo_entropy_coefficient) < float("inf")
            or not 0 < self.collector_lambda_tick <= 1
            or self.workers > self.lanes
            or self.lanes % self.ppo_lane_minibatch_size
            or self.curve_snapshots
            > min(
                self.minimum_calibration_snapshots,
                self.minimum_validation_snapshots,
                self.minimum_test_snapshots,
            )
            or self.evaluation_shards > self.curve_snapshots
            or self.evaluation_max_decisions < self.evaluation_max_simulated_ticks
            or self.primary_backend != "exact"
            or self.checkpoint_retention != "all_planned_boundaries"
            or self.transfer_eligible
            or not _SAFE_ID.fullmatch(self.snapshot_generator_version)
        ):
            raise ValueError("R3b operational configuration is invalid")

    @classmethod
    def from_mapping(cls, value: object) -> R3BOperationalConfig:
        root = _require_keys(
            value,
            {
                "version",
                "runtime",
                "collector",
                "model",
                "ppo",
                "evaluation",
                "snapshots",
                "artifacts",
            },
            label="root",
        )
        runtime = _require_keys(
            root["runtime"],
            {
                "lanes",
                "workers",
                "torch_threads",
                "max_consecutive_skips",
                "primary_backend",
                "transfer_eligible",
            },
            label="runtime",
        )
        collector = _require_keys(
            root["collector"], {"max_decisions", "lambda_tick"}, label="collector"
        )
        model = _require_keys(
            root["model"],
            {
                "global_hidden",
                "body_hidden",
                "fused_hidden",
                "recurrent_hidden",
                "recurrent_layers",
            },
            label="model",
        )
        ppo = _require_keys(
            root["ppo"],
            {
                "epochs",
                "lane_minibatch_size",
                "clip_ratio",
                "value_clip",
                "value_coefficient",
                "entropy_coefficient",
                "max_gradient_norm",
                "target_kl",
            },
            label="ppo",
        )
        evaluation = _require_keys(
            root["evaluation"],
            {
                "curve_snapshots",
                "shards",
                "calibration_repetitions",
                "validation_repetitions",
                "test_repetitions",
                "max_decisions",
                "max_simulated_ticks",
            },
            label="evaluation",
        )
        snapshots = _require_keys(
            root["snapshots"],
            {
                "generator_version",
                "minimum_train",
                "minimum_calibration",
                "minimum_validation",
                "minimum_test",
            },
            label="snapshots",
        )
        artifacts = _require_keys(
            root["artifacts"], {"checkpoint_retention"}, label="artifacts"
        )
        try:
            return cls(
                lanes=runtime["lanes"],  # type: ignore[arg-type]
                workers=runtime["workers"],  # type: ignore[arg-type]
                torch_threads=runtime["torch_threads"],  # type: ignore[arg-type]
                max_consecutive_skips=runtime["max_consecutive_skips"],  # type: ignore[arg-type]
                collector_max_decisions=collector["max_decisions"],  # type: ignore[arg-type]
                collector_lambda_tick=collector["lambda_tick"],  # type: ignore[arg-type]
                model_global_hidden=model["global_hidden"],  # type: ignore[arg-type]
                model_body_hidden=model["body_hidden"],  # type: ignore[arg-type]
                model_fused_hidden=model["fused_hidden"],  # type: ignore[arg-type]
                model_recurrent_hidden=model["recurrent_hidden"],  # type: ignore[arg-type]
                model_recurrent_layers=model["recurrent_layers"],  # type: ignore[arg-type]
                ppo_epochs=ppo["epochs"],  # type: ignore[arg-type]
                ppo_lane_minibatch_size=ppo["lane_minibatch_size"],  # type: ignore[arg-type]
                ppo_clip_ratio=ppo["clip_ratio"],  # type: ignore[arg-type]
                ppo_value_clip=ppo["value_clip"],  # type: ignore[arg-type]
                ppo_value_coefficient=ppo["value_coefficient"],  # type: ignore[arg-type]
                ppo_entropy_coefficient=ppo["entropy_coefficient"],  # type: ignore[arg-type]
                ppo_max_gradient_norm=ppo["max_gradient_norm"],  # type: ignore[arg-type]
                ppo_target_kl=ppo["target_kl"],  # type: ignore[arg-type]
                curve_snapshots=evaluation["curve_snapshots"],  # type: ignore[arg-type]
                evaluation_shards=evaluation["shards"],  # type: ignore[arg-type]
                calibration_repetitions=evaluation["calibration_repetitions"],  # type: ignore[arg-type]
                validation_repetitions=evaluation["validation_repetitions"],  # type: ignore[arg-type]
                test_repetitions=evaluation["test_repetitions"],  # type: ignore[arg-type]
                evaluation_max_decisions=evaluation["max_decisions"],  # type: ignore[arg-type]
                evaluation_max_simulated_ticks=evaluation["max_simulated_ticks"],  # type: ignore[arg-type]
                snapshot_generator_version=snapshots["generator_version"],  # type: ignore[arg-type]
                minimum_train_snapshots=snapshots["minimum_train"],  # type: ignore[arg-type]
                minimum_calibration_snapshots=snapshots["minimum_calibration"],  # type: ignore[arg-type]
                minimum_validation_snapshots=snapshots["minimum_validation"],  # type: ignore[arg-type]
                minimum_test_snapshots=snapshots["minimum_test"],  # type: ignore[arg-type]
                checkpoint_retention=artifacts["checkpoint_retention"],  # type: ignore[arg-type]
                primary_backend=runtime["primary_backend"],  # type: ignore[arg-type]
                transfer_eligible=runtime["transfer_eligible"],  # type: ignore[arg-type]
                version=root["version"],  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise ValueError("operational configuration types are malformed") from exc

    @classmethod
    def from_toml(cls, path: str | Path) -> R3BOperationalConfig:
        supplied = Path(path)
        if supplied.is_symlink() or not supplied.is_file():
            raise ValueError("operational configuration must be a regular TOML file")
        with supplied.open("rb") as handle:
            return cls.from_mapping(tomllib.load(handle))

    @classmethod
    def from_manifest(cls, value: object) -> R3BOperationalConfig:
        expected = set(cls.__dataclass_fields__)
        manifest = _require_keys(value, expected, label="operational manifest")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("operational manifest types are malformed") from exc
        if result.manifest() != dict(manifest):
            raise ValueError("operational manifest is noncanonical")
        return result

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class JobClaim:
    job_sha256: str
    phase: str
    token: str
    owner: str
    resume_from_update: int
    resume_checkpoint_sha256: str | None

    def __post_init__(self) -> None:
        if (
            _require_sha256(self.job_sha256, "job identity") != self.job_sha256
            or self.phase not in {"calibration", "validation", "test"}
            or not isinstance(self.token, str)
            or len(self.token) != 64
            or self.token == _SHA256_ZERO
            or any(character not in "0123456789abcdef" for character in self.token)
            or not isinstance(self.owner, str)
            or _SAFE_ID.fullmatch(self.owner) is None
            or _nonnegative_int(self.resume_from_update, "resume update")
            != self.resume_from_update
            or (self.resume_checkpoint_sha256 is None and self.resume_from_update != 0)
            or (
                self.resume_checkpoint_sha256 is not None
                and (
                    self.resume_from_update == 0
                    or _require_sha256(
                        self.resume_checkpoint_sha256, "resume checkpoint"
                    )
                    != self.resume_checkpoint_sha256
                )
            )
        ):
            raise ValueError("job claim is malformed")


class R3BWorkflow:
    """SQLite state machine; model artifacts remain in a content-addressed store."""

    schema_version = "r3b-workflow-v1"

    def __init__(self, path: str | Path) -> None:
        supplied = Path(path)
        if not supplied.is_absolute():
            supplied = supplied.absolute()
        current = Path(supplied.anchor)
        for component in supplied.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError("workflow database path crosses a symlink")
        if supplied.is_symlink():
            raise ValueError("workflow database path must not be a symlink")
        self.path = supplied

    @staticmethod
    def calibration_jobs(plan: R3BExperimentPlan) -> tuple[TrialJob, ...]:
        """Use one progressive job; the first budget is an evidence rung."""

        final_budget = plan.calibration_budgets_updates[-1]
        jobs = tuple(
            job
            for job in plan.trial_jobs("calibration")
            if job.budget_updates == final_budget
        )
        expected = len(plan.arms) * len(plan.calibration_learner_seeds)
        if len(jobs) != expected:
            raise ValueError("frozen calibration jobs cannot form progressive trials")
        return jobs

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        run_id: str,
        run_class: str,
        plan: R3BExperimentPlan,
        config: R3BOperationalConfig,
        snapshot_bundle_sha256: str,
        portable_snapshot_bundle_sha256: str | None = None,
        pairing_sha256s: Mapping[str, str] | None = None,
        source_identity_sha256: str,
        source_revision: str | None = None,
    ) -> R3BWorkflow:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run id is unsafe")
        if run_class not in {"smoke", "canonical"}:
            raise ValueError("run class must be smoke or canonical")
        bundle_sha = _require_sha256(snapshot_bundle_sha256, "snapshot bundle identity")
        portable_bundle_sha = (
            None
            if portable_snapshot_bundle_sha256 is None
            else _require_sha256(
                portable_snapshot_bundle_sha256,
                "portable snapshot bundle identity",
            )
        )
        pairings = None if pairing_sha256s is None else dict(pairing_sha256s)
        if pairings is not None and (
            set(pairings) != set(CANONICAL_PAIRING_SHA256S)
            or any(
                _require_sha256(value, f"{split} pairing") != value
                for split, value in pairings.items()
            )
        ):
            raise ValueError("snapshot pairing identities are malformed")
        if run_class == "canonical" and (
            plan.sha256 != CANONICAL_PLAN_SHA256
            or config.sha256 != CANONICAL_OPERATIONAL_CONFIG_SHA256
            or bundle_sha != CANONICAL_EXACT_SNAPSHOT_BUNDLE_SHA256
            or portable_bundle_sha != CANONICAL_PORTABLE_SNAPSHOT_BUNDLE_SHA256
            or pairings != CANONICAL_PAIRING_SHA256S
            or not isinstance(source_revision, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", source_revision) is None
        ):
            raise ValueError("canonical run inputs differ from the preregistered lock")
        if run_class == "smoke" and (
            portable_bundle_sha is not None
            or pairings is not None
            or source_revision is not None
        ):
            raise ValueError("smoke runs accept only one exact diagnostic bundle")
        source_sha = _require_sha256(source_identity_sha256, "source identity")
        workflow = cls(path)
        if workflow.path.exists():
            raise FileExistsError("workflow database already exists")
        workflow.path.parent.mkdir(parents=True, exist_ok=True)
        if workflow.path.parent.is_symlink():
            raise ValueError("workflow parent must not be a symlink")
        connection = workflow._connect(create=True)
        try:
            workflow._create_schema(connection)
            now = time_ns()
            manifest = {
                "version": cls.schema_version,
                "run_id": run_id,
                "run_class": run_class,
                "plan_sha256": plan.sha256,
                "operational_config_sha256": config.sha256,
                "snapshot_bundle_sha256": bundle_sha,
                "portable_snapshot_bundle_sha256": portable_bundle_sha,
                "pairing_sha256s": pairings,
                "source_identity_sha256": source_sha,
                "source_revision": source_revision,
                "acceptance_eligible": run_class == "canonical",
                "transfer_eligible": False,
            }
            with connection:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (
                        (
                            "schema_version",
                            _canonical_bytes(cls.schema_version).decode(),
                        ),
                        ("manifest", _canonical_bytes(manifest).decode()),
                        ("plan", _canonical_bytes(plan.manifest()).decode()),
                        (
                            "operational_config",
                            _canonical_bytes(config.manifest()).decode(),
                        ),
                    ),
                )
                for job in cls.calibration_jobs(plan):
                    connection.execute(
                        """
                        INSERT INTO jobs(
                            job_sha256, phase, sealed, budget_updates, status,
                            manifest_json, created_ns
                        ) VALUES (?, 'calibration', 0, ?, 'pending', ?, ?)
                        """,
                        (
                            job.sha256,
                            job.budget_updates,
                            _canonical_bytes(job.manifest()).decode(),
                            now,
                        ),
                    )
                connection.execute(
                    "INSERT INTO events(at_ns, kind, detail_json) VALUES (?, ?, ?)",
                    (now, "run_created", _canonical_bytes(manifest).decode()),
                )
        except BaseException:
            connection.close()
            try:
                workflow.path.unlink()
            except FileNotFoundError:
                pass
            raise
        connection.close()
        os.chmod(workflow.path, 0o600)
        workflow.verify()
        return workflow

    def _connect(self, *, create: bool = False) -> sqlite3.Connection:
        if not create:
            try:
                metadata = self.path.lstat()
            except FileNotFoundError as exc:
                raise ValueError("workflow database is missing or unsafe") from exc
            if (
                self.path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise ValueError("workflow database is missing or unsafe")
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE metadata(
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE jobs(
                job_sha256 TEXT PRIMARY KEY NOT NULL,
                phase TEXT NOT NULL CHECK(phase IN ('calibration','validation','test')),
                sealed INTEGER NOT NULL CHECK(sealed IN (0,1)),
                budget_updates INTEGER NOT NULL CHECK(budget_updates > 0),
                status TEXT NOT NULL CHECK(
                    status IN (
                        'pending','claimed','running','trained','completed','failed'
                    )
                ),
                manifest_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                owner TEXT,
                token_sha256 TEXT,
                claimed_ns INTEGER,
                started_ns INTEGER,
                finished_ns INTEGER,
                resume_count INTEGER NOT NULL DEFAULT 0,
                output_sha256 TEXT,
                failure TEXT
            ) STRICT;
            CREATE TABLE checkpoints(
                job_sha256 TEXT NOT NULL REFERENCES jobs(job_sha256),
                completed_updates INTEGER NOT NULL CHECK(completed_updates >= 0),
                artifact_sha256 TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                PRIMARY KEY(job_sha256, completed_updates)
            ) STRICT;
            CREATE TABLE resume_audits(
                job_sha256 TEXT NOT NULL REFERENCES jobs(job_sha256),
                artifact_sha256 TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                PRIMARY KEY(job_sha256, artifact_sha256)
            ) STRICT;
            CREATE TABLE events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                at_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                job_sha256 TEXT,
                detail_json TEXT NOT NULL
            ) STRICT;
            COMMIT;
            """
        )

    def _metadata(self, connection: sqlite3.Connection, key: str) -> object:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"workflow metadata is missing {key}")
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"workflow metadata {key} is corrupt") from exc

    def _verify_state_machine(
        self,
        connection: sqlite3.Connection,
        manifest: Mapping[str, object],
    ) -> None:
        rows = {
            str(row["job_sha256"]): row
            for row in connection.execute("SELECT * FROM jobs")
        }
        checkpoints = {
            (str(row["job_sha256"]), int(row["completed_updates"])): row
            for row in connection.execute("SELECT * FROM checkpoints")
        }
        resume_audits = {
            (str(row["job_sha256"]), str(row["artifact_sha256"])): row
            for row in connection.execute("SELECT * FROM resume_audits")
        }
        histories = {
            job_sha256: {
                "status": "pending",
                "owner": None,
                "claimed_ns": None,
                "started_ns": None,
                "finished_ns": None,
                "resume_count": 0,
                "output_sha256": None,
                "failure": None,
            }
            for job_sha256 in rows
        }
        seen_checkpoints: set[tuple[str, int]] = set()
        seen_resume_audits: set[tuple[str, str]] = set()
        seen_install_phases: set[str] = set()
        previous_sequence = 0
        previous_time = 0
        events = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        if not events:
            raise ValueError("workflow event history is empty")
        for event in events:
            sequence = int(event["sequence"])
            at_ns = int(event["at_ns"])
            if sequence != previous_sequence + 1 or at_ns < previous_time:
                raise ValueError("workflow event order is corrupt")
            previous_sequence, previous_time = sequence, at_ns
            try:
                detail = json.loads(event["detail_json"])
            except json.JSONDecodeError as exc:
                raise ValueError("workflow event detail is corrupt") from exc
            if event["detail_json"] != _canonical_bytes(detail).decode():
                raise ValueError("workflow event detail is noncanonical")
            kind = str(event["kind"])
            job_sha256 = event["job_sha256"]
            if kind == "run_created":
                if sequence != 1 or job_sha256 is not None or detail != manifest:
                    raise ValueError("workflow creation event differs")
                continue
            if kind in {"validation_jobs_installed", "test_jobs_installed"}:
                phase = kind.removesuffix("_jobs_installed")
                if (
                    job_sha256 is not None
                    or phase in seen_install_phases
                    or not isinstance(detail, dict)
                    or set(detail) != {"authorization_sha256", "job_sha256s"}
                    or not isinstance(detail["job_sha256s"], list)
                    or set(detail["job_sha256s"])
                    != {
                        identity
                        for identity, row in rows.items()
                        if row["phase"] == phase
                    }
                    or any(
                        json.loads(rows[identity]["manifest_json"]).get(
                            "authorization_sha256"
                        )
                        != detail["authorization_sha256"]
                        for identity in detail["job_sha256s"]
                    )
                ):
                    raise ValueError("workflow phase-install event differs")
                seen_install_phases.add(phase)
                continue
            if not isinstance(job_sha256, str) or job_sha256 not in histories:
                raise ValueError("workflow event references an unknown job")
            state = histories[job_sha256]
            if kind == "job_claimed":
                if state["status"] != "pending" or set(detail) != {"owner"}:
                    raise ValueError("workflow claim event transition is invalid")
                state.update(
                    status="claimed",
                    owner=detail["owner"],
                    claimed_ns=at_ns,
                )
            elif kind == "job_recovered":
                if (
                    state["status"] not in {"claimed", "running"}
                    or rows[job_sha256]["sealed"]
                    or set(detail)
                    != {"owner", "completed_updates", "checkpoint_sha256"}
                    or (
                        job_sha256,
                        detail["completed_updates"],
                    )
                    not in checkpoints
                    or checkpoints[(job_sha256, detail["completed_updates"])][
                        "artifact_sha256"
                    ]
                    != detail["checkpoint_sha256"]
                ):
                    raise ValueError("workflow recovery event transition is invalid")
                state.update(
                    status="claimed",
                    owner=detail["owner"],
                    claimed_ns=at_ns,
                    started_ns=None,
                    resume_count=int(state["resume_count"]) + 1,
                )
            elif kind == "job_started":
                if state["status"] != "claimed" or detail != {}:
                    raise ValueError("workflow start event transition is invalid")
                state.update(status="running", started_ns=at_ns)
            elif kind == "checkpoint_published":
                key = (
                    job_sha256,
                    detail.get("completed_updates")
                    if isinstance(detail, dict)
                    else None,
                )
                if (
                    state["status"] != "running"
                    or set(detail) != {"completed_updates", "artifact_sha256"}
                    or key not in checkpoints
                    or checkpoints[key]["artifact_sha256"] != detail["artifact_sha256"]
                    or checkpoints[key]["created_ns"] != at_ns
                    or key in seen_checkpoints
                ):
                    raise ValueError("workflow checkpoint event differs")
                seen_checkpoints.add(key)
            elif kind == "training_completed":
                if state["status"] != "running" or detail != {}:
                    raise ValueError("workflow training event transition is invalid")
                state["status"] = "trained"
            elif kind == "resume_audit_published":
                key = (
                    job_sha256,
                    detail.get("artifact_sha256") if isinstance(detail, dict) else None,
                )
                if (
                    state["status"] != "trained"
                    or set(detail) != {"artifact_sha256"}
                    or key not in resume_audits
                    or resume_audits[key]["created_ns"] != at_ns
                    or key in seen_resume_audits
                ):
                    raise ValueError("workflow resume-audit event differs")
                seen_resume_audits.add(key)
            elif kind == "job_completed":
                if state["status"] not in {"running", "trained"} or set(detail) != {
                    "output_sha256"
                }:
                    raise ValueError("workflow completion event transition is invalid")
                state.update(
                    status="completed",
                    finished_ns=at_ns,
                    output_sha256=detail["output_sha256"],
                )
            elif kind == "job_failed":
                if state["status"] not in {"claimed", "running", "trained"} or set(
                    detail
                ) != {"reason"}:
                    raise ValueError("workflow failure event transition is invalid")
                state.update(
                    status="failed",
                    finished_ns=at_ns,
                    failure=detail["reason"],
                )
            else:
                raise ValueError(f"unknown workflow event kind: {kind}")
        if seen_checkpoints != set(checkpoints):
            raise ValueError("workflow checkpoints and events differ")
        if seen_resume_audits != set(resume_audits):
            raise ValueError("workflow resume audits and events differ")
        present_phases = {str(row["phase"]) for row in rows.values()}
        if (present_phases - {"calibration"}) != seen_install_phases:
            raise ValueError("workflow phase installation history differs")
        for job_sha256, row in rows.items():
            state = histories[job_sha256]
            comparable = (
                "status",
                "owner",
                "claimed_ns",
                "started_ns",
                "finished_ns",
                "resume_count",
                "output_sha256",
                "failure",
            )
            if any(row[name] != state[name] for name in comparable):
                raise ValueError("workflow job row differs from its event history")
            active = row["status"] in {"claimed", "running", "trained"}
            if active != (row["token_sha256"] is not None):
                raise ValueError("workflow job token state is inconsistent")
            job_checkpoints = sorted(
                update for identity, update in checkpoints if identity == job_sha256
            )
            if any(
                update < 0 or update > row["budget_updates"]
                for update in job_checkpoints
            ) or (
                row["status"] in {"trained", "completed"}
                and (
                    not job_checkpoints or job_checkpoints[-1] != row["budget_updates"]
                )
            ):
                raise ValueError("workflow checkpoint progression is inconsistent")

    def verify(self) -> dict[str, object]:
        connection = self._connect()
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("workflow database integrity check failed")
            if self._metadata(connection, "schema_version") != self.schema_version:
                raise ValueError("workflow schema version differs")
            manifest = self._metadata(connection, "manifest")
            plan = self._metadata(connection, "plan")
            config = self._metadata(connection, "operational_config")
            if (
                not isinstance(manifest, dict)
                or manifest.get("version") != self.schema_version
                or manifest.get("plan_sha256") != _sha256(plan)
                or manifest.get("operational_config_sha256") != _sha256(config)
                or manifest.get("transfer_eligible") is not False
            ):
                raise ValueError("workflow manifest identities differ")
            if manifest.get("run_class") == "canonical" and (
                manifest.get("plan_sha256") != CANONICAL_PLAN_SHA256
                or manifest.get("operational_config_sha256")
                != CANONICAL_OPERATIONAL_CONFIG_SHA256
                or manifest.get("snapshot_bundle_sha256")
                != CANONICAL_EXACT_SNAPSHOT_BUNDLE_SHA256
                or manifest.get("portable_snapshot_bundle_sha256")
                != CANONICAL_PORTABLE_SNAPSHOT_BUNDLE_SHA256
                or manifest.get("pairing_sha256s") != CANONICAL_PAIRING_SHA256S
                or manifest.get("acceptance_eligible") is not True
            ):
                raise ValueError("canonical workflow differs from preregistered locks")
            if manifest.get("run_class") == "smoke" and (
                manifest.get("acceptance_eligible") is not False
                or manifest.get("portable_snapshot_bundle_sha256") is not None
                or manifest.get("pairing_sha256s") is not None
            ):
                raise ValueError("smoke workflow eligibility is inconsistent")
            for row in connection.execute(
                "SELECT job_sha256, manifest_json, sealed, phase,budget_updates "
                "FROM jobs"
            ):
                try:
                    job = TrialJob.from_manifest(json.loads(row["manifest_json"]))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("workflow job manifest is corrupt") from exc
                if (
                    job.sha256 != row["job_sha256"]
                    or job.phase != row["phase"]
                    or job.budget_updates != row["budget_updates"]
                    or bool(row["sealed"]) != job.sealed
                    or job.sealed != (row["phase"] == "test")
                ):
                    raise ValueError("workflow job identity differs")
            self._verify_state_machine(connection, manifest)
            return manifest
        finally:
            connection.close()

    def append_jobs(
        self,
        jobs: Sequence[TrialJob],
        *,
        authorization: ValidationRunAuthorization | SealedTestRunAuthorization,
    ) -> None:
        if not jobs:
            raise ValueError("cannot append an empty phase")
        phase = jobs[0].phase
        if phase not in {"validation", "test"} or any(
            job.phase != phase for job in jobs
        ):
            raise ValueError("appended jobs must form one authorized later phase")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            plan = R3BExperimentPlan.from_mapping(self._metadata(connection, "plan"))
            expected_type = (
                ValidationRunAuthorization
                if phase == "validation"
                else SealedTestRunAuthorization
            )
            if (
                not isinstance(authorization, expected_type)
                or authorization.plan.sha256 != plan.sha256
            ):
                raise ValueError("later-phase authorization type or plan differs")
            expected_jobs = plan.trial_jobs(phase, authorization)
            if tuple(jobs) != expected_jobs:
                raise ValueError(
                    f"{phase} jobs must equal the complete authorized job set"
                )
            authorization_sha256 = authorization.sha256
            if any(job.authorization_sha256 != authorization_sha256 for job in jobs):
                raise ValueError("job authorization identity differs")
            if connection.execute(
                "SELECT 1 FROM jobs WHERE phase = ? LIMIT 1", (phase,)
            ).fetchone():
                raise ValueError(f"{phase} jobs were already installed")
            prerequisite = "calibration" if phase == "validation" else "validation"
            states = {
                row[0]
                for row in connection.execute(
                    "SELECT status FROM jobs WHERE phase = ?", (prerequisite,)
                )
            }
            if states != {"completed"}:
                raise ValueError(f"{prerequisite} is not completely successful")
            now = time_ns()
            for job in jobs:
                connection.execute(
                    """
                    INSERT INTO jobs(
                        job_sha256, phase, sealed, budget_updates, status,
                        manifest_json, created_ns
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        job.sha256,
                        phase,
                        int(job.sealed),
                        job.budget_updates,
                        _canonical_bytes(job.manifest()).decode(),
                        now,
                    ),
                )
            connection.execute(
                "INSERT INTO events(at_ns, kind, detail_json) VALUES (?, ?, ?)",
                (
                    now,
                    f"{phase}_jobs_installed",
                    _canonical_bytes(
                        {
                            "authorization_sha256": authorization_sha256,
                            "job_sha256s": [job.sha256 for job in jobs],
                        }
                    ).decode(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_next(
        self,
        phase: str,
        *,
        owner: str,
        token: str | None = None,
    ) -> JobClaim | None:
        if phase not in {"calibration", "validation", "test"}:
            raise ValueError("unknown phase")
        if not _SAFE_ID.fullmatch(owner):
            raise ValueError("claim owner is unsafe")
        token = secrets.token_hex(32) if token is None else token
        if (
            not isinstance(token, str)
            or len(token) != 64
            or token == _SHA256_ZERO
            or any(character not in "0123456789abcdef" for character in token)
        ):
            raise ValueError("claim token is malformed")
        token_sha = hashlib.sha256(token.encode()).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT job_sha256 FROM jobs
                WHERE phase = ? AND status = 'pending'
                ORDER BY job_sha256 LIMIT 1
                """,
                (phase,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            now = time_ns()
            changed = connection.execute(
                """
                UPDATE jobs
                SET status='claimed', owner=?, token_sha256=?, claimed_ns=?
                WHERE job_sha256=? AND status='pending'
                """,
                (owner, token_sha, now, row["job_sha256"]),
            ).rowcount
            if changed != 1:
                raise RuntimeError("job claim race was not serialized")
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'job_claimed', ?, ?)
                """,
                (
                    now,
                    row["job_sha256"],
                    _canonical_bytes({"owner": owner}).decode(),
                ),
            )
            connection.commit()
            return JobClaim(row["job_sha256"], phase, token, owner, 0, None)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def resume_unstarted_claim(
        self,
        phase: str,
        *,
        owner: str,
        token: str,
    ) -> JobClaim | None:
        """Recover a claim whose caller durably stored its token before commit."""

        if phase not in {"calibration", "validation", "test"}:
            raise ValueError("unknown phase")
        if not _SAFE_ID.fullmatch(owner) or len(token) != 64:
            raise ValueError("claim recovery identity is malformed")
        token_sha256 = hashlib.sha256(token.encode()).hexdigest()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT job_sha256,status FROM jobs "
                "WHERE phase=? AND owner=? AND token_sha256=?",
                (phase, owner, token_sha256),
            ).fetchall()
        finally:
            connection.close()
        if not rows:
            return None
        if len(rows) != 1 or rows[0]["status"] != "claimed":
            raise RuntimeError("precommitted claim token is active but already started")
        return JobClaim(str(rows[0]["job_sha256"]), phase, token, owner, 0, None)

    @staticmethod
    def _authorized_job(
        connection: sqlite3.Connection,
        claim: JobClaim,
        allowed: tuple[str, ...],
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_sha256=?", (claim.job_sha256,)
        ).fetchone()
        if (
            row is None
            or row["phase"] != claim.phase
            or row["owner"] != claim.owner
            or row["status"] not in allowed
            or row["token_sha256"] != hashlib.sha256(claim.token.encode()).hexdigest()
        ):
            raise ValueError("job claim is absent, stale, or unauthorized")
        return row

    def begin(self, claim: JobClaim) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._authorized_job(connection, claim, ("claimed",))
            now = time_ns()
            connection.execute(
                "UPDATE jobs SET status='running', started_ns=? WHERE job_sha256=?",
                (now, claim.job_sha256),
            )
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'job_started', ?, '{}')
                """,
                (now, claim.job_sha256),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_checkpoint(
        self, claim: JobClaim, completed_updates: int, artifact_sha256: str
    ) -> None:
        updates = _nonnegative_int(completed_updates, "completed updates")
        artifact = _require_sha256(artifact_sha256, "checkpoint artifact")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            job = self._authorized_job(connection, claim, ("running",))
            if updates > job["budget_updates"]:
                raise ValueError("checkpoint exceeds the authorized job budget")
            previous = connection.execute(
                """
                SELECT completed_updates FROM checkpoints
                WHERE job_sha256=? ORDER BY completed_updates DESC LIMIT 1
                """,
                (claim.job_sha256,),
            ).fetchone()
            if previous is not None and updates <= previous["completed_updates"]:
                raise ValueError("checkpoint updates must increase")
            now = time_ns()
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?)",
                (claim.job_sha256, updates, artifact, now),
            )
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'checkpoint_published', ?, ?)
                """,
                (
                    now,
                    claim.job_sha256,
                    _canonical_bytes(
                        {
                            "completed_updates": updates,
                            "artifact_sha256": artifact,
                        }
                    ).decode(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_trained(self, claim: JobClaim) -> None:
        """Close mutation authority after the final training checkpoint."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            job = self._authorized_job(connection, claim, ("running",))
            latest = connection.execute(
                """
                SELECT completed_updates FROM checkpoints
                WHERE job_sha256=? ORDER BY completed_updates DESC LIMIT 1
                """,
                (claim.job_sha256,),
            ).fetchone()
            if latest is None or latest["completed_updates"] != job["budget_updates"]:
                raise ValueError("training lacks its final checkpoint")
            now = time_ns()
            connection.execute(
                """
                UPDATE jobs SET status='trained'
                WHERE job_sha256=?
                """,
                (claim.job_sha256,),
            )
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'training_completed', ?, '{}')
                """,
                (now, claim.job_sha256),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def publish_resume_audit(
        self,
        claim: JobClaim,
        verified_artifact: object,
        *,
        store: object,
        verifier_identity_sha256: str,
        build_identity_sha256: str,
    ) -> str:
        """Capture and anchor the unforgeable in-memory resume capability."""

        from .r3b_artifacts import ArtifactStore, ExactResumeVerificationReceipt

        if not isinstance(store, ArtifactStore):
            raise TypeError("resume audit publication requires an ArtifactStore")
        receipt = ExactResumeVerificationReceipt.capture(
            verified_artifact,
            verifier_identity_sha256=verifier_identity_sha256,
            build_identity_sha256=build_identity_sha256,
        ).publish(store)
        artifact = receipt.artifact_id
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._authorized_job(connection, claim, ("trained",))
            existing = connection.execute(
                "SELECT 1 FROM resume_audits WHERE job_sha256=? AND artifact_sha256=?",
                (claim.job_sha256, artifact),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return artifact
            if connection.execute(
                "SELECT 1 FROM resume_audits WHERE job_sha256=?",
                (claim.job_sha256,),
            ).fetchone():
                raise RuntimeError("job already anchors another resume audit")
            now = time_ns()
            connection.execute(
                "INSERT INTO resume_audits VALUES (?,?,?)",
                (claim.job_sha256, artifact, now),
            )
            connection.execute(
                "INSERT INTO events(at_ns,kind,job_sha256,detail_json) "
                "VALUES (?,'resume_audit_published',?,?)",
                (
                    now,
                    claim.job_sha256,
                    _canonical_bytes({"artifact_sha256": artifact}).decode(),
                ),
            )
            connection.commit()
            return artifact
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def verify_resume_audit(
        self,
        job_sha256: str,
        artifact_sha256: str,
    ) -> bool:
        job = _require_sha256(job_sha256, "resume audit job")
        artifact = _require_sha256(artifact_sha256, "resume audit artifact")
        self.verify()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM resume_audits WHERE job_sha256=? AND artifact_sha256=?",
                (job, artifact),
            ).fetchone()
        finally:
            connection.close()
        return row is not None

    def job_record(self, job_sha256: str) -> dict[str, object]:
        job_id = _require_sha256(job_sha256, "job identity")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT phase,sealed,budget_updates,status,manifest_json,owner,
                    resume_count,output_sha256,failure
                FROM jobs WHERE job_sha256=?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown workflow job: {job_id}")
            latest = connection.execute(
                """
                SELECT completed_updates,artifact_sha256 FROM checkpoints
                WHERE job_sha256=? ORDER BY completed_updates DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            return {
                "job_sha256": job_id,
                "phase": row["phase"],
                "sealed": bool(row["sealed"]),
                "budget_updates": row["budget_updates"],
                "status": row["status"],
                "manifest": json.loads(row["manifest_json"]),
                "owner": row["owner"],
                "resume_count": row["resume_count"],
                "output_sha256": row["output_sha256"],
                "failure": row["failure"],
                "latest_checkpoint": (
                    None
                    if latest is None
                    else {
                        "completed_updates": latest["completed_updates"],
                        "artifact_sha256": latest["artifact_sha256"],
                    }
                ),
            }
        finally:
            connection.close()

    def job_checkpoints(self, job_sha256: str) -> tuple[dict[str, object], ...]:
        """Return the verified immutable checkpoint receipts for one job."""

        identity = _require_sha256(job_sha256, "job identity")
        self.verify()
        connection = self._connect()
        try:
            if (
                connection.execute(
                    "SELECT 1 FROM jobs WHERE job_sha256=?", (identity,)
                ).fetchone()
                is None
            ):
                raise KeyError(identity)
            return tuple(
                {
                    "completed_updates": int(row["completed_updates"]),
                    "artifact_sha256": str(row["artifact_sha256"]),
                    "created_ns": int(row["created_ns"]),
                }
                for row in connection.execute(
                    "SELECT completed_updates,artifact_sha256,created_ns "
                    "FROM checkpoints WHERE job_sha256=? "
                    "ORDER BY completed_updates",
                    (identity,),
                )
            )
        finally:
            connection.close()

    def phase_job_records(self, phase: str) -> tuple[dict[str, object], ...]:
        """Return all jobs in one phase in deterministic identity order."""

        if phase not in {"calibration", "validation", "test"}:
            raise ValueError("unknown workflow phase")
        self.verify()
        connection = self._connect()
        try:
            identities = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT job_sha256 FROM jobs WHERE phase=? ORDER BY job_sha256",
                    (phase,),
                )
            )
        finally:
            connection.close()
        return tuple(self.job_record(identity) for identity in identities)

    def recover(
        self,
        job_sha256: str,
        *,
        checkpoint_sha256: str,
        owner: str,
    ) -> JobClaim:
        job_id = _require_sha256(job_sha256, "job identity")
        checkpoint = _require_sha256(checkpoint_sha256, "checkpoint identity")
        if not _SAFE_ID.fullmatch(owner):
            raise ValueError("claim owner is unsafe")
        token = secrets.token_hex(32)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_sha256=?", (job_id,)
            ).fetchone()
            latest = connection.execute(
                """
                SELECT completed_updates, artifact_sha256 FROM checkpoints
                WHERE job_sha256=? ORDER BY completed_updates DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if (
                job is None
                or job["sealed"]
                or job["status"] not in {"claimed", "running"}
                or latest is None
                or latest["artifact_sha256"] != checkpoint
                or latest["completed_updates"] >= job["budget_updates"]
            ):
                raise ValueError("job is not eligible for exact checkpoint recovery")
            now = time_ns()
            connection.execute(
                """
                UPDATE jobs SET status='claimed', owner=?, token_sha256=?,
                    claimed_ns=?, started_ns=NULL, resume_count=resume_count+1
                WHERE job_sha256=?
                """,
                (
                    owner,
                    hashlib.sha256(token.encode()).hexdigest(),
                    now,
                    job_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'job_recovered', ?, ?)
                """,
                (
                    now,
                    job_id,
                    _canonical_bytes(
                        {
                            "owner": owner,
                            "completed_updates": latest["completed_updates"],
                            "checkpoint_sha256": checkpoint,
                        }
                    ).decode(),
                ),
            )
            connection.commit()
            return JobClaim(
                job_id,
                job["phase"],
                token,
                owner,
                latest["completed_updates"],
                checkpoint,
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete(self, claim: JobClaim, output_sha256: str) -> None:
        output = _require_sha256(output_sha256, "job output")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            job = self._authorized_job(connection, claim, ("running", "trained"))
            latest = connection.execute(
                """
                SELECT completed_updates FROM checkpoints
                WHERE job_sha256=? ORDER BY completed_updates DESC LIMIT 1
                """,
                (claim.job_sha256,),
            ).fetchone()
            if latest is None or latest["completed_updates"] != job["budget_updates"]:
                raise ValueError("job cannot complete without its final checkpoint")
            now = time_ns()
            connection.execute(
                """
                UPDATE jobs SET status='completed', output_sha256=?, finished_ns=?,
                    token_sha256=NULL
                WHERE job_sha256=?
                """,
                (output, now, claim.job_sha256),
            )
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'job_completed', ?, ?)
                """,
                (
                    now,
                    claim.job_sha256,
                    _canonical_bytes({"output_sha256": output}).decode(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def reconcile_sealed_completion(
        self,
        *,
        ledger: object,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        outcome_sha256: str,
        output_sha256: str,
    ) -> None:
        """Finish a ledger-first sealed commit after the worker process is gone.

        The one-shot ledger replaces the lost workflow bearer token as authority.
        It can only authorize the exact immutable outcome already committed for
        this job; it never makes a sealed job retryable.
        """

        from .r3b_experiments import SealedTestLedger

        outcome = _require_sha256(outcome_sha256, "sealed learner outcome")
        output = _require_sha256(output_sha256, "sealed job output")
        if (
            not isinstance(ledger, SealedTestLedger)
            or not isinstance(sealed_run, SealedTestRunAuthorization)
            or not isinstance(job, TrialJob)
            or not job.sealed
            or job.phase != "test"
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
            or not ledger.verify_completed_job(sealed_run, job, outcome)
        ):
            raise ValueError("sealed workflow reconciliation lacks ledger authority")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT phase,sealed,budget_updates,status,manifest_json,"
                "output_sha256 FROM jobs WHERE job_sha256=?",
                (job.sha256,),
            ).fetchone()
            if row is None:
                raise ValueError("sealed workflow job is absent")
            try:
                stored_job = TrialJob.from_manifest(json.loads(row["manifest_json"]))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("sealed workflow job manifest is corrupt") from exc
            if (
                stored_job != job
                or row["phase"] != "test"
                or not bool(row["sealed"])
                or row["budget_updates"] != job.budget_updates
            ):
                raise ValueError("sealed workflow job differs from ledger authority")
            if row["status"] == "completed":
                if row["output_sha256"] != output:
                    raise RuntimeError(
                        "completed workflow output differs from the ledger result"
                    )
                connection.commit()
                return
            latest = connection.execute(
                "SELECT completed_updates FROM checkpoints WHERE job_sha256=? "
                "ORDER BY completed_updates DESC LIMIT 1",
                (job.sha256,),
            ).fetchone()
            if (
                row["status"] not in {"running", "trained"}
                or latest is None
                or latest["completed_updates"] != job.budget_updates
            ):
                raise RuntimeError(
                    "sealed workflow is not at a completed training boundary"
                )
            now = time_ns()
            updated = connection.execute(
                "UPDATE jobs SET status='completed',output_sha256=?,finished_ns=?,"
                "token_sha256=NULL WHERE job_sha256=? "
                "AND status IN ('running','trained')",
                (output, now, job.sha256),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed workflow reconciliation race was lost")
            connection.execute(
                "INSERT INTO events(at_ns,kind,job_sha256,detail_json) "
                "VALUES (?,'job_completed',?,?)",
                (
                    now,
                    job.sha256,
                    _canonical_bytes({"output_sha256": output}).decode(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def reconcile_sealed_failure(
        self,
        *,
        ledger: object,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        failure_reason: str,
    ) -> None:
        """Mirror an immutable ledger failure into workflow state after a crash."""

        from .r3b_experiments import SealedTestLedger

        if (
            not isinstance(ledger, SealedTestLedger)
            or not isinstance(sealed_run, SealedTestRunAuthorization)
            or not isinstance(job, TrialJob)
            or not isinstance(failure_reason, str)
            or not failure_reason
            or not ledger.verify_failed_job(sealed_run, job, failure_reason)
        ):
            raise ValueError("sealed workflow failure lacks ledger authority")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT phase,sealed,status,manifest_json,failure FROM jobs "
                "WHERE job_sha256=?",
                (job.sha256,),
            ).fetchone()
            if row is None:
                raise ValueError("sealed workflow job is absent")
            try:
                stored_job = TrialJob.from_manifest(json.loads(row["manifest_json"]))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("sealed workflow job manifest is corrupt") from exc
            if stored_job != job or row["phase"] != "test" or not bool(row["sealed"]):
                raise ValueError("sealed workflow job differs from ledger authority")
            if row["status"] == "failed":
                if row["failure"] != failure_reason:
                    raise RuntimeError(
                        "workflow failure differs from the ledger result"
                    )
                connection.commit()
                return
            if row["status"] not in {"claimed", "running", "trained"}:
                raise RuntimeError("sealed workflow failure state cannot reconcile")
            now = time_ns()
            updated = connection.execute(
                "UPDATE jobs SET status='failed',failure=?,finished_ns=?,"
                "token_sha256=NULL WHERE job_sha256=? "
                "AND status IN ('claimed','running','trained')",
                (failure_reason, now, job.sha256),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed workflow failure race was lost")
            connection.execute(
                "INSERT INTO events(at_ns,kind,job_sha256,detail_json) "
                "VALUES (?,'job_failed',?,?)",
                (
                    now,
                    job.sha256,
                    _canonical_bytes({"reason": failure_reason}).decode(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def fail(self, claim: JobClaim, reason: str) -> None:
        if not isinstance(reason, str) or not reason or len(reason) > 4096:
            raise ValueError("failure reason must be a bounded nonempty string")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._authorized_job(connection, claim, ("claimed", "running", "trained"))
            now = time_ns()
            connection.execute(
                """
                UPDATE jobs SET status='failed', failure=?, finished_ns=?,
                    token_sha256=NULL
                WHERE job_sha256=?
                """,
                (reason, now, claim.job_sha256),
            )
            connection.execute(
                """
                INSERT INTO events(at_ns, kind, job_sha256, detail_json)
                VALUES (?, 'job_failed', ?, ?)
                """,
                (
                    now,
                    claim.job_sha256,
                    _canonical_bytes({"reason": reason}).decode(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def status(self) -> dict[str, object]:
        manifest = self.verify()
        connection = self._connect()
        try:
            phases: dict[str, dict[str, int]] = {}
            for row in connection.execute(
                """
                SELECT phase, status, COUNT(*) AS count FROM jobs
                GROUP BY phase, status ORDER BY phase, status
                """
            ):
                phases.setdefault(row["phase"], {})[row["status"]] = row["count"]
            jobs = [
                {
                    "job_sha256": row["job_sha256"],
                    "phase": row["phase"],
                    "status": row["status"],
                    "owner": row["owner"],
                    "budget_updates": row["budget_updates"],
                    "completed_updates": row["completed_updates"] or 0,
                    "output_sha256": row["output_sha256"],
                    "failure": row["failure"],
                }
                for row in connection.execute(
                    "SELECT jobs.job_sha256,jobs.phase,jobs.status,jobs.owner,"
                    "jobs.budget_updates,jobs.output_sha256,jobs.failure,"
                    "MAX(checkpoints.completed_updates) AS completed_updates "
                    "FROM jobs LEFT JOIN checkpoints USING(job_sha256) "
                    "WHERE jobs.status != 'pending' "
                    "GROUP BY jobs.job_sha256 ORDER BY jobs.phase,jobs.job_sha256"
                )
            ]
            return {
                "version": "r3b-workflow-status-v1",
                "run_id": manifest["run_id"],
                "run_class": manifest["run_class"],
                "acceptance_eligible": manifest["acceptance_eligible"],
                "transfer_eligible": False,
                "phases": phases,
                "jobs": jobs,
            }
        finally:
            connection.close()
