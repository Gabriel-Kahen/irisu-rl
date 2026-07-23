"""Command-line entry point for R3 snapshot and experiment operations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from irisu_env import IrisuEnv

from .r3b_experiments import (
    R3BExperimentPlan,
    SealedLearnerOutcomeReference,
    SealedTestLedger,
    finalize_persisted_sealed_test,
    load_plan,
)
from .r3b_canonical_runner import (
    CanonicalRunInputs,
    load_published_canonical_outcome,
)
from .r3b_local_runner import (
    run_local_canonical_updates,
    run_local_smoke_updates,
)
from .r3b_operational import R3BOperationalConfig, R3BWorkflow
from .r3b_snapshots import (
    SnapshotBundle,
    SnapshotSourceManifest,
    SnapshotSourcePlan,
    generate_snapshot_bundle,
    load_snapshot_bundle,
    pair_snapshot_bundles,
)
from .r3b_supervisor import evaluate_trained_canonical_job
from .r3b_artifacts import ArtifactStore, ArtifactTypeError
from .r3b_baselines import run_sealed_baselines
from .r3b_lock import R3BRunLock
from .r3b_phases import (
    acquire_sealed_job,
    load_sealed_authorization,
    load_validation_authorization,
    prepare_sealed_test_phase,
    prepare_validation_phase,
)
from .curriculum import SnapshotBlobStore, SnapshotLibrary


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if payload != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{path} must use canonical JSON encoding")
    return value


def _source_identity(project_root: Path) -> str:
    inputs = (
        sorted((project_root / "python" / "irisu_env").glob("*.py"))
        + sorted((project_root / "python" / "irisu_rl").glob("*.py"))
        + [project_root / "pyproject.toml", project_root / "uv.lock"]
    )
    manifest: list[dict[str, object]] = []
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source identity input is missing or unsafe: {path}")
        payload = path.read_bytes()
        manifest.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _clean_source_revision(project_root: Path) -> str:
    scope = ("python", "configs/rl", "pyproject.toml", "uv.lock")
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all", "--", *scope),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("canonical source cleanliness could not be verified")
    if status.stdout:
        raise RuntimeError(
            "canonical runs require a clean reviewed source/config worktree"
        )
    revision = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = revision.stdout.strip()
    if (
        revision.returncode != 0
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError("canonical source revision could not be resolved")
    return value


def _offline_snapshot_bundle(
    root: Path,
    config: R3BOperationalConfig,
    *,
    expected_backend: str,
) -> SnapshotBundle:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot bundle must be a real directory")
    manifest = _load_json(root / "bundle.json")
    if not isinstance(manifest, dict):
        raise ValueError("snapshot bundle manifest must be an object")
    expected = {
        "version",
        "source_sha256",
        "library_sha256",
        "store_sha256",
        "runtime_backend",
        "runtime_identity_sha256",
        "action_spec_sha256",
        "generator_version",
    }
    if set(manifest) != expected or manifest.get("version") != "r3b-snapshot-bundle-v1":
        raise ValueError("snapshot bundle manifest schema differs")
    source_value = _load_json(root / "source.json")
    library_value = _load_json(root / "library.json")
    if not isinstance(source_value, dict) or not isinstance(library_value, dict):
        raise ValueError("snapshot source and library must be objects")
    source = SnapshotSourceManifest.from_manifest(source_value)
    library = SnapshotLibrary.from_manifest(library_value)
    store = SnapshotBlobStore.from_directory(library, root / "snapshots")
    try:
        bundle = SnapshotBundle(
            source,
            library,
            store,
            str(manifest["runtime_backend"]),
            str(manifest["runtime_identity_sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("snapshot bundle identities are malformed") from exc
    counts = {
        split: sum(recipe.split == split for recipe in library.recipes)
        for split in ("train", "calibration", "validation", "test")
    }
    minimums = {
        "train": config.minimum_train_snapshots,
        "calibration": config.minimum_calibration_snapshots,
        "validation": config.minimum_validation_snapshots,
        "test": config.minimum_test_snapshots,
    }
    if (
        manifest["source_sha256"] != source.sha256
        or manifest["library_sha256"] != library.sha256
        or manifest["store_sha256"] != store.sha256
        or manifest != bundle.manifest()
        or manifest["generator_version"] != config.snapshot_generator_version
        or not all(counts[split] >= minimum for split, minimum in minimums.items())
        or bundle.runtime_backend != expected_backend
    ):
        raise ValueError("snapshot bundle does not satisfy the operational config")
    return bundle


def _snapshot_bundle_sha256(root: Path, config: R3BOperationalConfig) -> str:
    return _offline_snapshot_bundle(
        root, config, expected_backend=config.primary_backend
    ).sha256


def _validate_config_plan(
    config: R3BOperationalConfig, plan: R3BExperimentPlan
) -> None:
    if (
        config.minimum_validation_snapshots * config.validation_repetitions
        != plan.validation_episodes_per_policy
        or config.minimum_test_snapshots * config.test_repetitions
        != plan.test_episodes_per_policy
        or config.minimum_test_snapshots * config.test_repetitions
        < plan.minimum_baseline_episodes
    ):
        raise ValueError(
            "operational snapshot/repetition counts disagree with the frozen plan"
        )


def _simulator(args: argparse.Namespace) -> IrisuEnv:
    if args.backend == "portable":
        if args.library is None or args.worker is not None:
            raise ValueError("portable backend requires --library and forbids --worker")
        library = Path(args.library)
        if not library.is_absolute():
            raise ValueError("portable library path must be absolute")
        return IrisuEnv(library_path=library, physics_backend="portable")
    if args.worker is None or args.library is not None:
        raise ValueError("exact backend requires --worker and forbids --library")
    worker = Path(args.worker)
    if not worker.is_absolute():
        raise ValueError("exact worker path must be absolute")
    return IrisuEnv(worker_path=worker, physics_backend="exact")


def _command_snapshots_build(args: argparse.Namespace) -> dict[str, object]:
    plan = SnapshotSourcePlan.from_toml(args.source_config)
    source = plan.materialize(args.backend)
    simulator = _simulator(args)
    try:
        bundle = generate_snapshot_bundle(simulator, source, args.output)
        return {
            "version": "r3b-snapshot-build-result-v1",
            "plan_sha256": plan.sha256,
            "source_sha256": source.sha256,
            "bundle_sha256": bundle.sha256,
            "backend": args.backend,
            "recipes": len(bundle.library.recipes),
        }
    finally:
        simulator.close()


def _command_snapshots_verify(args: argparse.Namespace) -> dict[str, object]:
    simulator = _simulator(args)
    try:
        bundle = load_snapshot_bundle(args.bundle, simulator)
        return {
            "version": "r3b-snapshot-verification-v1",
            "bundle_sha256": bundle.sha256,
            "backend": args.backend,
            "recipes": len(bundle.library.recipes),
            "replay_verified": True,
        }
    finally:
        simulator.close()


def _command_config_verify(args: argparse.Namespace) -> dict[str, object]:
    config = R3BOperationalConfig.from_toml(args.config)
    plan = load_plan(args.plan)
    _validate_config_plan(config, plan)
    return {
        "version": "r3b-config-verification-v1",
        "plan_sha256": plan.sha256,
        "operational_config_sha256": config.sha256,
        "primary_backend": config.primary_backend,
        "transfer_eligible": False,
    }


def _command_experiment_init(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).resolve(strict=True)
    plan_path = Path(args.plan).resolve(strict=True)
    snapshot_root = Path(args.snapshots).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError("experiment output already exists")
    project_root = Path(__file__).resolve().parents[2]
    config = R3BOperationalConfig.from_toml(config_path)
    plan = load_plan(plan_path)
    _validate_config_plan(config, plan)
    portable_root: Path | None = None
    pairing_manifests: dict[str, object] | None = None
    if args.run_class == "canonical":
        if args.portable_snapshots is None:
            raise ValueError("canonical runs require --portable-snapshots")
        portable_root = Path(args.portable_snapshots).resolve(strict=True)
        exact_bundle = _offline_snapshot_bundle(
            snapshot_root, config, expected_backend="exact"
        )
        portable_bundle = _offline_snapshot_bundle(
            portable_root, config, expected_backend="portable"
        )
        pairings = pair_snapshot_bundles(portable_bundle, exact_bundle)
        pairing_manifests = {
            split: manifest.manifest() for split, manifest in pairings.items()
        }
        bundle_sha = exact_bundle.sha256
        portable_bundle_sha = portable_bundle.sha256
        pairing_sha256s = {
            split: manifest.sha256 for split, manifest in pairings.items()
        }
        source_revision = _clean_source_revision(project_root)
    else:
        if args.portable_snapshots is not None:
            raise ValueError("smoke runs do not accept --portable-snapshots")
        bundle_sha = _snapshot_bundle_sha256(snapshot_root, config)
        portable_bundle_sha = None
        pairing_sha256s = None
        source_revision = None
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    try:
        workflow = R3BWorkflow.create(
            output / "workflow.sqlite3",
            run_id=args.run_id,
            run_class=args.run_class,
            plan=plan,
            config=config,
            snapshot_bundle_sha256=bundle_sha,
            portable_snapshot_bundle_sha256=portable_bundle_sha,
            pairing_sha256s=pairing_sha256s,
            source_identity_sha256=_source_identity(project_root),
            source_revision=source_revision,
        )
        resolved = {
            "version": "r3b-resolved-run-v1",
            "workflow": workflow.verify(),
            "plan": plan.manifest(),
            "operational_config": config.manifest(),
            "snapshot_bundle_path": str(snapshot_root),
            "portable_snapshot_bundle_path": (
                None if portable_root is None else str(portable_root)
            ),
            "pairing_manifests": pairing_manifests,
        }
        payload = _canonical_bytes(resolved) + b"\n"
        fd = os.open(
            output / "resolved-run.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return workflow.status()
    except BaseException:
        # Keep an initialized database for diagnosis, but remove an empty shell.
        if not any(output.iterdir()):
            output.rmdir()
        raise


def _workflow(args: argparse.Namespace) -> R3BWorkflow:
    root = Path(args.run).resolve(strict=True)
    return R3BWorkflow(root / "workflow.sqlite3")


def _command_experiment_status(args: argparse.Namespace) -> dict[str, object]:
    return _workflow(args).status()


def _command_experiment_verify(args: argparse.Namespace) -> dict[str, object]:
    workflow = _workflow(args)
    manifest = workflow.verify()
    return {
        "version": "r3b-run-verification-v1",
        "run_id": manifest["run_id"],
        "workflow_valid": True,
        "status": workflow.status(),
    }


def _command_experiment_smoke_update(args: argparse.Namespace) -> dict[str, object]:
    return run_local_smoke_updates(
        args.run,
        worker_path=args.worker,
        max_new_updates=args.max_new_updates,
        owner=args.owner,
    ).manifest()


def _command_experiment_canonical_update(
    args: argparse.Namespace,
) -> dict[str, object]:
    if args.phase == "test":
        raise ValueError(
            "sealed test jobs must use canonical-run-job without a process boundary"
        )
    authorization = None
    ledger = None
    if args.phase in {"validation", "test"}:
        if args.authorization is None or args.library is None:
            raise ValueError(
                f"{args.phase} training requires --authorization and --library"
            )
        inputs = _canonical_inputs(args.run, args.worker, args.library)
        store = ArtifactStore(inputs.root / "artifacts")
        if args.phase == "validation":
            authorization = load_validation_authorization(
                inputs, store, args.authorization
            )
        else:
            sealed = load_sealed_authorization(inputs, store, args.authorization)
            acquired = acquire_sealed_job(inputs, sealed, owner=args.owner)
            authorization = acquired.lease
            ledger = acquired.ledger
    return run_local_canonical_updates(
        args.run,
        worker_path=args.worker,
        max_new_updates=args.max_new_updates,
        owner=args.owner,
        phase=args.phase,
        authorization=authorization,
        sealed_test_ledger=ledger,
    ).manifest()


def _command_experiment_canonical_evaluate(
    args: argparse.Namespace,
) -> dict[str, object]:
    if args.phase == "test":
        raise ValueError(
            "sealed test jobs must use canonical-run-job without a process boundary"
        )
    authorization = None
    ledger = None
    if args.phase in {"validation", "test"}:
        if args.authorization is None:
            raise ValueError(f"{args.phase} evaluation requires --authorization")
        inputs = _canonical_inputs(args.run, args.worker, args.library)
        store = ArtifactStore(inputs.root / "artifacts")
        if args.phase == "validation":
            authorization = load_validation_authorization(
                inputs, store, args.authorization
            )
        else:
            sealed = load_sealed_authorization(inputs, store, args.authorization)
            acquired = acquire_sealed_job(inputs, sealed, owner=args.owner)
            authorization = acquired.lease
            ledger = acquired.ledger
    return evaluate_trained_canonical_job(
        args.run,
        exact_worker_path=args.worker,
        portable_library_path=args.library,
        phase=args.phase,
        authorization=authorization,
        sealed_test_ledger=ledger,
    ).manifest()


def _canonical_inputs(
    run: str | Path, worker: str | Path, library: str | Path
) -> CanonicalRunInputs:
    root = Path(run).resolve(strict=True)
    worker_path = Path(worker)
    library_path = Path(library)
    if (
        not worker_path.is_absolute()
        or worker_path.is_symlink()
        or not worker_path.is_file()
    ):
        raise ValueError("exact worker path must be an absolute regular file")
    if (
        not library_path.is_absolute()
        or library_path.is_symlink()
        or not library_path.is_file()
    ):
        raise ValueError("portable library path must be an absolute regular file")
    exact = IrisuEnv(physics_backend="exact", worker_path=worker_path)
    portable = IrisuEnv(physics_backend="portable", library_path=library_path)
    try:
        return CanonicalRunInputs.load(
            root,
            exact_simulator=exact,
            portable_simulator=portable,
        )
    finally:
        exact.close()
        portable.close()


def _command_experiment_prepare_validation(
    args: argparse.Namespace,
) -> dict[str, object]:
    inputs = _canonical_inputs(args.run, args.worker, args.library)
    published = prepare_validation_phase(
        inputs, ArtifactStore(inputs.root / "artifacts")
    )
    return {
        "version": "r3b-validation-phase-prepared-v1",
        "authorization_artifact_sha256": published.artifact_sha256,
        "authorization_sha256": published.authorization.sha256,
        "jobs": len(inputs.workflow.phase_job_records("validation")),
        "transfer_eligible": False,
    }


def _command_experiment_prepare_test(args: argparse.Namespace) -> dict[str, object]:
    inputs = _canonical_inputs(args.run, args.worker, args.library)
    published = prepare_sealed_test_phase(
        inputs,
        ArtifactStore(inputs.root / "artifacts"),
        validation_artifact_sha256=args.authorization,
    )
    return {
        "version": "r3b-sealed-test-phase-prepared-v1",
        "authorization_artifact_sha256": published.artifact_sha256,
        "authorization_sha256": published.authorization.sha256,
        "jobs": len(inputs.workflow.phase_job_records("test")),
        "transfer_eligible": False,
    }


def _command_experiment_canonical_run_job(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Train, audit, evaluate, and commit exactly one canonical job."""

    inputs = _canonical_inputs(args.run, args.worker, args.library)
    store = ArtifactStore(inputs.root / "artifacts")
    authorization = None
    ledger = None
    if args.phase == "validation":
        if args.authorization is None:
            raise ValueError("validation requires --authorization")
        authorization = load_validation_authorization(inputs, store, args.authorization)
    elif args.phase == "test":
        if args.authorization is None:
            raise ValueError("test requires --authorization")
        sealed = load_sealed_authorization(inputs, store, args.authorization)
        acquired = acquire_sealed_job(inputs, sealed, owner=args.owner)
        authorization = acquired.lease
        ledger = acquired.ledger
    elif args.authorization is not None:
        raise ValueError("calibration does not accept --authorization")
    training = run_local_canonical_updates(
        args.run,
        worker_path=args.worker,
        max_new_updates=2**31 - 1,
        owner=args.owner,
        phase=args.phase,
        authorization=authorization,
        sealed_test_ledger=ledger,
    )
    if not training.training_complete:
        raise RuntimeError("canonical operational job stopped before full training")
    evaluation = evaluate_trained_canonical_job(
        args.run,
        exact_worker_path=args.worker,
        portable_library_path=args.library,
        phase=args.phase,
        authorization=authorization,
        sealed_test_ledger=ledger,
    )
    return {
        "version": "r3b-canonical-job-run-v1",
        "training": training.manifest(),
        "evaluation": evaluation.manifest(),
        "acceptance_eligible": True,
        "transfer_eligible": False,
    }


def _command_experiment_run_baselines(
    args: argparse.Namespace,
) -> dict[str, object]:
    inputs = _canonical_inputs(args.run, args.worker, args.library)
    store = ArtifactStore(inputs.root / "artifacts")
    sealed = load_sealed_authorization(inputs, store, args.authorization)
    return run_sealed_baselines(
        inputs,
        store,
        sealed,
        exact_worker_path=args.worker,
        portable_library_path=args.library,
    ).manifest()


def _command_experiment_finalize_test(
    args: argparse.Namespace,
) -> dict[str, object]:
    inputs = _canonical_inputs(args.run, args.worker, args.library)
    store = ArtifactStore(inputs.root / "artifacts")
    sealed = load_sealed_authorization(
        inputs,
        store,
        args.authorization,
        allow_finalized=True,
    )
    ledger = SealedTestLedger(sealed.authorization.ledger_path)
    finalized_sha256 = ledger.finalized_confirmation_sha256(sealed.authorization)
    completed = (
        {}
        if finalized_sha256 is not None
        else {
            record.job_sha256: record.outcome_sha256
            for record in ledger.terminal_job_records(sealed.authorization)
            if record.state == "complete"
        }
    )
    references: list[str] = []
    for artifact_id in store.list():
        try:
            reference = SealedLearnerOutcomeReference.load(store, artifact_id)
        except ArtifactTypeError:
            continue
        if (
            reference.plan_sha256 == inputs.plan.sha256
            and reference.authorization_sha256
            == sealed.authorization.authorization.sha256
            and (
                finalized_sha256 is not None
                or completed.get(reference.job_sha256)
                == reference.learner_outcome_sha256
            )
        ):
            references.append(artifact_id)
    package = store.load(
        args.authorization,
        expected_kind="irisu.r3b.sealed-test-run-package",
        expected_version="r3b-sealed-test-run-package-v1",
    ).payload
    if (
        not isinstance(package, dict)
        or type(package.get("sealed_authorization_artifact_sha256")) is not str
    ):
        raise ValueError("sealed test package lacks its authorization artifact")
    finalization = finalize_persisted_sealed_test(
        store=store,
        workflow=inputs.workflow,
        ledger=ledger,
        sealed_run=sealed.authorization,
        authorization_artifact_sha256=package["sealed_authorization_artifact_sha256"],
        baseline_artifact_sha256=args.baseline_artifact,
        outcome_reference_sha256s=tuple(sorted(references)),
        outcome_loader=lambda output_id: load_published_canonical_outcome(
            inputs=inputs,
            store=store,
            output_artifact_sha256=output_id,
        )[0],
    )
    return {
        "version": "r3b-sealed-test-finalization-v1",
        "report_artifact_sha256": finalization.report_artifact_sha256,
        "report_sha256": finalization.report.sha256,
        "accepted": finalization.report.accepted,
        "decision": finalization.report.decision.manifest(),
        "transfer_eligible": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irisu-r3",
        description="Fail-closed R3 snapshot and experiment operations",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    snapshots = commands.add_parser(
        "snapshots", help="build or verify replayable snapshot bundles"
    )
    snapshot_commands = snapshots.add_subparsers(dest="snapshot_command", required=True)
    snapshot_build = snapshot_commands.add_parser("build")
    snapshot_build.add_argument(
        "--source-config",
        default="configs/rl/snapshots/r3b-source-plan-v1.toml",
    )
    snapshot_build.add_argument(
        "--backend", choices=("portable", "exact"), required=True
    )
    snapshot_build.add_argument("--library")
    snapshot_build.add_argument("--worker")
    snapshot_build.add_argument("--output", required=True)
    snapshot_build.set_defaults(handler=_command_snapshots_build)
    snapshot_verify = snapshot_commands.add_parser("verify")
    snapshot_verify.add_argument("--bundle", required=True)
    snapshot_verify.add_argument(
        "--backend", choices=("portable", "exact"), required=True
    )
    snapshot_verify.add_argument("--library")
    snapshot_verify.add_argument("--worker")
    snapshot_verify.set_defaults(handler=_command_snapshots_verify)

    config = commands.add_parser("config", help="validate frozen configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_verify = config_commands.add_parser("verify")
    config_verify.add_argument(
        "--config",
        default="configs/rl/experiments/r3b-operational-v1.toml",
    )
    config_verify.add_argument(
        "--plan",
        default="configs/rl/experiments/r3b-completion-v1.toml",
    )
    config_verify.set_defaults(handler=_command_config_verify)

    experiment = commands.add_parser("experiment", help="manage durable R3 runs")
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    initialize = experiment_commands.add_parser("init")
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument(
        "--run-class", choices=("smoke", "canonical"), required=True
    )
    initialize.add_argument("--snapshots", required=True)
    initialize.add_argument("--portable-snapshots")
    initialize.add_argument("--output", required=True)
    initialize.add_argument(
        "--config",
        default="configs/rl/experiments/r3b-operational-v1.toml",
    )
    initialize.add_argument(
        "--plan",
        default="configs/rl/experiments/r3b-completion-v1.toml",
    )
    initialize.set_defaults(handler=_command_experiment_init)

    for name, handler in (
        ("status", _command_experiment_status),
        ("verify", _command_experiment_verify),
    ):
        command = experiment_commands.add_parser(name)
        command.add_argument("--run", required=True)
        command.set_defaults(handler=handler)
    smoke_update = experiment_commands.add_parser(
        "smoke-update", help="run a bounded exact-backend training segment"
    )
    smoke_update.add_argument("--run", required=True)
    smoke_update.add_argument("--worker", required=True)
    smoke_update.add_argument("--max-new-updates", type=int, default=1)
    smoke_update.add_argument("--owner", default=f"local-{os.getpid()}")
    smoke_update.set_defaults(handler=_command_experiment_smoke_update)
    canonical_update = experiment_commands.add_parser(
        "canonical-update",
        help="run a bounded canonical calibration training segment",
    )
    canonical_update.add_argument("--run", required=True)
    canonical_update.add_argument("--worker", required=True)
    canonical_update.add_argument(
        "--phase",
        choices=("calibration", "validation"),
        default="calibration",
    )
    canonical_update.add_argument("--library")
    canonical_update.add_argument("--authorization")
    canonical_update.add_argument("--max-new-updates", type=int, default=50)
    canonical_update.add_argument("--owner", default="canonical-runner")
    canonical_update.set_defaults(handler=_command_experiment_canonical_update)
    canonical_evaluate = experiment_commands.add_parser(
        "canonical-evaluate",
        help="evaluate and complete a trained canonical calibration job",
    )
    canonical_evaluate.add_argument("--run", required=True)
    canonical_evaluate.add_argument("--worker", required=True)
    canonical_evaluate.add_argument("--library", required=True)
    canonical_evaluate.add_argument(
        "--phase",
        choices=("calibration", "validation"),
        default="calibration",
    )
    canonical_evaluate.add_argument("--authorization")
    canonical_evaluate.add_argument("--owner", default="canonical-runner")
    canonical_evaluate.set_defaults(handler=_command_experiment_canonical_evaluate)
    prepare_validation = experiment_commands.add_parser("prepare-validation")
    prepare_validation.add_argument("--run", required=True)
    prepare_validation.add_argument("--worker", required=True)
    prepare_validation.add_argument("--library", required=True)
    prepare_validation.set_defaults(handler=_command_experiment_prepare_validation)
    prepare_test = experiment_commands.add_parser("prepare-test")
    prepare_test.add_argument("--run", required=True)
    prepare_test.add_argument("--worker", required=True)
    prepare_test.add_argument("--library", required=True)
    prepare_test.add_argument("--authorization", required=True)
    prepare_test.set_defaults(handler=_command_experiment_prepare_test)
    canonical_run = experiment_commands.add_parser(
        "canonical-run-job",
        help="train and evaluate one canonical job in one supervised process",
    )
    canonical_run.add_argument("--run", required=True)
    canonical_run.add_argument("--worker", required=True)
    canonical_run.add_argument("--library", required=True)
    canonical_run.add_argument(
        "--phase",
        choices=("calibration", "validation", "test"),
        required=True,
    )
    canonical_run.add_argument("--authorization")
    canonical_run.add_argument("--owner", default="canonical-runner")
    canonical_run.set_defaults(handler=_command_experiment_canonical_run_job)
    baselines = experiment_commands.add_parser(
        "run-baselines",
        help="execute the one-shot sealed scripted-baseline batch",
    )
    baselines.add_argument("--run", required=True)
    baselines.add_argument("--worker", required=True)
    baselines.add_argument("--library", required=True)
    baselines.add_argument("--authorization", required=True)
    baselines.set_defaults(handler=_command_experiment_run_baselines)
    finalize = experiment_commands.add_parser(
        "finalize-test",
        help="make the sole sealed confirmation decision",
    )
    finalize.add_argument("--run", required=True)
    finalize.add_argument("--worker", required=True)
    finalize.add_argument("--library", required=True)
    finalize.add_argument("--authorization", required=True)
    finalize.add_argument("--baseline-artifact")
    finalize.set_defaults(handler=_command_experiment_finalize_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        locked_commands = {
            "smoke-update",
            "canonical-update",
            "canonical-evaluate",
            "prepare-validation",
            "prepare-test",
            "canonical-run-job",
            "run-baselines",
            "finalize-test",
        }
        lock = (
            R3BRunLock(args.run)
            if args.command == "experiment"
            and args.experiment_command in locked_commands
            else nullcontext()
        )
        with lock:
            result = args.handler(args)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"irisu-r3: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
