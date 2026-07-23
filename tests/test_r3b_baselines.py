from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from irisu_rl.r3b_artifacts import ArtifactStore
from irisu_rl.r3b_baselines import (
    _load_baseline_bundle,
    _publish_baseline_bundle,
    run_sealed_baselines,
)
from irisu_rl.r3b_canonical_runner import PairedEvaluationSuites
from irisu_rl.r3b_experiments import CandidateArm, SealedTestLedger
from irisu_rl.r3b_phases import PublishedSealedAuthorization
from tests.test_r3b_experiments import (
    TEST_EVALUATION_SUITE,
    TEST_PLAN,
    VALIDATION_EVALUATION_SUITE,
    authorization_validation_results,
    valid_baseline_artifacts,
    validation_context,
)


def _sealed() -> tuple[PublishedSealedAuthorization, SealedTestLedger]:
    control = CandidateArm(0, TEST_PLAN.learning_rates[1])
    candidate = CandidateArm(100_000, TEST_PLAN.learning_rates[1])
    ledger, validation_run = validation_context(TEST_PLAN, control.learning_rate)
    authorization = ledger.authorize_once(
        TEST_PLAN,
        validation_run,
        authorization_validation_results(
            TEST_PLAN,
            control,
            candidate,
            validation_run=validation_run,
        ),
        VALIDATION_EVALUATION_SUITE,
        TEST_EVALUATION_SUITE,
    )
    return (
        PublishedSealedAuthorization(authorization, "a" * 64, "b" * 64),
        ledger,
    )


def _suites() -> SimpleNamespace:
    bundle = valid_baseline_artifacts(TEST_PLAN)[0]
    return SimpleNamespace(
        exact=bundle.primary_suite,
        portable=bundle.diagnostic_suite,
        logical_manifest=bundle.logical_manifest,
        exact_library=bundle.exact_library,
        portable_library=bundle.portable_library,
        workflow_manifest_sha256="c" * 64,
    )


def _inputs(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=root,
        plan=TEST_PLAN,
        workflow_manifest_sha256="c" * 64,
        config=SimpleNamespace(evaluation_shards=1),
    )


class SealedBaselineRunnerTests(unittest.TestCase):
    def test_completed_batch_recovers_without_lookup_index_or_binaries(self) -> None:
        sealed, _ledger = _sealed()
        suites = _suites()
        bundles = valid_baseline_artifacts(TEST_PLAN)
        reports = tuple(
            report
            for bundle in bundles
            for report in (
                bundle.primary_report,
                bundle.primary_replay_report,
                bundle.diagnostic_report,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            worker = root / "worker"
            library = root / "library.so"
            worker.touch()
            library.touch()
            inputs = _inputs(root)
            with (
                mock.patch.object(
                    PairedEvaluationSuites,
                    "build",
                    return_value=suites,
                ),
                mock.patch(
                    "irisu_rl.r3b_baselines._scripted_report",
                    side_effect=reports,
                ) as evaluate,
                mock.patch("irisu_rl.r3b_baselines.IrisuEnv") as environment,
            ):
                first = run_sealed_baselines(
                    inputs,  # type: ignore[arg-type]
                    store,
                    sealed,
                    exact_worker_path=worker,
                    portable_library_path=library,
                )
            self.assertEqual(evaluate.call_count, len(reports))
            self.assertEqual(
                [call.kwargs["purpose"] for call in evaluate.call_args_list],
                ["primary", "replay", "diagnostic"] * len(bundles),
            )
            self.assertEqual(
                [call.kwargs["physics_backend"] for call in environment.call_args_list],
                ["exact", "exact", "portable"] * len(bundles),
            )
            indexes = tuple(root.glob("sealed-baseline-index-*.sqlite3"))
            self.assertEqual(len(indexes), 1)
            indexes[0].unlink()
            worker.unlink()
            library.unlink()

            recovered = run_sealed_baselines(
                inputs,  # type: ignore[arg-type]
                store,
                sealed,
                exact_worker_path=worker,
                portable_library_path=library,
            )
            self.assertEqual(recovered.evidence_artifact_sha256, first.evidence_artifact_sha256)
            self.assertEqual(recovered.evidence_sha256s, first.evidence_sha256s)
            self.assertEqual(recovered.bundle_artifact_sha256s, ())

    def test_bundle_cache_is_bound_to_running_batch_lease(self) -> None:
        sealed_a, ledger_a = _sealed()
        sealed_b, ledger_b = _sealed()
        suites = _suites()
        bundle = valid_baseline_artifacts(TEST_PLAN)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            inputs = _inputs(root)
            lease_a = ledger_a.claim_baseline_batch(sealed_a.authorization)
            ledger_a.begin_baseline_batch(lease_a)
            published = _publish_baseline_bundle(
                inputs=inputs,  # type: ignore[arg-type]
                store=store,
                suites=suites,
                baseline=bundle.baseline,
                primary=bundle.primary_report,
                replay=bundle.primary_replay_report,
                diagnostic=bundle.diagnostic_report,
                lease=lease_a,
            )
            self.assertEqual(
                _load_baseline_bundle(
                    inputs=inputs,  # type: ignore[arg-type]
                    store=store,
                    suites=suites,
                    baseline=bundle.baseline,
                    lease=lease_a,
                ),
                published,
            )

            lease_b = ledger_b.claim_baseline_batch(sealed_b.authorization)
            ledger_b.begin_baseline_batch(lease_b)
            self.assertIsNone(
                _load_baseline_bundle(
                    inputs=inputs,  # type: ignore[arg-type]
                    store=store,
                    suites=suites,
                    baseline=bundle.baseline,
                    lease=lease_b,
                )
            )

    def test_failed_execution_is_terminal_and_not_retried(self) -> None:
        sealed, ledger = _sealed()
        suites = _suites()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            worker = root / "worker"
            library = root / "library.so"
            worker.touch()
            library.touch()
            inputs = _inputs(root)
            with (
                mock.patch.object(
                    PairedEvaluationSuites,
                    "build",
                    return_value=suites,
                ),
                mock.patch(
                    "irisu_rl.r3b_baselines._scripted_report",
                    side_effect=RuntimeError("simulator failed"),
                ),
                mock.patch("irisu_rl.r3b_baselines.IrisuEnv"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulator failed"):
                    run_sealed_baselines(
                        inputs,  # type: ignore[arg-type]
                        store,
                        sealed,
                        exact_worker_path=worker,
                        portable_library_path=library,
                    )
                with self.assertRaisesRegex(RuntimeError, "not active"):
                    run_sealed_baselines(
                        inputs,  # type: ignore[arg-type]
                        store,
                        sealed,
                        exact_worker_path=worker,
                        portable_library_path=library,
                    )
            self.assertFalse(
                ledger.verify_completed_baseline_batch(
                    sealed.authorization,
                    tuple(item.evidence().sha256 for item in valid_baseline_artifacts(TEST_PLAN)),
                )
            )


if __name__ == "__main__":
    unittest.main()
