from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from irisu_env import ActionKind
from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.curriculum import SnapshotBlobStore, SnapshotLibrary
from irisu_rl.r3b_snapshots import (
    GENERATOR_VERSION,
    SnapshotBundle,
    SnapshotIntent,
    SnapshotSourcePlan,
    SnapshotSourceManifest,
    generate_snapshot_bundle,
    load_snapshot_bundle,
    pair_snapshot_bundles,
)


_STATE = struct.Struct("<qqQ")


class FakeSnapshotSimulator:
    library_path = str(Path(__file__).resolve())
    build_info = {
        "physics_backend": "portable-test-r58",
        "snapshot_schema": 7,
        "clone_version": "r3b-snapshot-pipeline-test-v1",
    }

    def __init__(self) -> None:
        self.tick = 0
        self.score = 0
        self.seed = 0
        self.terminal = False
        self.invalid = False
        self.unstable_clone = False
        self.clone_calls = 0

    def _observation(self):
        return {
            "tick": self.tick,
            "score": self.score,
            "terminated": self.terminal,
            "truncated": False,
        }

    def reset(self, *, seed: int):
        self.tick = 0
        self.score = 0
        self.seed = seed
        self.terminal = False
        return self._observation(), {"seed": seed}

    def step(self, action):
        delta = int(action.wait_ticks) if action.kind is ActionKind.WAIT else 1
        self.tick += delta
        self.score += delta
        self.terminal = self.terminal or self.tick >= 100
        return (
            self._observation(),
            delta,
            self.terminal,
            False,
            {"invalid_action": self.invalid},
        )

    def config(self):
        return {"mode": "snapshot-pipeline-test"}

    def config_hash(self):
        return 71

    def state_hash(self):
        return self.seed * 1_000 + self.tick

    def clone_state(self):
        self.clone_calls += 1
        salt = self.clone_calls if self.unstable_clone else 0
        return _STATE.pack(self.tick, self.score, self.state_hash() + salt)


def _source(*, wait: int = 3) -> SnapshotSourceManifest:
    action_spec = ActionSpec()
    return SnapshotSourceManifest(
        "unit-source",
        action_spec.sha256,
        (
            SnapshotIntent(
                "train-a",
                "stage",
                "train",
                "family-train",
                "portable-pool",
                11,
                (action_spec.serialize(SemanticAction.wait(wait)).hex(),),
            ),
            SnapshotIntent(
                "validation-a",
                "stage",
                "validation",
                "family-validation",
                "portable-pool",
                23,
                (action_spec.serialize(SemanticAction.wait(min(wait + 1, 100))).hex(),),
            ),
        ),
    )


class R3BSnapshotPipelineTests(unittest.TestCase):
    def test_backend_pairing_uses_logical_provenance_not_native_state(self) -> None:
        action_spec = ActionSpec()
        trace = (action_spec.serialize(SemanticAction.wait(3)).hex(),)
        source = SnapshotSourceManifest(
            "pair-portable",
            action_spec.sha256,
            tuple(
                SnapshotIntent(
                    f"portable-{split}",
                    "stage",
                    split,
                    f"family-{split}",
                    "portable-pool",
                    index + 1,
                    trace,
                )
                for index, split in enumerate(
                    ("train", "calibration", "validation", "test")
                )
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            portable = generate_snapshot_bundle(
                FakeSnapshotSimulator(),
                source,
                Path(directory) / "portable",
            )
        exact_source = replace(
            source,
            source_id="pair-exact",
            intents=tuple(
                replace(
                    intent,
                    snapshot_id=intent.snapshot_id.replace("portable-", "exact-"),
                    environment_pool="exact-pool",
                )
                for intent in source.intents
            ),
        )
        exact_blobs: dict[str, bytes] = {}
        exact_recipes = []
        for portable_recipe, exact_intent in zip(
            portable.library.recipes,
            sorted(exact_source.intents, key=lambda value: value.snapshot_id),
            strict=True,
        ):
            payload = b"exact:" + portable.store[portable_recipe.snapshot_id]
            exact_blobs[exact_intent.snapshot_id] = payload
            exact_recipes.append(
                replace(
                    portable_recipe,
                    snapshot_id=exact_intent.snapshot_id,
                    environment_pool=exact_intent.environment_pool,
                    expected_state_hash=portable_recipe.expected_state_hash + 1,
                    snapshot_sha256=hashlib.sha256(payload).hexdigest(),
                    runtime_identity_sha256="e" * 64,
                )
            )
        exact_library = SnapshotLibrary(exact_recipes)
        exact = SnapshotBundle(
            exact_source,
            exact_library,
            SnapshotBlobStore(exact_library, exact_blobs),
            "exact",
            "e" * 64,
        )

        paired = pair_snapshot_bundles(portable, exact)

        self.assertEqual(set(paired), {"calibration", "validation", "test"})
        for split, manifest in paired.items():
            self.assertEqual(len(manifest.pairs), 1)
            pair = manifest.pairs[0]
            self.assertEqual(pair.logical_cell.split, split)
            self.assertNotEqual(
                exact.library[pair.exact_snapshot_id].expected_state_hash,
                portable.library[pair.portable_snapshot_id].expected_state_hash,
            )

    def test_canonical_plan_materializes_disjoint_deterministic_sources(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "rl"
            / "snapshots"
            / "r3b-source-plan-v1.toml"
        )
        plan = SnapshotSourcePlan.from_toml(path)
        portable = plan.materialize("portable")
        exact = plan.materialize("exact")
        replay = plan.materialize("portable")
        self.assertEqual(portable, replay)
        self.assertEqual(
            Counter(value.split for value in portable.intents),
            {"train": 64, "calibration": 64, "validation": 512, "test": 512},
        )
        self.assertEqual(
            len({value.reset_seed for value in portable.intents}),
            len(portable.intents),
        )
        families = {
            split: {
                value.scenario_family
                for value in portable.intents
                if value.split == split
            }
            for split in ("train", "calibration", "validation", "test")
        }
        for split, values in families.items():
            self.assertFalse(
                values
                & set().union(
                    *(other for name, other in families.items() if name != split)
                )
            )
        action_spec = ActionSpec()
        for portable_intent, exact_intent in zip(
            portable.intents, exact.intents, strict=True
        ):
            self.assertNotEqual(portable_intent.snapshot_id, exact_intent.snapshot_id)
            self.assertNotEqual(
                portable_intent.environment_pool, exact_intent.environment_pool
            )
            self.assertEqual(
                replace(
                    portable_intent,
                    snapshot_id=exact_intent.snapshot_id,
                    environment_pool=exact_intent.environment_pool,
                ),
                exact_intent,
            )
            for payload in portable_intent.semantic_actions_hex:
                action_spec.validate(action_spec.deserialize(bytes.fromhex(payload)))

    def test_source_plan_mapping_is_strict(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "rl"
            / "snapshots"
            / "r3b-source-plan-v1.toml"
        )
        plan = SnapshotSourcePlan.from_toml(path)
        self.assertEqual(
            SnapshotSourcePlan.from_mapping(plan.manifest()).sha256, plan.sha256
        )
        with self.assertRaisesRegex(ValueError, "keys differ"):
            SnapshotSourcePlan.from_mapping({**plan.manifest(), "extra": 1})
        duplicate_namespaces = plan.manifest()
        duplicate_namespaces["splits"]["test"]["scenario_family_namespace"] = (
            duplicate_namespaces["splits"]["validation"]["scenario_family_namespace"]
        )
        with self.assertRaisesRegex(ValueError, "namespaces must be disjoint"):
            SnapshotSourcePlan.from_mapping(duplicate_namespaces)
        with self.assertRaisesRegex(ValueError, "portable or exact"):
            plan.materialize("other")

    def test_generation_publishes_exact_verified_bundle_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            simulator = FakeSnapshotSimulator()
            generated = generate_snapshot_bundle(simulator, _source(), output)

            self.assertEqual(
                {value.name for value in output.iterdir()},
                {"source.json", "library.json", "bundle.json", "snapshots"},
            )
            self.assertEqual(
                {value.name for value in (output / "snapshots").iterdir()},
                {"train-a.snapshot", "validation-a.snapshot"},
            )
            loaded = load_snapshot_bundle(output, simulator)
            self.assertEqual(loaded.sha256, generated.sha256)
            self.assertEqual(loaded.source.generator_version, GENERATOR_VERSION)
            train = loaded.library["train-a"]
            self.assertEqual(train.expected_tick, 3)
            self.assertEqual(train.expected_score, 3)
            self.assertEqual(train.expected_state_hash, 11_003)
            self.assertEqual(
                hashlib.sha256(loaded.store["train-a"]).hexdigest(),
                train.snapshot_sha256,
            )

    def test_source_manifest_is_strict_versioned_and_canonical(self) -> None:
        source = _source()
        manifest = source.manifest()
        self.assertEqual(
            SnapshotSourceManifest.from_manifest(manifest).sha256, source.sha256
        )
        with self.assertRaisesRegex(ValueError, "keys differ"):
            SnapshotSourceManifest.from_manifest({**manifest, "extra": True})
        bad_intent = {
            **manifest["intents"][0],
            "semantic_actions_hex": [
                " " + str(manifest["intents"][0]["semantic_actions_hex"][0])
            ],
        }
        with self.assertRaisesRegex(ValueError, "canonical lowercase"):
            SnapshotIntent.from_manifest(bad_intent)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                SnapshotSourceManifest.from_json(path)

    def test_generation_rejects_unsafe_or_nonempty_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nonempty = root / "nonempty"
            nonempty.mkdir()
            (nonempty / "owned.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                generate_snapshot_bundle(FakeSnapshotSimulator(), _source(), nonempty)
            self.assertEqual((nonempty / "owned.txt").read_text(), "keep")

            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                generate_snapshot_bundle(FakeSnapshotSimulator(), _source(), link)

    def test_failures_publish_nothing_and_clean_staging(self) -> None:
        cases = (
            ("invalid", {"invalid": True}, "invalid native action"),
            ("terminal", {}, "live decision boundary"),
            ("unstable", {"unstable_clone": True}, "replay identity"),
        )
        for name, changes, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                simulator = FakeSnapshotSimulator()
                for key, value in changes.items():
                    setattr(simulator, key, value)
                source = _source(wait=100) if name == "terminal" else _source()
                output = Path(directory) / "bundle"
                with self.assertRaisesRegex(ValueError, message):
                    generate_snapshot_bundle(simulator, source, output)
                self.assertFalse(output.exists())
                self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_loader_rejects_tampering_and_symlinked_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            simulator = FakeSnapshotSimulator()
            generate_snapshot_bundle(simulator, _source(), output)
            blob = output / "snapshots" / "train-a.snapshot"
            blob.write_bytes(blob.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "blob hash mismatch"):
                load_snapshot_bundle(output, simulator)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            simulator = FakeSnapshotSimulator()
            generate_snapshot_bundle(simulator, _source(), output)
            blob = output / "snapshots" / "train-a.snapshot"
            replacement = Path(directory) / "owned.snapshot"
            replacement.write_bytes(blob.read_bytes())
            blob.unlink()
            blob.symlink_to(replacement)
            with self.assertRaisesRegex(ValueError, "exactly the library blobs"):
                load_snapshot_bundle(output, simulator)

    def test_empty_output_directory_is_replaced_only_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            output.mkdir()
            source = _source()
            generated = generate_snapshot_bundle(
                FakeSnapshotSimulator(), source, output
            )
            self.assertEqual(
                load_snapshot_bundle(output, FakeSnapshotSimulator()).sha256,
                generated.sha256,
            )

            second_source = replace(source, source_id="different-source")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                generate_snapshot_bundle(FakeSnapshotSimulator(), second_source, output)


if __name__ == "__main__":
    unittest.main()
