#!/usr/bin/env python3
"""Focused fail-closed tests for the Generation-02 campaign chain."""

from __future__ import annotations

import hashlib
import importlib
import json
import io
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "benchmarks", ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ["IRISU_G2_UNFROZEN_AUDIT"] = "1"
try:
    import r3h_g2_exact_collect as collector  # noqa: E402
    import rl_r3h_g2 as campaign  # noqa: E402
finally:
    os.environ.pop("IRISU_G2_UNFROZEN_AUDIT", None)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class AuthorizationFixture:
    source_sha = "a" * 64
    dataset_sha = "b" * 64
    collection_sha = {
        "train-board": "1" * 64,
        "support-board": "2" * 64,
        "margin-board": "3" * 64,
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.model = root / "model"
        self.paths = {
            "EXPERIMENT_ROOT": root,
            "SOURCE_IDENTITY_PATH": root / "source-identity.json",
            "PREFLIGHT_PATH": root / "preflight.json",
            "PREFLIGHT_RECEIPT_PATH": root / "preflight.receipt.json",
            "PILOT_PATH": root / "pilot.json",
            "PILOT_RECEIPT_PATH": root / "pilot.receipt.json",
            "MODEL_ROOT": self.model,
            "TRAINING_PATH": self.model / "training.json",
            "CHECKPOINT_PATH": self.model / "resolution-first-g2.pt",
            "CHECKPOINT_RECEIPT_PATH": self.model / "checkpoint.receipt.json",
            "SUPPORT_PATH": self.model / "support-calibration.json",
            "SUPPORT_RECEIPT_PATH": self.model
            / "support-calibration.receipt.json",
            "MARGIN_PATH": self.model / "margin-calibration.json",
            "MARGIN_RECEIPT_PATH": self.model
            / "margin-calibration.receipt.json",
        }

    def patch(self) -> ExitStack:
        stack = ExitStack()
        for name, value in self.paths.items():
            stack.enter_context(mock.patch.object(collector, name, value))
        stack.enter_context(
            mock.patch.object(
                collector,
                "_verify_source_identity",
                return_value=({}, self.source_sha),
            )
        )
        stack.enter_context(
            mock.patch.object(
                collector,
                "_collection_identity_sha256",
                side_effect=lambda split: self.collection_sha[split],
            )
        )
        return stack

    def _receipt(
        self,
        artifact: Path,
        receipt: Path,
        report: dict[str, object],
        schema: str,
        **extra: object,
    ) -> None:
        _write(artifact, report)
        _write(
            receipt,
            {
                "schema": schema,
                "artifact": str(artifact),
                "artifact_sha256": collector._sha256_file(artifact),
                "source_identity_sha256": self.source_sha,
                **extra,
            },
        )

    def preflight(self) -> None:
        hashes: dict[str, str] = {}
        for split, count in collector.SPLIT_COUNTS.items():
            path = self.root / "seed-manifests" / f"{split}.json"
            _write(
                path,
                {
                    "schema": "irisu-r3h-g2-seed-manifest-v1",
                    "experiment_id": collector.EXPERIMENT_ID,
                    "split": split,
                    "rows": [
                        {
                            "index": index,
                            "seed": collector.derive_seed(split, index),
                        }
                        for index in range(count)
                    ],
                },
            )
            hashes[split] = collector._sha256_file(path)
        self._receipt(
            self.paths["PREFLIGHT_PATH"],
            self.paths["PREFLIGHT_RECEIPT_PATH"],
            {
                "schema": "irisu-r3h-g2-preflight-v1",
                "development_only": True,
                "sealed_test_allowed": False,
                "source_identity_sha256": self.source_sha,
                "seed_manifest_sha256": hashes,
            },
            "irisu-r3h-g2-preflight-receipt-v1",
        )

    def pilot_checkpoint(self) -> None:
        self._receipt(
            self.paths["PILOT_PATH"],
            self.paths["PILOT_RECEIPT_PATH"],
            {
                "schema": "irisu-r3h-g2-structural-pilot-v1",
                "development_only": True,
                "sealed_test_allowed": False,
                "source_identity_sha256": self.source_sha,
                "dataset_sha256": self.dataset_sha,
                "collection_identity_sha256": self.collection_sha[
                    "train-board"
                ],
                "passed": True,
            },
            "irisu-r3h-g2-pilot-receipt-v1",
            passed=True,
            dataset_sha256=self.dataset_sha,
            collection_identity_sha256=self.collection_sha["train-board"],
        )
        checkpoint = self.paths["CHECKPOINT_PATH"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"finite checkpoint")
        checkpoint_sha = collector._sha256_file(checkpoint)
        manifest = {"schema": "synthetic-g2-model"}
        self._receipt(
            self.paths["TRAINING_PATH"],
            self.paths["CHECKPOINT_RECEIPT_PATH"],
            {
                "schema": "irisu-r3h-g2-training-v1",
                "development_only": True,
                "sealed_test_allowed": False,
                "source_identity_sha256": self.source_sha,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "dataset_sha256": self.dataset_sha,
                "collection_identity_sha256": self.collection_sha[
                    "train-board"
                ],
                "model_manifest": manifest,
            },
            "irisu-r3h-g2-checkpoint-receipt-v1",
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,
            dataset_sha256=self.dataset_sha,
            collection_identity_sha256=self.collection_sha["train-board"],
            model_manifest_sha256=collector._canonical_sha256(manifest),
        )

    def support(self, *, passed: bool = True) -> None:
        checkpoint_sha = collector._sha256_file(
            self.paths["CHECKPOINT_PATH"]
        )
        auroc = {"passed": passed, "auroc": 0.8 if passed else 0.7}
        support = {
            "threshold": 0.5,
            "resolution_auroc": auroc["auroc"],
            "grid": [{"threshold": 0.5, "passed": passed}],
        }
        report = {
            "schema": "irisu-r3h-g2-support-calibration-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": self.source_sha,
            "checkpoint_sha256": checkpoint_sha,
            "support_dataset_sha256": "c" * 64,
            "support_collection_identity_sha256": self.collection_sha[
                "support-board"
            ],
            "resolution_auroc": auroc,
            "support_grid": [{"threshold": 0.5, "passed": passed}],
            "support": support if passed else None,
            "passed": passed,
        }
        self._receipt(
            self.paths["SUPPORT_PATH"],
            self.paths["SUPPORT_RECEIPT_PATH"],
            report,
            "irisu-r3h-g2-support-calibration-receipt-v1",
            checkpoint_sha256=checkpoint_sha,
            passed=passed,
            support_dataset_sha256="c" * 64,
            support_collection_identity_sha256=self.collection_sha[
                "support-board"
            ],
            resolution_auroc_sha256=collector._canonical_sha256(auroc),
            support_sha256=(
                collector._canonical_sha256(support) if passed else None
            ),
        )

    def margin(self, *, q: float | str = 1.0, passed: bool = True) -> None:
        support_receipt = json.loads(
            self.paths["SUPPORT_RECEIPT_PATH"].read_text()
        )
        checkpoint_sha = collector._sha256_file(
            self.paths["CHECKPOINT_PATH"]
        )
        selective = {
            "q": q,
            "support_calibration_sha256": support_receipt["support_sha256"],
        }
        support_receipt_sha = collector._sha256_file(
            self.paths["SUPPORT_RECEIPT_PATH"]
        )
        report = {
            "schema": "irisu-r3h-g2-margin-calibration-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": self.source_sha,
            "checkpoint_sha256": checkpoint_sha,
            "support_calibration_receipt_sha256": support_receipt_sha,
            "support_calibration_sha256": support_receipt["support_sha256"],
            "margin_dataset_sha256": "d" * 64,
            "margin_collection_identity_sha256": self.collection_sha[
                "margin-board"
            ],
            "calibration_target": "absolute_b2",
            "selective": selective,
            "passed": passed,
        }
        self._receipt(
            self.paths["MARGIN_PATH"],
            self.paths["MARGIN_RECEIPT_PATH"],
            report,
            "irisu-r3h-g2-margin-calibration-receipt-v1",
            checkpoint_sha256=checkpoint_sha,
            passed=passed,
            finite_q_b2=passed,
            support_calibration_receipt_sha256=support_receipt_sha,
            support_calibration_sha256=support_receipt["support_sha256"],
            margin_dataset_sha256="d" * 64,
            margin_collection_identity_sha256=self.collection_sha[
                "margin-board"
            ],
            calibration_target="absolute_b2",
            selective_sha256=collector._canonical_sha256(selective),
        )


class G2CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.chain = AuthorizationFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_every_split_rejects_without_its_exact_authorization(self) -> None:
        with self.chain.patch():
            for split in collector.SPLIT_COUNTS:
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    collector._authorize_collection(split)
            self.chain.preflight()
            self.assertEqual(
                collector._authorize_collection("train-board")["split"],
                "train-board",
            )
            for split in ("support-board", "margin-board", "offline-board"):
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    collector._authorize_collection(split)

    def test_authorization_chain_opens_only_in_order(self) -> None:
        with self.chain.patch():
            self.chain.preflight()
            self.chain.pilot_checkpoint()
            self.assertEqual(
                collector._authorize_collection("support-board")["split"],
                "support-board",
            )
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                collector._authorize_collection("margin-board")
            self.chain.support()
            self.assertEqual(
                collector._authorize_collection("margin-board")["split"],
                "margin-board",
            )
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                collector._authorize_collection("offline-board")
            self.chain.margin()
            self.assertEqual(
                collector._authorize_collection("offline-board")["split"],
                "offline-board",
            )

    def test_receipt_tamper_rejects_before_external_loader(self) -> None:
        with self.chain.patch():
            self.chain.preflight()
            self.chain.pilot_checkpoint()
            self.chain.support()
            receipt = json.loads(
                self.chain.paths["SUPPORT_RECEIPT_PATH"].read_text()
            )
            receipt["artifact_sha256"] = "0" * 64
            _write(self.chain.paths["SUPPORT_RECEIPT_PATH"], receipt)
            loader = mock.Mock(side_effect=AssertionError("science opened"))
            with mock.patch.object(collector, "_load_frozen_b", loader):
                with self.assertRaises(RuntimeError):
                    collector.collect_episode("margin-board", 0)
            loader.assert_not_called()

    def test_collect_episode_rejects_module_injection(self) -> None:
        fake_core = mock.Mock()
        fake_campaign = mock.Mock()
        with self.assertRaises(TypeError):
            collector.collect_episode(
                "train-board",
                0,
                core=fake_core,
                campaign=fake_campaign,
                identities={},
            )
        fake_core.assert_not_called()
        fake_campaign.assert_not_called()

    def test_public_observation_canonicalizes_numpy_scalars(self) -> None:
        gym_body_sequence = getattr(
            importlib.import_module("irisu_env.env"),
            "_GymBodySequence",
        )
        observation = {
            "tick": campaign.np.uint64(9),
            "score": campaign.np.int64(17),
            "bodies": gym_body_sequence((
                {
                    "id": campaign.np.uint32(4),
                    "age_ticks": campaign.np.uint64(8),
                    "x": campaign.np.float64(1.25),
                },
            )),
        }
        numpy_scalar_types = tuple(
            scalar_type
            for scalar_type in campaign.np.ScalarType
            if isinstance(scalar_type, type)
            and issubclass(scalar_type, campaign.np.generic)
            and campaign.np.dtype(scalar_type).kind in "biuf"
        )
        with (
            mock.patch.object(
                collector, "_NUMPY_GENERIC_TYPE", campaign.np.generic
            ),
            mock.patch.object(
                collector, "_NUMPY_SCALAR_TYPES", numpy_scalar_types
            ),
            mock.patch.object(
                collector,
                "_GYM_BODY_SEQUENCE_TYPE",
                gym_body_sequence,
            ),
        ):
            value, payload, body_ids = collector._canonical_observation(
                observation
            )
            self.assertEqual(body_ids, [4])
            self.assertEqual(value["tick"], 9)
            self.assertEqual(value["score"], 17)
            self.assertEqual(value["bodies"][0]["age_ticks"], 8)
            self.assertEqual(value["bodies"][0]["x"], 1.25)
            self.assertEqual(
                payload,
                collector._canonical_bytes(value),
            )
            self.assertEqual(
                collector._canonical_public_value((1, 2)),
                [1, 2],
            )
            self.assertEqual(
                collector._canonical_public_value([1, 2]),
                [1, 2],
            )
            with self.assertRaisesRegex(TypeError, "string keys"):
                collector._canonical_public_value({1: "foreign"})

            class IntSubclass(int):
                pass

            class ArbitraryItem:
                def item(self) -> list[int]:
                    return [1, 2]

            class ClassSpoof:
                @property
                def __class__(self) -> type:
                    return campaign.np.int64

                def item(self) -> int:
                    return 7

            class MappingSpoof:
                @property
                def __class__(self) -> type:
                    return dict

                def __iter__(self):
                    return iter(("safe",))

                def items(self):
                    return ((1, 5),)

            class SequenceSpoof:
                @property
                def __class__(self) -> type:
                    return list

                def __len__(self) -> int:
                    return 1

                def __getitem__(self, index: int) -> int:
                    if index == 0:
                        return 9
                    raise IndexError

            class EvilFloat(campaign.np.float64):
                def item(self) -> int:
                    return 777

            class NativeTypeEqualitySpoof(type):
                def __eq__(cls, other: object) -> bool:
                    return other is int

                __hash__ = type.__hash__

            class EvilInt(int, metaclass=NativeTypeEqualitySpoof):
                pass

            class NumpyTypeEqualitySpoof(type):
                def __eq__(cls, other: object) -> bool:
                    return other is campaign.np.int64

                def __hash__(cls) -> int:
                    return hash(campaign.np.int64)

            class EvilNumpyInt(
                campaign.np.int64,
                metaclass=NumpyTypeEqualitySpoof,
            ):
                pass

            class ScalarTupleSubclass(tuple):
                def __iter__(self):
                    return iter((EvilNumpyInt,))

            class VirtualMapping:
                def __getitem__(self, key: str) -> int:
                    if key == "safe":
                        return 5
                    raise KeyError(key)

                def __iter__(self):
                    return iter(("safe",))

                def __len__(self) -> int:
                    return 1

                def items(self):
                    return (("safe", 5),)

            class VirtualSequence:
                def __len__(self) -> int:
                    return 2

                def __getitem__(self, index: int) -> int:
                    if 0 <= index < 2:
                        return index + 7
                    raise IndexError

            collector.Mapping.register(VirtualMapping)
            collector.Sequence.register(VirtualSequence)

            class DictSubclass(dict):
                pass

            class ListSubclass(list):
                pass

            for foreign in (
                IntSubclass(4),
                memoryview(b"x"),
                ArbitraryItem(),
                ClassSpoof(),
                MappingSpoof(),
                SequenceSpoof(),
                EvilFloat(1.25),
                EvilInt(7),
                EvilNumpyInt(7),
                VirtualMapping(),
                VirtualSequence(),
                DictSubclass(safe=5),
                ListSubclass((7, 8)),
            ):
                with self.subTest(type=type(foreign).__name__):
                    with self.assertRaisesRegex(TypeError, "unsupported"):
                        collector._canonical_public_value(foreign)

            with mock.patch.object(
                collector,
                "_NUMPY_SCALAR_TYPES",
                ScalarTupleSubclass(numpy_scalar_types),
            ):
                with self.assertRaisesRegex(TypeError, "unsupported"):
                    collector._canonical_public_value(EvilNumpyInt(7))

            class InconsistentMapping(collector.Mapping):
                def __getitem__(self, key: str) -> int:
                    if key == "safe":
                        return 5
                    raise KeyError(key)

                def __iter__(self):
                    return iter(("safe",))

                def __len__(self) -> int:
                    return 1

                def items(self):
                    return ((1, 5),)

            with self.assertRaisesRegex(TypeError, "unsupported"):
                collector._canonical_public_value(InconsistentMapping())

    def test_strategy_environment_modules_cannot_escape_closed_root(self) -> None:
        collector._load_closed_irisu_env()
        expected_root = (ROOT / "python/irisu_env").resolve()
        for name, module in sys.modules.items():
            if name == "irisu_env" or name.startswith("irisu_env."):
                self.assertEqual(
                    Path(str(module.__file__)).resolve().parent,
                    expected_root,
                )

        class ForeignGymBodySequence(tuple):
            pass

        with mock.patch.object(
            collector,
            "_GYM_BODY_SEQUENCE_TYPE",
            ForeignGymBodySequence,
        ):
            with self.assertRaisesRegex(RuntimeError, "type changed"):
                collector._load_closed_irisu_env()
        foreign = self.root / "foreign/render.py"
        foreign.parent.mkdir(parents=True)
        foreign.write_text("VALUE = 1\n")
        with mock.patch.dict(
            sys.modules,
            {"irisu_env.render": SimpleNamespace(__file__=str(foreign))},
        ):
            with self.assertRaisesRegex(RuntimeError, "foreign preloaded"):
                collector._load_closed_irisu_env()

    def test_strategy_loader_requires_exact_closed_sys_path(self) -> None:
        with mock.patch.object(
            collector.sys,
            "path",
            [*collector.CLOSED_RUNTIME_SYS_PATH, "/foreign/worktree/python"],
        ):
            with self.assertRaisesRegex(RuntimeError, "import path escaped"):
                collector._validate_closed_sys_path()

    def test_indirect_partial_is_rejected_before_read(self) -> None:
        identity: dict[str, object] = {}
        identity_sha = collector._canonical_sha256(identity)
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                directory = self.root / kind
                episode = directory / "episode.json"
                query = directory / "episode.queries.jsonl"
                target = directory / "foreign"
                directory.mkdir(parents=True)
                target.write_bytes(b"not-json")
                if kind == "symlink":
                    episode.symlink_to(target)
                else:
                    episode.hardlink_to(target)
                with self.assertRaisesRegex(
                    RuntimeError, "indirect|foreign.*links"
                ):
                    collector._load_complete(
                        episode,
                        query,
                        identity=identity,
                        identity_sha256=identity_sha,
                    )

    def test_collector_recovers_and_quarantines_writer_temps(self) -> None:
        output = self.root / "collection"
        directory = output / "train-board"
        directory.mkdir(parents=True)

        prelink = directory / "prelink.json"
        prelink_temp = directory / ".prelink.json.tmp-123-456"
        prelink_temp.write_bytes(b"partial")
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            collector._recover_writer_temps(prelink)
        self.assertFalse(prelink.exists())
        self.assertFalse(prelink_temp.exists())
        preserved = list(
            (output / "_incomplete/_atomic/train-board").iterdir()
        )
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"partial")

        interrupted = directory / "interrupted.json"
        interrupted_temp = directory / ".interrupted.json.tmp-321-654"
        interrupted_temp.write_bytes(b"interrupted")
        interrupted_digest = collector._sha256_file(interrupted_temp)
        interrupted_quarantine = (
            output
            / "_incomplete/_atomic/train-board"
            / f"{interrupted.name}.{interrupted_digest}.321.654.0"
        )
        interrupted_quarantine.hardlink_to(interrupted_temp)
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            collector._recover_writer_temps(interrupted)
        self.assertFalse(interrupted_temp.exists())
        self.assertEqual(interrupted_quarantine.stat().st_nlink, 1)
        self.assertEqual(interrupted_quarantine.read_bytes(), b"interrupted")

        postlink = directory / "postlink.json"
        postlink_temp = directory / ".postlink.json.tmp-123-456"
        postlink_temp.write_bytes(b"complete")
        postlink.hardlink_to(postlink_temp)
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            collector._recover_writer_temps(postlink)
        self.assertTrue(postlink.is_file())
        self.assertEqual(postlink.stat().st_nlink, 1)
        self.assertFalse(postlink_temp.exists())

        unknown = directory / "unknown.json"
        (directory / ".unknown.json.foreign").write_bytes(b"x")
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                collector._recover_writer_temps(unknown)

    def test_partial_quarantine_recovers_after_link_boundary(self) -> None:
        output = self.root / "partial-recovery/collection"
        source = output / "train-board/episode.json"
        source.parent.mkdir(parents=True)
        source.write_bytes(b'{"collector_identity_sha256":"x"}\n')
        identity = "a" * 64
        digest = collector._sha256_file(source)
        quarantine = output / "_incomplete/train-board"
        quarantine.mkdir(parents=True)
        linked = quarantine / (
            f"{source.name}.{identity}.{digest}.123.456.0"
        )
        linked.hardlink_to(source)
        self.assertEqual(source.stat().st_nlink, 2)
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            recovered = collector._quarantine_partial(
                source, identity_sha256=identity
            )
        self.assertEqual(recovered, linked)
        self.assertFalse(source.exists())
        self.assertEqual(linked.stat().st_nlink, 1)
        self.assertEqual(linked.read_bytes(), b'{"collector_identity_sha256":"x"}\n')

        foreign_source = output / "train-board/foreign.json"
        foreign_source.write_bytes(b"foreign")
        foreign_digest = collector._sha256_file(foreign_source)
        foreign_link = quarantine / f"wrong.{foreign_digest}.123.456.0"
        foreign_link.hardlink_to(foreign_source)
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            with self.assertRaisesRegex(RuntimeError, "foreign"):
                collector._quarantine_partial(
                    foreign_source, identity_sha256=identity
                )

    def test_load_complete_finishes_interrupted_partial_quarantine(self) -> None:
        output = self.root / "partial-load/collection"
        episode = output / "train-board/episode.json"
        query = output / "train-board/episode.queries.jsonl"
        query.parent.mkdir(parents=True)
        identity: dict[str, object] = {}
        identity_sha = collector._canonical_sha256(identity)
        query.write_text(
            json.dumps({"collector_identity_sha256": identity_sha}) + "\n"
        )
        digest = collector._sha256_file(query)
        quarantine = output / "_incomplete/train-board"
        quarantine.mkdir(parents=True)
        linked = quarantine / (
            f"{query.name}.{identity_sha}.{digest}.123.456.0"
        )
        linked.hardlink_to(query)
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            self.assertIsNone(
                collector._load_complete(
                    episode,
                    query,
                    identity=identity,
                    identity_sha256=identity_sha,
                )
            )
        self.assertFalse(query.exists())
        self.assertEqual(linked.stat().st_nlink, 1)

    def test_empty_query_partial_is_foreign_and_not_quarantined(self) -> None:
        output = self.root / "empty-partial/collection"
        episode = output / "train-board/episode.json"
        query = output / "train-board/episode.queries.jsonl"
        query.parent.mkdir(parents=True)
        query.write_bytes(b"")
        identity: dict[str, object] = {}
        identity_sha = collector._canonical_sha256(identity)
        with mock.patch.object(collector, "OUTPUT_ROOT", output):
            with self.assertRaisesRegex(RuntimeError, "foreign partial"):
                collector._load_complete(
                    episode,
                    query,
                    identity=identity,
                    identity_sha256=identity_sha,
                )
        self.assertTrue(query.is_file())
        self.assertFalse((output / "_incomplete").exists())

    def test_collect_episode_precheck_accepts_exact_partial_link_boundary(self) -> None:
        output = self.root / "partial-entrypoint/collection"
        split = output / "train-board"
        split.mkdir(parents=True)
        identity_sha = "a" * 64
        with (
            mock.patch.object(collector, "OUTPUT_ROOT", output),
            mock.patch.object(collector, "SPLIT_COUNTS", {"train-board": 1}),
        ):
            seed = collector.derive_seed("train-board", 0)
            query = split / f"{seed:010d}.queries.jsonl"
            query.write_text(
                json.dumps(
                    {"collector_identity_sha256": identity_sha}
                )
                + "\n"
            )
            digest = collector._sha256_file(query)
            quarantine = output / "_incomplete/train-board"
            quarantine.mkdir(parents=True)
            linked = quarantine / (
                f"{query.name}.{identity_sha}.{digest}.123.456.0"
            )
            linked.hardlink_to(query)
            with (
                mock.patch.object(
                    collector,
                    "_load_g1",
                    return_value=SimpleNamespace(
                        _pin_one_cpu=mock.Mock(return_value=0)
                    ),
                ),
                mock.patch.object(collector, "_verify_external_constants"),
                mock.patch.object(collector, "_verify_source_identity"),
                mock.patch.object(
                    collector, "_authorize_collection", return_value={}
                ),
                mock.patch.object(
                    collector, "_load_frozen_b", return_value=(object(), object(), {})
                ),
                mock.patch.object(
                    collector,
                    "_collector_identity",
                    side_effect=RuntimeError("reached-identity"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "reached-identity"):
                    collector.collect_episode("train-board", 0)

    def test_collect_episode_pins_before_authorization_or_science(self) -> None:
        authorization = mock.Mock(
            side_effect=AssertionError("authorization opened")
        )
        loader = mock.Mock(side_effect=AssertionError("science opened"))
        g1 = SimpleNamespace(
            _pin_one_cpu=mock.Mock(side_effect=RuntimeError("cannot-pin"))
        )
        with (
            mock.patch.object(collector, "_load_g1", return_value=g1),
            mock.patch.object(
                collector, "_authorize_collection", authorization
            ),
            mock.patch.object(collector, "_load_frozen_b", loader),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot-pin"):
                collector.collect_episode("train-board", 0)
        authorization.assert_not_called()
        loader.assert_not_called()

    def test_collection_closure_rejects_every_foreign_entry(self) -> None:
        output = self.root / "closure/collection"
        split = output / "train-board"
        split.mkdir(parents=True)
        with (
            mock.patch.object(collector, "OUTPUT_ROOT", output),
            mock.patch.object(collector, "SPLIT_COUNTS", {"train-board": 1}),
        ):
            seed = collector.derive_seed("train-board", 0)
            episode = split / f"{seed:010d}.json"
            query = split / f"{seed:010d}.queries.jsonl"
            temporary = split / f".{episode.name}.tmp-123-456"
            temporary.write_bytes(b"episode")
            episode.hardlink_to(temporary)
            query.write_bytes(b"query")
            collector._validate_collection_split(
                "train-board",
                allow_writer_temps=True,
                require_complete=True,
            )
            temporary.unlink()
            collector._validate_collection_split(
                "train-board",
                allow_writer_temps=False,
                require_complete=True,
            )
            nested = split / "nested"
            nested.mkdir()
            with self.assertRaisesRegex(RuntimeError, "foreign"):
                collector._validate_collection_split(
                    "train-board",
                    allow_writer_temps=False,
                    require_complete=True,
                )
            nested.rmdir()
            (split / "unknown").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                collector._validate_collection_split(
                    "train-board",
                    allow_writer_temps=False,
                    require_complete=True,
                )

    def test_incomplete_closure_recomputes_quarantine_hash(self) -> None:
        output = self.root / "incomplete-closure/collection"
        quarantine = output / "_incomplete/train-board"
        quarantine.mkdir(parents=True)
        with (
            mock.patch.object(collector, "OUTPUT_ROOT", output),
            mock.patch.object(collector, "SPLIT_COUNTS", {"train-board": 1}),
        ):
            seed = collector.derive_seed("train-board", 0)
            name = (
                f"{seed:010d}.json."
                f"{'a' * 64}.{'0' * 64}.123.456.0"
            )
            (quarantine / name).write_bytes(b"hash-mismatch")
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                collector._validate_collection_incomplete()

        experiment = self.root / "campaign-incomplete"
        model = experiment / "model"
        quarantine = experiment / "_incomplete/_atomic/model"
        model.mkdir(parents=True)
        quarantine.mkdir(parents=True)
        payload = quarantine / "payload"
        payload.write_bytes(b"foreign-target")
        digest = campaign.sha256_file(payload)
        payload.rename(quarantine / f"evil.{digest}.123.456.0")
        with (
            mock.patch.object(campaign, "EXPERIMENT", experiment),
            mock.patch.object(campaign, "MODEL", model),
        ):
            with self.assertRaisesRegex(RuntimeError, "target"):
                campaign._validate_campaign_incomplete()

    def test_campaign_collection_identity_has_exact_directory_closure(self) -> None:
        output = self.root / "campaign-closure"
        split = output / "train-board"
        split.mkdir(parents=True)
        with (
            mock.patch.object(campaign, "COLLECTION", output),
            mock.patch.object(campaign, "SPLITS", {"train-board": 1}),
        ):
            seed = campaign.derive_seeds("train-board", 1)[0]
            (split / f"{seed:010d}.json").write_bytes(b"episode")
            (split / f"{seed:010d}.queries.jsonl").write_bytes(b"query")
            self.assertEqual(
                campaign._require_complete_collection("train-board"),
                (seed,),
            )
            (split / "nested").mkdir()
            with self.assertRaisesRegex(RuntimeError, "foreign"):
                campaign.collection_identity_sha256("train-board")

    def test_atomic_writers_reject_indirect_parent(self) -> None:
        direct = self.root / "writer-direct"
        indirect = self.root / "writer-indirect"
        direct.mkdir()
        indirect.symlink_to(direct, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "indirect"):
            campaign.write_once(indirect / "report.json", {"x": 1})
        with self.assertRaisesRegex(RuntimeError, "indirect"):
            collector._write_new(indirect / "episode.json", b"x")

    def test_recorded_dependency_hash_mutation_is_rejected(self) -> None:
        identity = campaign.source_identity()
        preregistration = self.root / "preregistration.json"
        _write(
            preregistration,
            {
                "experiment_id": campaign.EXPERIMENT_ID,
                "source_revision": campaign.SOURCE_REVISION,
                "development_only": True,
                "sealed_test_allowed": False,
                "outcomes_viewed_before_preregistration": False,
                "generation_01_outcomes_are_training_data": False,
                "generation_02_outcomes_collected_before_freeze": False,
                "status": collector.PREREGISTRATION_STATUS,
                "protocol_sha256": identity["files"]["protocol"]["sha256"],
                "frozen_source_sha256": {
                    name: row["sha256"]
                    for name, row in identity["files"].items()
                    if name != "preregistration"
                },
                "runtime_manifest": identity["runtime"],
            },
        )
        identity["files"]["preregistration"] = {
            "path": str(preregistration),
            "sha256": collector._sha256_file(preregistration),
        }
        identity_path = self.root / "source-identity.json"
        _write(identity_path, identity)
        with (
            mock.patch.object(collector, "SOURCE_IDENTITY_PATH", identity_path),
            mock.patch.object(collector, "EXPERIMENT_ROOT", self.root),
            mock.patch.object(
                collector, "_validate_all_incomplete", return_value=campaign
            ),
        ):
            collector._verify_source_identity()
            original_sha = collector._sha256_file

            def changed_lock(path: Path) -> str:
                if path.resolve() == (ROOT / "uv.lock").resolve():
                    return "0" * 64
                return original_sha(path)

            with mock.patch.object(
                collector, "_sha256_file", side_effect=changed_lock
            ):
                with self.assertRaises(RuntimeError):
                    collector._verify_source_identity()
            identity["files"]["uv_lock"]["sha256"] = "0" * 64
            _write(identity_path, identity)
            with self.assertRaises(RuntimeError):
                collector._verify_source_identity()

    def test_cross_namespace_scans_ignore_preloaded_shims(self) -> None:
        fake_collection_scan = mock.Mock()
        fake_collector = SimpleNamespace(
            __file__=collector.__file__,
            OUTPUT_ROOT=campaign.COLLECTION,
            _validate_collection_incomplete=fake_collection_scan,
        )
        with mock.patch.dict(
            sys.modules, {"r3h_g2_exact_collect": fake_collector}
        ):
            campaign._validate_all_incomplete(
                collector_sha256=campaign.sha256_file(
                    Path(collector.__file__).resolve()
                )
            )
        fake_collection_scan.assert_not_called()

        fake_campaign_scan = mock.Mock()
        fake_campaign = SimpleNamespace(
            __file__=campaign.__file__,
            EXPERIMENT=collector.EXPERIMENT_ROOT,
            _validate_campaign_incomplete=fake_campaign_scan,
        )
        with (
            mock.patch.dict(sys.modules, {"rl_r3h_g2": fake_campaign}),
            mock.patch.dict(
                os.environ, {"IRISU_G2_UNFROZEN_AUDIT": "1"}
            ),
        ):
            collector._validate_all_incomplete(
                campaign_sha256=collector._sha256_file(
                    Path(campaign.__file__).resolve()
                )
            )
        fake_campaign_scan.assert_not_called()

    def test_peer_validator_hash_is_checked_before_execution(self) -> None:
        with mock.patch.object(
            campaign.importlib.util, "spec_from_file_location"
        ) as loader:
            with self.assertRaisesRegex(RuntimeError, "before validator load"):
                campaign._validate_all_incomplete(collector_sha256="0" * 64)
        loader.assert_not_called()

        with mock.patch.object(
            collector.importlib.util, "spec_from_file_location"
        ) as loader:
            with self.assertRaisesRegex(RuntimeError, "before validator load"):
                collector._validate_all_incomplete(campaign_sha256="0" * 64)
        loader.assert_not_called()

    def test_collector_peer_load_is_anchored_to_preregistration(self) -> None:
        root = self.root / "peer-anchor"
        source_identity = root / "source-identity.json"
        preregistration = root / "preregistration.json"
        frozen_sha = "a" * 64
        altered_sha = "b" * 64
        _write(
            preregistration,
            {
                "experiment_id": collector.EXPERIMENT_ID,
                "source_revision": collector.SOURCE_REVISION,
                "development_only": True,
                "sealed_test_allowed": False,
                "status": collector.PREREGISTRATION_STATUS,
                "frozen_source_sha256": {"campaign": frozen_sha},
            },
        )
        _write(
            source_identity,
            {
                "files": {
                    "campaign": {
                        "path": str(
                            (
                                ROOT / "benchmarks/rl_r3h_g2.py"
                            ).resolve()
                        ),
                        "sha256": altered_sha,
                    }
                }
            },
        )
        with (
            mock.patch.object(
                collector, "SOURCE_IDENTITY_PATH", source_identity
            ),
            mock.patch.object(collector, "EXPERIMENT_ROOT", root),
            mock.patch.object(
                collector,
                "_sha256_file",
                return_value=altered_sha,
            ),
            mock.patch.object(
                collector.importlib.util, "spec_from_file_location"
            ) as loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "before validator load"):
                collector._verify_source_identity()
        loader.assert_not_called()

    def test_runtime_tree_identity_is_exact_and_ignores_caches(self) -> None:
        root = self.root / "runtime-tree"
        root.mkdir()
        source = root / "module.py"
        source.write_text("VALUE = 1\n")
        cache = root / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-314.pyc").write_bytes(b"cache-one")
        first = campaign._runtime_tree_identity(root)
        (cache / "module.cpython-314.pyc").write_bytes(b"cache-two")
        self.assertEqual(first, campaign._runtime_tree_identity(root))

        direct_bytecode = root / "sourceless.pyc"
        direct_bytecode.write_bytes(b"importable-bytecode")
        with self.assertRaisesRegex(RuntimeError, "bytecode"):
            campaign._runtime_tree_identity(root)
        with self.assertRaisesRegex(RuntimeError, "bytecode"):
            campaign._bootstrap_tree(root)
        direct_bytecode.unlink()

        data = root / "weights.bin"
        data.write_bytes(b"one")
        second = campaign._runtime_tree_identity(root)
        self.assertNotEqual(
            first["manifest_sha256"], second["manifest_sha256"]
        )
        data.write_bytes(b"two")
        third = campaign._runtime_tree_identity(root)
        self.assertNotEqual(
            second["manifest_sha256"], third["manifest_sha256"]
        )

        indirect = root / "indirect"
        indirect.symlink_to(source)
        with self.assertRaisesRegex(RuntimeError, "indirect"):
            campaign._runtime_tree_identity(root)

    def test_runtime_requires_absent_startup_cache_prefix(self) -> None:
        with mock.patch.object(sys, "pycache_prefix", None):
            with self.assertRaisesRegex(RuntimeError, "PYTHONPYCACHEPREFIX"):
                campaign._validate_pycache_boundary()
            with self.assertRaisesRegex(RuntimeError, "PYTHONPYCACHEPREFIX"):
                collector._validate_pycache_boundary()

        blocked = self.root / "blocked-pycache"
        blocked.mkdir()
        with (
            mock.patch.object(campaign, "PYCACHE_PREFIX", blocked),
            mock.patch.object(collector, "PYCACHE_PREFIX", blocked),
        ):
            with self.assertRaisesRegex(RuntimeError, "PYTHONPYCACHEPREFIX"):
                campaign._validate_pycache_boundary()
            with self.assertRaisesRegex(RuntimeError, "PYTHONPYCACHEPREFIX"):
                collector._validate_pycache_boundary()

    def test_preflight_requires_exact_frozen_preregistration(self) -> None:
        root = self.root / "prereg"
        protocol = root / "protocol.md"
        protocol.parent.mkdir(parents=True)
        protocol.write_text("protocol\n")
        source_hashes = {"campaign": "a" * 64}
        value = {
            "experiment_id": campaign.EXPERIMENT_ID,
            "source_revision": campaign.SOURCE_REVISION,
            "development_only": True,
            "sealed_test_allowed": False,
            "outcomes_viewed_before_preregistration": False,
            "generation_01_outcomes_are_training_data": False,
            "generation_02_outcomes_collected_before_freeze": False,
            "status": campaign.PREREGISTRATION_STATUS,
            "protocol_sha256": campaign.sha256_file(protocol),
            "frozen_source_sha256": source_hashes,
            "runtime_manifest": campaign.runtime_identity(),
        }
        _write(root / "preregistration.json", value)
        with (
            mock.patch.object(campaign, "EXPERIMENT", root),
            mock.patch.object(campaign, "PROTOCOL", protocol),
            mock.patch.object(
                campaign, "SOURCE_IDENTITY", root / "source-identity.json"
            ),
            mock.patch.object(campaign, "PREFLIGHT", root / "preflight.json"),
            mock.patch.object(
                campaign,
                "PREFLIGHT_RECEIPT",
                root / "preflight.receipt.json",
            ),
            mock.patch.object(
                campaign,
                "_preregistered_source_hashes",
                return_value=source_hashes,
            ),
        ):
            campaign._validate_preregistration()
            campaign._validate_preflight_prefix(0)
            (root / "unexpected").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                campaign._validate_preflight_prefix(0)
            (root / "unexpected").unlink()
            value["runtime_manifest"]["numpy_version"] = "foreign"
            _write(root / "preregistration.json", value)
            with self.assertRaisesRegex(RuntimeError, "preregistration"):
                campaign._validate_preregistration()
            value["runtime_manifest"] = campaign.runtime_identity()
            value["status"] = "draft"
            _write(root / "preregistration.json", value)
            with self.assertRaisesRegex(RuntimeError, "preregistration"):
                campaign._validate_preregistration()

    def test_preflight_accepts_only_exact_restartable_writer_prefix(self) -> None:
        root = self.root / "preflight-prefix"
        protocol = root / "protocol.md"
        protocol.parent.mkdir(parents=True)
        protocol.write_text("protocol\n")
        preregistration = {
            "experiment_id": campaign.EXPERIMENT_ID,
            "source_revision": campaign.SOURCE_REVISION,
            "development_only": True,
            "sealed_test_allowed": False,
            "outcomes_viewed_before_preregistration": False,
            "generation_01_outcomes_are_training_data": False,
            "generation_02_outcomes_collected_before_freeze": False,
            "status": campaign.PREREGISTRATION_STATUS,
            "protocol_sha256": campaign.sha256_file(protocol),
            "frozen_source_sha256": {},
        }
        _write(root / "preregistration.json", preregistration)
        patches = (
            mock.patch.object(campaign, "EXPERIMENT", root),
            mock.patch.object(campaign, "PROTOCOL", protocol),
            mock.patch.object(
                campaign, "SOURCE_IDENTITY", root / "source-identity.json"
            ),
            mock.patch.object(campaign, "PREFLIGHT", root / "preflight.json"),
            mock.patch.object(
                campaign,
                "PREFLIGHT_RECEIPT",
                root / "preflight.receipt.json",
            ),
        )
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            rows = campaign._preflight_rows(0)
            self.assertEqual(rows, campaign._preflight_rows(1))
            campaign._validate_preflight_prefix(0)

            first, first_value = rows[0]
            first.parent.mkdir(parents=True)
            future, _future_value = rows[1]
            quarantine_root = (
                root / "_incomplete/_atomic" / first.parent.name
            )
            quarantine_root.mkdir(parents=True)
            future_payload = b"future"
            future_digest = hashlib.sha256(future_payload).hexdigest()
            future_quarantine = quarantine_root / (
                f"{future.name}.{future_digest}.123.456.0"
            )
            future_quarantine.write_bytes(future_payload)
            with self.assertRaisesRegex(RuntimeError, "out-of-order"):
                campaign._validate_preflight_prefix(1)
            future_quarantine.unlink()

            prelink = first.parent / f".{first.name}.tmp-123-456"
            prelink.write_bytes(b"partial")
            campaign._validate_preflight_prefix(0)
            prelink.unlink()

            prelink.write_bytes(b"wrong-content")
            digest = campaign.sha256_file(prelink)
            quarantine = quarantine_root / (
                f"{first.name}.{digest}.123.456.0"
            )
            quarantine.hardlink_to(prelink)
            campaign._validate_preflight_prefix(1)
            campaign.write_once(first, first_value)
            self.assertFalse(prelink.exists())
            self.assertEqual(quarantine.stat().st_nlink, 1)

            postlink = first.parent / f".{first.name}.tmp-123-456"
            postlink.hardlink_to(first)
            campaign._validate_preflight_prefix(1)
            postlink.unlink()

            for path, value in rows[1:]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(campaign._artifact_bytes(value))
                campaign._validate_preflight_prefix(1)

            first.unlink()
            with self.assertRaisesRegex(RuntimeError, "out-of-order"):
                campaign._validate_preflight_prefix(1)

    def test_checkpoint_link_boundary_is_recoverable(self) -> None:
        for boundary in ("pre-link", "post-link"):
            with self.subTest(boundary=boundary):
                model_root = self.root / boundary / "model"
                model_root.mkdir(parents=True)
                checkpoint = model_root / "resolution-first-g2.pt"
                temporary = (
                    model_root / ".resolution-first-g2.pt.tmp-123-456"
                )
                temporary.write_bytes(b"linked finite checkpoint")
                if boundary == "post-link":
                    checkpoint.hardlink_to(temporary)
                dataset = SimpleNamespace(sha256="e" * 64)
                config = campaign._frozen_model_config_manifest()
                model = SimpleNamespace(
                    manifest=mock.Mock(
                        return_value={
                            "schema": "synthetic",
                            "config": config,
                        }
                    )
                )
                identity_sha = "f" * 64
                report = {
                    "schema": "irisu-r3h-g2-training-v1",
                    "development_only": True,
                    "sealed_test_allowed": False,
                    "source_identity_sha256": identity_sha,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": campaign.sha256_file(temporary),
                    "dataset_sha256": dataset.sha256,
                    "collection_identity_sha256": "1" * 64,
                    "model_config_sha256": campaign.canonical_sha256(config),
                    "model_manifest": model.manifest(),
                }
                with (
                    mock.patch.object(campaign, "EXPERIMENT", self.root),
                    mock.patch.object(campaign, "MODEL", model_root),
                    mock.patch.object(campaign, "CHECKPOINT", checkpoint),
                    mock.patch.object(
                        campaign, "TRAINING", model_root / "training.json"
                    ),
                    mock.patch.object(
                        campaign,
                        "CHECKPOINT_RECEIPT",
                        model_root / "checkpoint.receipt.json",
                    ),
                    mock.patch.object(
                        campaign,
                        "load_checkpoint_g2",
                        return_value=(
                            model,
                            {
                                "experiment_id": campaign.EXPERIMENT_ID,
                                "dataset_sha256": dataset.sha256,
                                "source_identity_sha256": identity_sha,
                                "model_config_sha256": (
                                    campaign.canonical_sha256(config)
                                ),
                            },
                        ),
                    ),
                    mock.patch.object(
                        campaign, "_training_report", return_value=dict(report)
                    ),
                ):
                    recovered = campaign._recover_training(dataset, identity_sha)
                self.assertEqual(
                    recovered["checkpoint_sha256"],
                    report["checkpoint_sha256"],
                )
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(checkpoint.stat().st_nlink, 1)
                self.assertFalse(temporary.exists())
                self.assertTrue((model_root / "training.json").is_file())
                self.assertTrue(
                    (model_root / "checkpoint.receipt.json").is_file()
                )

    def test_altered_config_checkpoint_temp_is_quarantined(self) -> None:
        model_root = self.root / "altered-config/model"
        model_root.mkdir(parents=True)
        checkpoint = model_root / "resolution-first-g2.pt"
        temporary = model_root / ".resolution-first-g2.pt.tmp-123-456"
        temporary.write_bytes(b"finite-foreign-config")
        dataset = SimpleNamespace(sha256="e" * 64)
        identity_sha = "f" * 64
        expected = campaign._frozen_model_config_manifest()
        altered = {**expected, "hidden_width": int(expected["hidden_width"]) + 1}
        model = SimpleNamespace(
            manifest=mock.Mock(
                return_value={"schema": "synthetic", "config": altered}
            )
        )
        with (
            mock.patch.object(campaign, "EXPERIMENT", self.root),
            mock.patch.object(campaign, "CHECKPOINT", checkpoint),
            mock.patch.object(campaign, "TRAINING", model_root / "training.json"),
            mock.patch.object(
                campaign,
                "CHECKPOINT_RECEIPT",
                model_root / "checkpoint.receipt.json",
            ),
            mock.patch.object(
                campaign,
                "load_checkpoint_g2",
                return_value=(
                    model,
                    {
                        "experiment_id": campaign.EXPERIMENT_ID,
                        "dataset_sha256": dataset.sha256,
                        "source_identity_sha256": identity_sha,
                        "model_config_sha256": (
                            campaign.canonical_sha256(expected)
                        ),
                    },
                ),
            ),
        ):
            self.assertIsNone(
                campaign._recover_training(dataset, identity_sha)
            )
        self.assertFalse(checkpoint.exists())
        self.assertFalse(temporary.exists())
        preserved = list(
            (self.root / "_incomplete/_atomic/model").iterdir()
        )
        self.assertEqual(len(preserved), 1)
        self.assertEqual(
            preserved[0].read_bytes(), b"finite-foreign-config"
        )

    def test_write_once_recovers_and_quarantines_atomic_temps(self) -> None:
        value = {"schema": "synthetic"}
        encoded = (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        for boundary in ("pre-link", "post-link"):
            with self.subTest(boundary=boundary):
                directory = self.root / f"json-{boundary}"
                directory.mkdir()
                target = directory / "report.json"
                temporary = directory / ".report.json.tmp-123-456"
                temporary.write_bytes(encoded)
                if boundary == "post-link":
                    target.hardlink_to(temporary)
                with mock.patch.object(campaign, "EXPERIMENT", self.root):
                    campaign.write_once(target, value)
                self.assertEqual(target.read_bytes(), encoded)
                self.assertEqual(target.stat().st_nlink, 1)
                self.assertFalse(temporary.exists())

        directory = self.root / "partial"
        directory.mkdir()
        target = directory / "report.json"
        temporary = directory / ".report.json.tmp-123-456"
        temporary.write_bytes(b"partial")
        with mock.patch.object(campaign, "EXPERIMENT", self.root):
            campaign.write_once(target, value)
        preserved = list((self.root / "_incomplete/_atomic/partial").iterdir())
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"partial")

        interrupted = self.root / "interrupted/report.json"
        interrupted.parent.mkdir()
        interrupted_temp = (
            interrupted.parent / ".report.json.tmp-321-654"
        )
        interrupted_temp.write_bytes(b"interrupted")
        digest = campaign.sha256_file(interrupted_temp)
        interrupted_quarantine = (
            self.root
            / "_incomplete/_atomic/interrupted"
            / f"{interrupted.name}.{digest}.321.654.0"
        )
        interrupted_quarantine.parent.mkdir(parents=True)
        interrupted_quarantine.hardlink_to(interrupted_temp)
        with mock.patch.object(campaign, "EXPERIMENT", self.root):
            campaign.write_once(interrupted, value)
        self.assertEqual(interrupted.read_bytes(), encoded)
        self.assertFalse(interrupted_temp.exists())
        self.assertEqual(interrupted_quarantine.stat().st_nlink, 1)

        unknown = self.root / "unknown" / "report.json"
        unknown.parent.mkdir()
        (unknown.parent / ".report.json.foreign").write_bytes(b"x")
        with mock.patch.object(campaign, "EXPERIMENT", self.root):
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                campaign.write_once(unknown, value)

    def test_unusable_checkpoint_temp_is_hash_preserved(self) -> None:
        model_root = self.root / "corrupt/model"
        model_root.mkdir(parents=True)
        checkpoint = model_root / "resolution-first-g2.pt"
        temporary = model_root / ".resolution-first-g2.pt.tmp-123-456"
        temporary.write_bytes(b"corrupt-checkpoint")
        dataset = SimpleNamespace(sha256="e" * 64)
        with (
            mock.patch.object(campaign, "EXPERIMENT", self.root),
            mock.patch.object(campaign, "CHECKPOINT", checkpoint),
            mock.patch.object(campaign, "TRAINING", model_root / "training.json"),
            mock.patch.object(
                campaign,
                "CHECKPOINT_RECEIPT",
                model_root / "checkpoint.receipt.json",
            ),
            mock.patch.object(
                campaign,
                "load_checkpoint_g2",
                side_effect=RuntimeError("corrupt"),
            ),
        ):
            self.assertIsNone(campaign._recover_training(dataset, "f" * 64))
        self.assertFalse(temporary.exists())
        preserved = list((self.root / "_incomplete/_atomic/model").iterdir())
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"corrupt-checkpoint")

    def test_checkpoint_finishes_interrupted_quarantine(self) -> None:
        model_root = self.root / "checkpoint-interrupted/model"
        model_root.mkdir(parents=True)
        checkpoint = model_root / "resolution-first-g2.pt"
        temporary = model_root / ".resolution-first-g2.pt.tmp-123-456"
        temporary.write_bytes(b"corrupt-checkpoint")
        digest = campaign.sha256_file(temporary)
        quarantine = (
            self.root
            / "_incomplete/_atomic/model"
            / f"{checkpoint.name}.{digest}.123.456.0"
        )
        quarantine.parent.mkdir(parents=True)
        quarantine.hardlink_to(temporary)
        dataset = SimpleNamespace(sha256="e" * 64)
        with (
            mock.patch.object(campaign, "EXPERIMENT", self.root),
            mock.patch.object(campaign, "CHECKPOINT", checkpoint),
            mock.patch.object(campaign, "TRAINING", model_root / "training.json"),
            mock.patch.object(
                campaign,
                "CHECKPOINT_RECEIPT",
                model_root / "checkpoint.receipt.json",
            ),
        ):
            self.assertIsNone(campaign._recover_training(dataset, "f" * 64))
        self.assertFalse(temporary.exists())
        self.assertEqual(quarantine.stat().st_nlink, 1)
        self.assertEqual(quarantine.read_bytes(), b"corrupt-checkpoint")

    def test_failed_support_never_opens_margin(self) -> None:
        support_path = self.root / "model/support-calibration.json"
        support_receipt = self.root / "model/support-calibration.receipt.json"
        margin_path = self.root / "model/margin-calibration.json"
        auroc = {
            "schema": "irisu-r3h-resolution-auroc-g2-v2",
            "auroc": 0.6,
            "passed": False,
        }
        dataset = SimpleNamespace(sha256="8" * 64)
        with (
            mock.patch.object(
                campaign, "_load_model", return_value=(object(), "7" * 64)
            ),
            mock.patch.object(
                campaign, "verify_source_identity", return_value="6" * 64
            ),
            mock.patch.object(
                campaign,
                "_compute_support_calibration",
                return_value=(dataset, None, auroc, [{"passed": False}]),
            ),
            mock.patch.object(
                campaign,
                "collection_identity_sha256",
                return_value="5" * 64,
            ),
            mock.patch.object(campaign, "SUPPORT_CALIBRATION", support_path),
            mock.patch.object(
                campaign,
                "SUPPORT_CALIBRATION_RECEIPT",
                support_receipt,
            ),
            mock.patch.object(campaign, "MARGIN_CALIBRATION", margin_path),
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    campaign.command_calibrate_support(SimpleNamespace())
        self.assertFalse(margin_path.exists())
        self.assertFalse(json.loads(support_receipt.read_text())["passed"])

    def test_support_no_go_blocks_margin_computation(self) -> None:
        no_go = RuntimeError("support NO-GO")
        margin = mock.Mock(side_effect=AssertionError("margin opened"))
        with (
            mock.patch.object(campaign, "_load_model", return_value=(object(), "a")),
            mock.patch.object(campaign, "verify_source_identity", return_value="b"),
            mock.patch.object(campaign, "_support_calibration", side_effect=no_go),
            mock.patch.object(campaign, "_compute_margin_calibration", margin),
        ):
            with self.assertRaisesRegex(RuntimeError, "support NO-GO"):
                campaign.command_calibrate_margin(SimpleNamespace())
        margin.assert_not_called()

    def test_infinite_margin_q_writes_terminal_receipt(self) -> None:
        model_root = self.root / "infinite-margin/model"
        margin_path = model_root / "margin-calibration.json"
        margin_receipt = model_root / "margin-calibration.receipt.json"
        support_receipt = model_root / "support-calibration.receipt.json"
        support_receipt.parent.mkdir(parents=True)
        support_receipt.write_bytes(b"support-receipt\n")
        support = SimpleNamespace(sha256="4" * 64)
        selective_manifest = {
            "schema": "synthetic-selective",
            "q": float("inf"),
        }
        selective = SimpleNamespace(
            q=float("inf"),
            manifest=mock.Mock(return_value=selective_manifest),
        )
        dataset = SimpleNamespace(sha256="5" * 64)
        with (
            mock.patch.object(campaign, "runtime_identity"),
            mock.patch.object(
                campaign, "_load_model", return_value=(object(), "1" * 64)
            ),
            mock.patch.object(
                campaign, "verify_source_identity", return_value="2" * 64
            ),
            mock.patch.object(
                campaign, "_support_calibration", return_value=support
            ),
            mock.patch.object(
                campaign,
                "_compute_margin_calibration",
                return_value=(dataset, selective),
            ),
            mock.patch.object(
                campaign,
                "collection_identity_sha256",
                return_value="3" * 64,
            ),
            mock.patch.object(
                campaign,
                "SUPPORT_CALIBRATION_RECEIPT",
                support_receipt,
            ),
            mock.patch.object(
                campaign, "MARGIN_CALIBRATION", margin_path
            ),
            mock.patch.object(
                campaign,
                "MARGIN_CALIBRATION_RECEIPT",
                margin_receipt,
            ),
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    campaign.command_calibrate_margin(SimpleNamespace())
        self.assertEqual(stopped.exception.code, 2)
        report = json.loads(margin_path.read_text())
        receipt = json.loads(margin_receipt.read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["selective"]["q"], "Infinity")
        self.assertFalse(receipt["finite_q_b2"])
        self.assertFalse(receipt["passed"])
        self.assertEqual(
            receipt["selective_sha256"],
            campaign.canonical_sha256(
                campaign._jsonable(selective_manifest)
            ),
        )

    def test_failed_margin_never_opens_offline(self) -> None:
        offline = mock.Mock(side_effect=AssertionError("offline opened"))
        with (
            mock.patch.object(campaign, "_load_model", return_value=(object(), "a")),
            mock.patch.object(campaign, "_support_calibration", return_value=object()),
            mock.patch.object(
                campaign,
                "_margin_calibration",
                side_effect=RuntimeError("margin NO-GO"),
            ),
            mock.patch.object(campaign, "load_dataset", offline),
        ):
            with self.assertRaisesRegex(RuntimeError, "margin NO-GO"):
                campaign.command_screen(SimpleNamespace())
        offline.assert_not_called()

    def test_campaign_requires_exactly_one_cpu(self) -> None:
        with mock.patch.object(
            campaign.os, "sched_getaffinity", return_value={0, 1}
        ):
            with self.assertRaisesRegex(RuntimeError, "one logical CPU"):
                campaign._one_cpu()


if __name__ == "__main__":
    unittest.main()
