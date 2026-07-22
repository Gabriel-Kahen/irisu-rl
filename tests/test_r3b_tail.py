from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, replace

import torch

from irisu_rl.r3b_tail import ScoreOnlyTailController


@dataclass(frozen=True)
class AuditRow:
    shaping_weight_ppm: tuple[int, ...]
    scaled_raw_rewards: tuple[float, ...]
    shaping_rewards: tuple[float, ...]
    optimizer_rewards: tuple[float, ...]


@dataclass(frozen=True)
class Audit:
    decision_rows: int
    transitions: int
    raw_reward: int
    optimizer_reward: float
    invalid_actions: int
    decisions: tuple[AuditRow, ...]


def audit(*rows: tuple[int, ...]) -> Audit:
    decisions = tuple(
        AuditRow(
            weights,
            tuple(float(index + 1) for index in range(len(weights))),
            tuple(0.25 for _ in weights),
            tuple(
                float(index + 1) if weight == 0 else float(index + 1) + 0.25
                for index, weight in enumerate(weights)
            ),
        )
        for weights in rows
    )
    return Audit(
        len(decisions),
        sum(len(row.shaping_weight_ppm) for row in decisions),
        1,
        sum(sum(row.optimizer_rewards) for row in decisions),
        0,
        decisions,
    )


class ScoreOnlyTailTests(unittest.TestCase):
    def complete_update(
        self, controller: ScoreOnlyTailController, evidence: Audit
    ) -> None:
        before = controller.completed_updates
        controller.validate_optimizer_update(evidence, completed_updates=before)
        controller.record_optimizer_update(evidence, completed_updates=before + 1)

    def test_sweep_drain_and_four_hundred_score_only_updates(self) -> None:
        controller = ScoreOnlyTailController(2)
        shaped = audit((100_000, 100_000), (100_000, 100_000))
        zero = audit((0, 0), (0, 0))

        self.assertEqual(
            controller.collection_mode(
                completed_updates=0,
                lane_shaping_weight_ppm=torch.tensor(
                    [100_000, 100_000], dtype=torch.int64
                ),
            ),
            "train",
        )
        self.complete_update(controller, shaped)
        self.complete_update(controller, shaped)
        self.assertEqual(controller.completed_updates, 2)

        self.assertEqual(
            controller.collection_mode(
                completed_updates=2,
                lane_shaping_weight_ppm=(100_000, 0),
            ),
            "drain",
        )
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            controller.validate_optimizer_update(shaped, completed_updates=2)
        controller.record_drain(shaped, completed_updates=2)
        self.assertEqual(controller.completed_updates, 2)
        self.assertEqual(controller.drain_collections, 1)

        self.assertEqual(
            controller.collection_mode(
                completed_updates=2, lane_shaping_weight_ppm=(0, 0)
            ),
            "train",
        )
        self.assertEqual(controller.phase, "score_only")
        for _ in range(399):
            self.complete_update(controller, zero)
        self.assertEqual(controller.phase, "score_only")
        self.assertEqual(controller.score_only_updates, 399)
        self.complete_update(controller, zero)
        self.assertEqual(controller.phase, "complete")
        self.assertEqual(controller.score_only_updates, 400)
        self.assertEqual(controller.completed_updates, 402)
        self.assertEqual(
            controller.collection_mode(
                completed_updates=402, lane_shaping_weight_ppm=(0, 0)
            ),
            "closed",
        )

    def test_direct_zero_boundary_skips_draining(self) -> None:
        controller = ScoreOnlyTailController(1)
        zero = audit(
            (0, 0),
        )
        self.complete_update(controller, zero)
        self.assertEqual(
            controller.collection_mode(
                completed_updates=1, lane_shaping_weight_ppm=(0, 0)
            ),
            "train",
        )
        self.assertEqual(controller.phase, "score_only")
        with self.assertRaisesRegex(RuntimeError, "only valid"):
            controller.record_drain(zero, completed_updates=1)

    def test_drain_and_optimizer_evidence_cannot_be_relabelled(self) -> None:
        controller = ScoreOnlyTailController(1)
        shaped = audit(
            (100_000, 0),
        )
        zero = audit(
            (0, 0),
        )
        self.complete_update(controller, shaped)
        controller.collection_mode(
            completed_updates=1, lane_shaping_weight_ppm=(100_000, 0)
        )
        with self.assertRaisesRegex(ValueError, "all-zero"):
            controller.record_drain(zero, completed_updates=1)
        controller.record_drain(shaped, completed_updates=1)
        controller.collection_mode(completed_updates=1, lane_shaping_weight_ppm=(0, 0))
        with self.assertRaisesRegex(ValueError, "nonzero shaping"):
            controller.validate_optimizer_update(shaped, completed_updates=1)

    def test_missing_nonfinite_and_false_score_only_audits_fail(self) -> None:
        controller = ScoreOnlyTailController(1)
        zero = audit(
            (0, 0),
        )
        with self.assertRaisesRegex(ValueError, "required"):
            controller.validate_optimizer_update(None, completed_updates=0)
        with self.assertRaisesRegex(ValueError, "decision audits"):
            controller.validate_optimizer_update(
                replace(zero, decisions=None),
                completed_updates=0,  # type: ignore[arg-type]
            )
        nonfinite_row = replace(zero.decisions[0], optimizer_rewards=(math.inf, 2.0))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            controller.validate_optimizer_update(
                replace(zero, decisions=(nonfinite_row,)), completed_updates=0
            )
        false_score_only = replace(zero.decisions[0], optimizer_rewards=(1.5, 2.0))
        with self.assertRaisesRegex(ValueError, "exactly score-only"):
            controller.validate_optimizer_update(
                replace(zero, decisions=(false_score_only,)), completed_updates=0
            )

    def test_update_transaction_rejects_clock_or_audit_substitution(self) -> None:
        controller = ScoreOnlyTailController(2)
        first = audit(
            (100_000,),
        )
        second = audit(
            (250_000,),
        )
        controller.validate_optimizer_update(first, completed_updates=0)
        with self.assertRaisesRegex(RuntimeError, "optimizer transaction"):
            controller.state_dict()
        with self.assertRaisesRegex(ValueError, "advance by exactly one"):
            controller.record_optimizer_update(first, completed_updates=2)
        with self.assertRaisesRegex(ValueError, "changed"):
            controller.record_optimizer_update(second, completed_updates=1)
        controller.record_optimizer_update(first, completed_updates=1)
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            controller.collection_mode(
                completed_updates=0, lane_shaping_weight_ppm=(100_000,)
            )

    def test_checkpoint_round_trip_and_identity_validation(self) -> None:
        source = ScoreOnlyTailController(1, minimum_score_only_updates=401)
        shaped = audit(
            (100_000, 0),
        )
        self.complete_update(source, shaped)
        source.collection_mode(
            completed_updates=1, lane_shaping_weight_ppm=(100_000, 0)
        )
        source.record_drain(shaped, completed_updates=1)
        state = source.state_dict()

        restored = ScoreOnlyTailController(1, minimum_score_only_updates=401)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.sha256, source.sha256)

        wrong_identity = ScoreOnlyTailController(1)
        with self.assertRaisesRegex(ValueError, "identity"):
            wrong_identity.load_state_dict(state)
        malformed = dict(state)
        malformed["score_only_updates"] = 1
        with self.assertRaisesRegex(ValueError, "clocks disagree"):
            restored.load_state_dict(malformed)
        self.assertEqual(restored.state_dict(), state)

    def test_configuration_and_weight_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            ScoreOnlyTailController(0)
        with self.assertRaisesRegex(ValueError, "at least 400"):
            ScoreOnlyTailController(1, minimum_score_only_updates=399)
        controller = ScoreOnlyTailController(1)
        with self.assertRaisesRegex(ValueError, "CPU int64"):
            controller.collection_mode(
                completed_updates=0,
                lane_shaping_weight_ppm=torch.tensor([0.0]),
            )
        with self.assertRaisesRegex(ValueError, "ppm integers"):
            controller.collection_mode(
                completed_updates=0, lane_shaping_weight_ppm=(True,)
            )


if __name__ == "__main__":
    unittest.main()
