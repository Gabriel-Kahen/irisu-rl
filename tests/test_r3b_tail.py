from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, replace

import torch

from irisu_rl.r3b_tail import ScoreOnlyTailController


@dataclass(frozen=True)
class AuditRow:
    shaping_weight_ppm: tuple[int, ...]
    raw_rewards: tuple[int, ...]
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
    reward_sha256: str
    decisions: tuple[AuditRow, ...]


REWARD_SHA256 = "a" * 64


def audit(*rows: tuple[int, ...]) -> Audit:
    decisions = []
    for weights in rows:
        raw = tuple(index + 1 for index in range(len(weights)))
        scaled = tuple(float(value) for value in raw)
        shaping = tuple(0.0 if weight == 0 else 0.25 for weight in weights)
        optimizer = tuple(
            float(value)
            for value in (
                torch.tensor(scaled, dtype=torch.float32)
                + torch.tensor(shaping, dtype=torch.float32)
                * (
                    torch.tensor(weights, dtype=torch.int64).to(torch.float32)
                    / 1_000_000.0
                )
            ).tolist()
        )
        decisions.append(AuditRow(weights, raw, scaled, shaping, optimizer))
    decisions = tuple(decisions)
    return Audit(
        len(decisions),
        sum(len(row.shaping_weight_ppm) for row in decisions),
        sum(sum(row.raw_rewards) for row in decisions),
        sum(
            float(torch.tensor(row.optimizer_rewards, dtype=torch.float32).sum())
            for row in decisions
        ),
        0,
        REWARD_SHA256,
        decisions,
    )


def tail_controller(
    sweep_updates: int, *, minimum_score_only_updates: int = 400
) -> ScoreOnlyTailController:
    return ScoreOnlyTailController(
        sweep_updates,
        minimum_score_only_updates=minimum_score_only_updates,
        reward_scale=1.0,
        reward_sha256=REWARD_SHA256,
    )


class ScoreOnlyTailTests(unittest.TestCase):
    def complete_update(
        self, controller: ScoreOnlyTailController, evidence: Audit
    ) -> None:
        before = controller.completed_updates
        controller.validate_optimizer_update(evidence, completed_updates=before)
        controller.record_optimizer_update(evidence, completed_updates=before + 1)

    def test_sweep_drain_and_four_hundred_score_only_updates(self) -> None:
        controller = tail_controller(2)
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
        controller = tail_controller(1)
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
        controller = tail_controller(1)
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
        controller = tail_controller(1)
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
        with self.assertRaisesRegex(ValueError, "reward identity"):
            controller.validate_optimizer_update(
                replace(zero, reward_sha256="b" * 64), completed_updates=0
            )
        with self.assertRaisesRegex(ValueError, "invalid actions"):
            controller.validate_optimizer_update(
                replace(zero, invalid_actions=1), completed_updates=0
            )
        with self.assertRaisesRegex(ValueError, "raw reward and scale"):
            controller.validate_optimizer_update(
                replace(
                    zero,
                    decisions=(replace(zero.decisions[0], raw_rewards=(2, 2)),),
                ),
                completed_updates=0,
            )
        with self.assertRaisesRegex(ValueError, "raw reward does not match"):
            controller.validate_optimizer_update(
                replace(zero, raw_reward=zero.raw_reward + 1), completed_updates=0
            )
        false_score_only = replace(zero.decisions[0], optimizer_rewards=(1.5, 2.0))
        with self.assertRaisesRegex(ValueError, "composition"):
            controller.validate_optimizer_update(
                replace(zero, decisions=(false_score_only,)), completed_updates=0
            )

    def test_update_transaction_rejects_clock_or_audit_substitution(self) -> None:
        controller = tail_controller(2)
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
        source = tail_controller(1, minimum_score_only_updates=401)
        shaped = audit(
            (100_000, 0),
        )
        self.complete_update(source, shaped)
        source.collection_mode(
            completed_updates=1, lane_shaping_weight_ppm=(100_000, 0)
        )
        source.record_drain(shaped, completed_updates=1)
        state = source.state_dict()

        restored = tail_controller(1, minimum_score_only_updates=401)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.sha256, source.sha256)
        self.assertGreater(restored.event_count, 0)

        wrong_identity = tail_controller(1)
        with self.assertRaisesRegex(ValueError, "identity"):
            wrong_identity.load_state_dict(state)
        malformed = dict(state)
        malformed["score_only_updates"] = 1
        with self.assertRaisesRegex(ValueError, "clocks disagree"):
            restored.load_state_dict(malformed)
        self.assertEqual(restored.state_dict(), state)

        missing_chain = dict(state)
        missing_chain["event_head"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "event-chain head"):
            restored.load_state_dict(missing_chain)
        self.assertEqual(restored.state_dict(), state)

        wrong_count = dict(state)
        wrong_count["event_count"] = int(state["event_count"]) + 1
        with self.assertRaisesRegex(ValueError, "event count"):
            restored.load_state_dict(wrong_count)
        self.assertEqual(restored.state_dict(), state)

        old_version = dict(state)
        old_version["version"] = "score-only-tail-controller-v2"
        with self.assertRaisesRegex(ValueError, "version"):
            restored.load_state_dict(old_version)

        fresh = tail_controller(1)
        impossible_fresh = fresh.state_dict()
        impossible_fresh["event_head"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "event-chain head"):
            fresh.load_state_dict(impossible_fresh)

    def test_configuration_and_weight_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            tail_controller(0)
        with self.assertRaisesRegex(ValueError, "at least 400"):
            tail_controller(1, minimum_score_only_updates=399)
        controller = tail_controller(1)
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
