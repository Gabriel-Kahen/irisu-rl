from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from irisu_pointer.sequence_replay import (
    SEQUENCE_REPLAY_EVENT_FEATURES,
    SequenceReplayConfig,
    SequenceReplayModel,
    load_sequence_replay_checkpoint,
    save_sequence_replay_checkpoint,
)
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    SteeringModelConfig,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.schema import TEACHER_V1


def _body(
    identifier: int,
    *,
    color: int,
    x: float,
    lifecycle: str = "dynamic_fresh",
    chain_id: int = 0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": lifecycle,
        "color": color,
        "x": x,
        "y": 200.0,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": chain_id,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1_000,
        "rot_timer": 0,
    }


def _observation(tick: int) -> dict[str, object]:
    return {
        "tick": tick,
        "score": 100,
        "gauge": 800,
        "gauge_max": 1_000,
        "level": 2,
        "highest_chain": 1,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {
            "x": 0.0,
            "y": 0.0,
            "width": 640.0,
            "height": 480.0,
        },
        "difficulty": {
            "active_colors": 4,
            "spawn_interval_ticks": 100,
        },
        "bodies": (
            _body(11, color=0, x=120.0),
            _body(22, color=0, x=240.0),
            _body(33, color=1, x=400.0),
            _body(44, color=1, x=520.0),
        ),
    }


def _inputs(
    *,
    time: int = 3,
) -> tuple[torch.Tensor, ...]:
    encoded = TeacherStateEncoder().encode(
        [_observation(10 + index) for index in range(time)]
    )
    active = encoded.body_mask.any(axis=0).nonzero()[0]
    width = int(active[-1]) + 1
    global_features = torch.from_numpy(encoded.global_features).unsqueeze(1)
    body_features = torch.from_numpy(
        encoded.body_features[:, :width]
    ).unsqueeze(1)
    body_mask = torch.from_numpy(encoded.body_mask[:, :width]).unsqueeze(1)
    event_features = torch.zeros(
        time, 1, len(SEQUENCE_REPLAY_EVENT_FEATURES)
    )
    reset_before = torch.zeros(time, 1, dtype=torch.bool)
    reset_before[0] = True
    valid_mask = torch.ones(time, 1, dtype=torch.bool)
    return (
        global_features,
        body_features,
        body_mask,
        event_features,
        reset_before,
        valid_mask,
    )


def _models() -> tuple[GoalConditionedSteeringModel, SequenceReplayModel]:
    torch.manual_seed(719)
    base = GoalConditionedSteeringModel(
        TEACHER_V1,
        config=SteeringModelConfig(
            body_hidden=12,
            global_hidden=8,
            pair_hidden=16,
        ),
    )
    sequence = SequenceReplayModel(
        base,
        config=SequenceReplayConfig(
            body_hidden=12,
            global_hidden=8,
            event_hidden=6,
            recurrent_hidden=14,
            pair_hidden=16,
            value_quantiles=5,
        ),
        base_checkpoint_sha256="1" * 64,
    )
    sequence.eval()
    return base, sequence


def _activate_residuals(model: SequenceReplayModel) -> None:
    generator = torch.Generator().manual_seed(991)
    with torch.no_grad():
        for layer in model._residual_heads():
            layer.weight.uniform_(-0.25, 0.25, generator=generator)
            layer.bias.uniform_(-0.1, 0.1, generator=generator)


def _forward(
    model: SequenceReplayModel,
    inputs: tuple[torch.Tensor, ...],
):
    return model(
        *inputs[:5],
        valid_mask=inputs[5],
    )


def _assert_close(
    case: unittest.TestCase,
    first: object,
    second: object,
    *,
    fields: tuple[str, ...],
) -> None:
    for name in fields:
        left = getattr(first, name)
        right = getattr(second, name)
        case.assertEqual(left.dtype, right.dtype, name)
        if left.dtype == torch.bool:
            case.assertTrue(torch.equal(left, right), name)
        else:
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


POLICY_FIELDS = (
    "act_logits",
    "wait_logits",
    "pair_logits",
    "kind_logits",
    "template_logits",
    "intent_logits",
    "legal_pair_mask",
)


class PointerSequenceReplayTests(unittest.TestCase):
    def test_zero_residual_is_exactly_the_frozen_base_policy(self) -> None:
        base, sequence = _models()
        inputs = _inputs()
        with torch.no_grad():
            output = _forward(sequence, inputs)
            time, batch, bodies, features = inputs[1].shape
            base_output = base(
                inputs[0].reshape(time * batch, -1),
                inputs[1].reshape(time * batch, bodies, features),
                inputs[2].reshape(time * batch, bodies),
            )
        for name in POLICY_FIELDS:
            expected = getattr(base_output, name).reshape(
                time, batch, *getattr(base_output, name).shape[1:]
            )
            self.assertTrue(
                torch.equal(getattr(output, name), expected),
                name,
            )
        self.assertEqual(float(output.residual_energy), 0.0)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in base.parameters())
        )

    def test_body_row_permutation_is_equivariant(self) -> None:
        _, sequence = _models()
        _activate_residuals(sequence)
        inputs = _inputs()
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = (
            inputs[0],
            inputs[1].index_select(2, permutation),
            inputs[2].index_select(2, permutation),
            *inputs[3:],
        )
        with torch.no_grad():
            original = _forward(sequence, inputs)
            changed = _forward(sequence, permuted)
        invariant = (
            "act_logits",
            "wait_logits",
            "return_quantiles",
            "viability_logits",
            "outcome_values",
            "recurrent_state",
        )
        _assert_close(self, original, changed, fields=invariant)
        for name in (
            "pair_logits",
            "kind_logits",
            "template_logits",
            "intent_logits",
            "legal_pair_mask",
        ):
            expected = getattr(original, name).index_select(
                2, permutation
            ).index_select(3, permutation)
            actual = getattr(changed, name)
            if actual.dtype == torch.bool:
                self.assertTrue(torch.equal(actual, expected), name)
            else:
                torch.testing.assert_close(
                    actual, expected, rtol=1e-5, atol=1e-6
                )

    def test_identifier_and_chain_renaming_cannot_change_outputs(self) -> None:
        _, sequence = _models()
        _activate_residuals(sequence)
        inputs = _inputs()
        first_bodies = inputs[1].clone()
        second_bodies = inputs[1].clone()
        id_index = TEACHER_V1.body_features.index("id_scaled")
        chain_index = TEACHER_V1.body_features.index("chain_id_scaled")
        first_bodies[..., id_index] = torch.tensor(
            [0.01, 0.02, 0.03, 0.04]
        )
        second_bodies[..., id_index] = torch.tensor(
            [0.91, 0.37, 0.72, 0.18]
        )
        first_bodies[..., chain_index] = torch.tensor(
            [0.0, 0.2, 0.2, 0.6]
        )
        second_bodies[..., chain_index] = torch.tensor(
            [0.0, 0.9, 0.9, 0.1]
        )
        first_inputs = (inputs[0], first_bodies, *inputs[2:])
        second_inputs = (inputs[0], second_bodies, *inputs[2:])
        with torch.no_grad():
            first = _forward(sequence, first_inputs)
            second = _forward(sequence, second_inputs)
        _assert_close(
            self,
            first,
            second,
            fields=(
                *POLICY_FIELDS,
                "return_quantiles",
                "viability_logits",
                "outcome_values",
                "recurrent_state",
            ),
        )

    def test_future_events_are_causal_and_reset_erases_prior_history(
        self,
    ) -> None:
        _, sequence = _models()
        _activate_residuals(sequence)
        inputs = _inputs()
        future_events = inputs[3].clone()
        future_events[-1, 0, 0] = 20.0
        future = (*inputs[:3], future_events, *inputs[4:])
        with torch.no_grad():
            original = _forward(sequence, inputs)
            changed_future = _forward(sequence, future)
        for name in (
            *POLICY_FIELDS,
            "return_quantiles",
            "viability_logits",
            "outcome_values",
        ):
            left = getattr(original, name)[:-1]
            right = getattr(changed_future, name)[:-1]
            if left.dtype == torch.bool:
                self.assertTrue(torch.equal(left, right), name)
            else:
                self.assertTrue(torch.equal(left, right), name)
        self.assertFalse(
            torch.allclose(
                original.recurrent_state,
                changed_future.recurrent_state,
            )
        )

        short = tuple(value[:2].clone() for value in inputs)
        alternate_events = short[3].clone()
        alternate_events[0, 0, 1] = 30.0
        alternate = (*short[:3], alternate_events, *short[4:])
        with torch.no_grad():
            no_reset_first = _forward(sequence, short)
            no_reset_second = _forward(sequence, alternate)
        self.assertFalse(
            torch.allclose(
                no_reset_first.recurrent_state,
                no_reset_second.recurrent_state,
            )
        )

        reset = short[4].clone()
        reset[1] = True
        first_reset = (*short[:4], reset, short[5])
        second_reset = (*alternate[:4], reset, alternate[5])
        with torch.no_grad():
            reset_first = _forward(sequence, first_reset)
            reset_second = _forward(sequence, second_reset)
        self.assertTrue(
            torch.equal(
                reset_first.recurrent_state,
                reset_second.recurrent_state,
            )
        )
        for name in POLICY_FIELDS:
            self.assertTrue(
                torch.equal(
                    getattr(reset_first, name)[1],
                    getattr(reset_second, name)[1],
                ),
                name,
            )

    def test_residual_path_cannot_unmask_illegal_pairs(self) -> None:
        _, sequence = _models()
        _activate_residuals(sequence)
        inputs = _inputs(time=1)
        with torch.no_grad():
            output = _forward(sequence, inputs)
        legal = output.legal_pair_mask[0, 0]
        self.assertTrue(bool(legal.any()))
        self.assertFalse(bool(legal.diagonal().any()))
        floor = torch.finfo(output.pair_logits.dtype).min
        self.assertTrue(
            torch.equal(
                output.pair_logits[0, 0][~legal],
                torch.full_like(output.pair_logits[0, 0][~legal], floor),
            )
        )
        self.assertTrue(
            bool(torch.isfinite(output.pair_logits[0, 0][legal]).all())
        )

    def test_checkpoint_round_trip_preserves_residual_policy(self) -> None:
        _, first = _models()
        _activate_residuals(first)
        source_identity = "2" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence-replay.pt"
            checkpoint_sha256 = save_sequence_replay_checkpoint(
                path,
                first,
                source_identity=source_identity,
                metadata={"development_only": True},
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertFalse(
                any(
                    name.startswith("base_model.")
                    for name in payload["state_dict"]
                )
            )
            second_base, _ = _models()
            checkpoint = load_sequence_replay_checkpoint(
                path,
                second_base,
                expected_sha256=checkpoint_sha256,
                expected_base_checkpoint_sha256="1" * 64,
                expected_source_identity=source_identity,
            )
        second = checkpoint.model
        self.assertEqual(
            checkpoint.metadata, {"development_only": True}
        )
        self.assertEqual(first.manifest(), second.manifest())
        self.assertEqual(
            first.architecture_sha256, second.architecture_sha256
        )
        inputs = _inputs()
        with torch.no_grad():
            first_output = _forward(first, inputs)
            second_output = _forward(second, inputs)
        _assert_close(
            self,
            first_output,
            second_output,
            fields=(
                *POLICY_FIELDS,
                "return_quantiles",
                "viability_logits",
                "outcome_values",
                "recurrent_state",
            ),
        )


if __name__ == "__main__":
    unittest.main()
