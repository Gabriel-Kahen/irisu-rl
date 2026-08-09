from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from irisu_pointer.geometry_search import GeometrySearchConfig
from irisu_pointer.policy import encoded_body_ids
from irisu_pointer.sequence_replay import (
    SEQUENCE_REPLAY_EVENT_FEATURES,
    SequenceReplayConfig,
    SequenceReplayModel,
)
from irisu_pointer.sequence_replay_policy import (
    CausalEventHistory,
    SequenceReplayPolicy,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringModelConfig,
)
from irisu_rl.actions import SemanticAction
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.schema import TEACHER_V1


EVENT_INDEX = {
    name: index for index, name in enumerate(SEQUENCE_REPLAY_EVENT_FEATURES)
}
GEOMETRY_SHA256 = "a" * 64


def _body(
    identifier: int,
    *,
    color: int,
    x: float,
    lifecycle: str = "dynamic_fresh",
    chain_id: int = 0,
    projectile_hits: int = 0,
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
        "projectile_hits": projectile_hits,
        "age_ticks": 20,
        "remaining_lifetime": 1_000,
        "rot_timer": 0,
    }


def _observation(
    tick: int,
    *,
    identifiers: tuple[int, int, int, int] = (11, 22, 33, 44),
    grouped_chain: int = 7,
    score: int = 100,
    gauge: int = 800,
) -> dict[str, object]:
    return {
        "tick": tick,
        "score": score,
        "gauge": gauge,
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
            _body(identifiers[0], color=0, x=120.0),
            _body(identifiers[1], color=0, x=240.0),
            _body(identifiers[2], color=1, x=400.0),
            _body(
                identifiers[3],
                color=1,
                x=520.0,
                lifecycle="confirmed",
                chain_id=grouped_chain,
            ),
        ),
    }


def _transition(
    observation: dict[str, object],
    *,
    tick: int,
    score: int,
    gauge: int,
    joined: tuple[int, int] | None = None,
    chain_id: int = 0,
) -> dict[str, object]:
    bodies = [dict(value) for value in observation["bodies"]]
    if joined is not None:
        for body in bodies:
            if int(body["id"]) in joined:
                body["chain_id"] = chain_id
                body["lifecycle"] = "confirmed"
        source = next(value for value in bodies if value["id"] == joined[0])
        source["projectile_hits"] = int(source["projectile_hits"]) + 1
    return {
        **observation,
        "tick": tick,
        "score": score,
        "gauge": gauge,
        "highest_chain": 2 if joined is not None else 1,
        "bodies": tuple(bodies),
    }


class _GeometryModel(nn.Module):
    candidate_count = 32
    candidate_set_sha256 = GEOMETRY_SHA256

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(
        self,
        global_features: torch.Tensor,
        body_features: torch.Tensor,
        body_mask: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
    ) -> torch.Tensor:
        del body_features, body_mask, source_index, destination_index
        return torch.zeros(
            global_features.shape[0],
            self.candidate_count,
            device=global_features.device,
        )


def _base_model(*, shot: bool) -> GoalConditionedSteeringModel:
    torch.manual_seed(101)
    model = GoalConditionedSteeringModel(
        TEACHER_V1,
        config=SteeringModelConfig(
            body_hidden=12,
            global_hidden=8,
            pair_hidden=16,
        ),
    )
    with torch.no_grad():
        model.act_head.weight.zero_()
        model.act_head.bias.copy_(
            torch.tensor([-4.0, 4.0] if shot else [4.0, -4.0])
        )
    return model


def _base_policy(
    model: GoalConditionedSteeringModel,
) -> GoalConditionedSteeringPolicy:
    return GoalConditionedSteeringPolicy(
        model,
        cooldown_ticks=4,
        act_logit_bias=0.0,
        artifact_sha256="1" * 64,
    )


def _wrapper(
    *,
    shot: bool = True,
    minimum_restraint_probability: float = 0.90,
    minimum_pair_probability: float = 1.0,
    minimum_geometry_probability: float = 1.0,
    minimum_geometry_margin: float = 1.0,
) -> tuple[SequenceReplayPolicy, GoalConditionedSteeringPolicy]:
    base = _base_model(shot=shot)
    reference_model = _base_model(shot=shot)
    reference_model.load_state_dict(base.state_dict())
    model = SequenceReplayModel(
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
        geometry_candidate_count=32,
        geometry_candidate_set_sha256=GEOMETRY_SHA256,
    )
    geometry = SimpleNamespace(
        model=_GeometryModel(),
        geometry_config=GeometrySearchConfig(),
        sha256="2" * 64,
    )
    policy = SequenceReplayPolicy(
        model,
        _base_policy(base),
        geometry,
        checkpoint_sha256="3" * 64,
        source_identity="4" * 64,
        minimum_restraint_probability=minimum_restraint_probability,
        minimum_pair_probability=minimum_pair_probability,
        minimum_geometry_probability=minimum_geometry_probability,
        minimum_geometry_margin=minimum_geometry_margin,
    )
    return policy, _base_policy(reference_model)


class _MutatingStream:
    def __init__(self, delegate: object, mutate: object) -> None:
        self.delegate = delegate
        self.mutate = mutate

    @property
    def state(self) -> torch.Tensor:
        return self.delegate.state

    def reset(self) -> None:
        self.delegate.reset()

    def step(self, *args: object, **kwargs: object):
        return self.mutate(self.delegate.step(*args, **kwargs), kwargs)


class SequenceReplayPolicyTests(unittest.TestCase):
    def test_history_uses_only_the_previous_committed_transition(self) -> None:
        history = CausalEventHistory()
        first = _observation(0)
        shot = SteeringDecision(
            SemanticAction.strong(0.2, 0.5),
            SteeringIntent.STEER_MATCH,
            source_body_id=11,
            destination_body_id=22,
        )
        self.assertEqual(float(history.features(first).abs().sum()), 0.0)
        history.commit(first, shot)
        current = _transition(
            first,
            tick=16,
            score=140,
            gauge=760,
            joined=(11, 22),
            chain_id=9,
        )
        features = history.features(current)
        self.assertEqual(features[EVENT_INDEX["previous_action_shot"]], 1.0)
        self.assertEqual(features[EVENT_INDEX["pair_joined"]], 1.0)
        self.assertGreater(features[EVENT_INDEX["delta_score_scaled"]], 0.0)
        later_view = _transition(first, tick=20, score=100, gauge=800)
        self.assertEqual(
            history.features(later_view)[EVENT_INDEX["previous_action_shot"]],
            1.0,
        )
        history.commit(
            current,
            SteeringDecision(
                SemanticAction.wait(8),
                SteeringIntent.WAIT,
            ),
        )
        after_wait = history.features(
            _transition(current, tick=24, score=140, gauge=760)
        )
        self.assertEqual(after_wait[EVENT_INDEX["previous_action_wait"]], 1.0)
        self.assertEqual(after_wait[EVENT_INDEX["previous_action_shot"]], 0.0)
        history.reset()
        self.assertEqual(float(history.features(current).abs().sum()), 0.0)

    def test_history_is_invariant_to_id_and_chain_numeric_renaming(self) -> None:
        left = CausalEventHistory()
        right = CausalEventHistory()
        left_first = _observation(0, grouped_chain=7)
        right_first = _observation(
            0,
            identifiers=(111, 222, 333, 444),
            grouped_chain=91,
        )
        left.commit(
            left_first,
            SteeringDecision(
                SemanticAction.strong(0.2, 0.5),
                SteeringIntent.STEER_MATCH,
                source_body_id=11,
                destination_body_id=22,
            ),
        )
        right.commit(
            right_first,
            SteeringDecision(
                SemanticAction.strong(0.2, 0.5),
                SteeringIntent.STEER_MATCH,
                source_body_id=111,
                destination_body_id=222,
            ),
        )
        left_next = _transition(
            left_first,
            tick=16,
            score=140,
            gauge=760,
            joined=(11, 22),
            chain_id=8,
        )
        right_next = _transition(
            right_first,
            tick=16,
            score=140,
            gauge=760,
            joined=(111, 222),
            chain_id=123,
        )
        self.assertTrue(
            torch.equal(left.features(left_next), right.features(right_next))
        )

    def test_zero_initialized_policy_matches_frozen_v5_sequence_exactly(
        self,
    ) -> None:
        policy, reference = _wrapper()
        policy.reset(17)
        reference.reset(17)
        for tick in (0, 1, 4, 5, 8):
            observation = _observation(tick)
            self.assertEqual(
                policy.predict(observation),
                reference.predict(observation),
            )
        policy.reset(17)
        reference.reset(17)
        self.assertEqual(
            policy.predict(_observation(0)),
            reference.predict(_observation(0)),
        )

    def test_policy_decisions_ignore_id_and_chain_numeric_names(self) -> None:
        left, _ = _wrapper()
        right, _ = _wrapper()
        inverse_ids = {111: 11, 222: 22, 333: 33, 444: 44}
        left.reset(5)
        right.reset(5)
        for tick in (0, 1, 4, 5, 8):
            first = left.predict(_observation(tick, grouped_chain=7))
            second = right.predict(
                _observation(
                    tick,
                    identifiers=(111, 222, 333, 444),
                    grouped_chain=91,
                )
            )
            self.assertEqual(first.action, second.action)
            self.assertEqual(first.intent, second.intent)
            self.assertEqual(first.is_shot, second.is_shot)
            self.assertEqual(
                first.source_body_id,
                (
                    None
                    if second.source_body_id is None
                    else inverse_ids[second.source_body_id]
                ),
            )
            self.assertEqual(
                first.destination_body_id,
                (
                    None
                    if second.destination_body_id is None
                    else inverse_ids[second.destination_body_id]
                ),
            )

    def test_actual_pair_override_is_the_only_committed_attempt(self) -> None:
        policy, reference = _wrapper(minimum_pair_probability=0.80)
        observation = _observation(0)
        incumbent = reference.predict(observation)
        encoded = TeacherStateEncoder().encode([observation])
        identifiers = encoded_body_ids(encoded, observation)
        width = int(torch.from_numpy(encoded.body_mask[0]).nonzero()[-1]) + 1
        with torch.no_grad():
            base_output = policy.model.base_model(
                torch.from_numpy(encoded.global_features),
                torch.from_numpy(encoded.body_features[:, :width]),
                torch.from_numpy(encoded.body_mask[:, :width]),
            )
        alternatives = [
            (source, destination)
            for source, destination in base_output.legal_pair_mask[0].nonzero()
            if (
                identifiers[int(source)],
                identifiers[int(destination)],
            )
            != (incumbent.source_body_id, incumbent.destination_body_id)
        ]
        self.assertTrue(alternatives)
        source, destination = map(int, alternatives[0])

        def force_pair(output: object, kwargs: object):
            del kwargs
            logits = torch.full_like(output.pair_logits, -100.0)
            logits[0, 0, source, destination] = 100.0
            return replace(output, pair_logits=logits)

        policy.stream = _MutatingStream(policy.stream, force_pair)
        decision = policy.predict(observation)
        expected_pair = (identifiers[source], identifiers[destination])
        self.assertEqual(
            (decision.source_body_id, decision.destination_body_id),
            expected_pair,
        )
        pending = policy.base_policy._progress.pending_pair
        self.assertIsNotNone(pending)
        self.assertEqual(
            (pending.source_id, pending.destination_id),
            expected_pair,
        )
        self.assertEqual(policy.statistics()["pair_overrides"], 1)

    def test_learned_restraint_does_not_commit_the_unexecuted_base_shot(
        self,
    ) -> None:
        policy, _ = _wrapper(minimum_restraint_probability=0.80)

        def force_wait(output: object, kwargs: object):
            del kwargs
            logits = output.act_logits.clone()
            logits[0, 0] = torch.tensor([20.0, -20.0])
            return replace(output, act_logits=logits)

        policy.stream = _MutatingStream(policy.stream, force_wait)
        decision = policy.predict(_observation(0))
        self.assertFalse(decision.is_shot)
        self.assertIsNone(policy.base_policy._progress.pending_pair)
        self.assertEqual(policy.base_policy._cooldown_until, 0)
        event = policy.history.features(_observation(1))
        self.assertEqual(event[EVENT_INDEX["previous_action_wait"]], 1.0)

    def test_masks_gates_and_nonfinite_outputs_fall_back_closed(self) -> None:
        observation = _observation(0)

        policy, reference = _wrapper(minimum_pair_probability=0.0)
        incumbent = reference.predict(observation)

        def mask_every_pair(output: object, kwargs: object):
            del kwargs
            return replace(
                output,
                pair_logits=torch.full_like(output.pair_logits, 100.0),
                legal_pair_mask=torch.zeros_like(output.legal_pair_mask),
            )

        policy.stream = _MutatingStream(policy.stream, mask_every_pair)
        self.assertEqual(policy.predict(observation), incumbent)

        policy, reference = _wrapper()
        incumbent = reference.predict(observation)
        exercised = {"slot": None}

        def close_geometry_gate(output: object, kwargs: object):
            available = kwargs["geometry_candidate_mask"][0]
            slots = (available & (torch.arange(32) > 0)).nonzero().reshape(-1)
            self.assertTrue(bool(slots.numel()))
            slot = int(slots[0])
            exercised["slot"] = slot
            logits = torch.full_like(output.geometry_logits, -100.0)
            logits[0, 0, slot] = 100.0
            return replace(
                output,
                geometry_logits=logits,
                geometry_apply_mask=torch.zeros_like(
                    output.geometry_apply_mask
                ),
            )

        policy.stream = _MutatingStream(policy.stream, close_geometry_gate)
        self.assertEqual(policy.predict(observation), incumbent)
        self.assertIsNotNone(exercised["slot"])

        policy, reference = _wrapper()
        incumbent = reference.predict(observation)

        def inject_nan(output: object, kwargs: object):
            del kwargs
            logits = output.act_logits.clone()
            logits[0, 0, 0] = torch.nan
            return replace(output, act_logits=logits)

        policy.stream = _MutatingStream(policy.stream, inject_nan)
        self.assertEqual(policy.predict(observation), incumbent)
        self.assertEqual(
            policy.statistics()["nonfinite_output_fallbacks"], 1
        )
        self.assertEqual(float(policy.stream.state.abs().sum()), 0.0)

        wait_policy, _ = _wrapper(shot=False)

        def force_acceleration(output: object, kwargs: object):
            del kwargs
            logits = output.act_logits.clone()
            logits[0, 0] = torch.tensor([-20.0, 20.0])
            return replace(output, act_logits=logits)

        wait_policy.stream = _MutatingStream(
            wait_policy.stream, force_acceleration
        )
        self.assertFalse(wait_policy.predict(observation).is_shot)


if __name__ == "__main__":
    unittest.main()
