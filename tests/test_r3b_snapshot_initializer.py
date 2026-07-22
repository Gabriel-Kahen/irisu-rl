from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import types
import unittest
from dataclasses import replace

from irisu_env import ActionKind
from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.collector import CurriculumTaskContract
from irisu_rl.curriculum import (
    CurriculumCoordinator,
    CurriculumSnapshotInitializer,
    CurriculumSpec,
    SnapshotBlobStore,
    SnapshotLibrary,
    SnapshotRecipe,
    StageSpec,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.rewards import (
    LinearGaugePotential,
    RewardComposer,
    RewardKnot,
    RewardSchedule,
)
from irisu_rl.vector_adapter import MacroVectorAdapter


_SNAPSHOT = struct.Struct("<qqqQ")
_CONFIG_HASH = 99
_CONFIG_SHA256 = "1" * 64
_RUNTIME_SHA256 = "2" * 64
_POOL = "snapshot-pool"


def _observation(
    tick: int,
    score: int,
    gauge: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
):
    return types.SimpleNamespace(
        tick=tick,
        score=score,
        gauge=gauge,
        gauge_max=1000,
        qualifying_clear_count=0,
        level=1,
        active_colors=3,
        spawn_interval_ticks=50,
        highest_chain=0,
        left_held=False,
        right_held=False,
        terminated=terminated,
        truncated=truncated,
        body_count=0,
        bodies=(),
    )


def _payload(tick: int, score: int, gauge: int, state_hash: int) -> bytes:
    return _SNAPSHOT.pack(tick, score, gauge, state_hash)


class FakeSnapshotVector:
    """Small stateful vector with the snapshot subset contract."""

    num_envs = 2

    def __init__(self) -> None:
        self.states = [(0, 0, 100, 1), (0, 0, 100, 2)]
        self.seeds = [0, 0]
        self.terminate_next: set[int] = set()
        self.bad_hash_once = False
        self.restore_many_calls: list[tuple[int, ...]] = []
        self.full_restore_calls = 0

    @staticmethod
    def _decode(snapshot: bytes) -> tuple[int, int, int, int]:
        tick, score, gauge, state_hash = _SNAPSHOT.unpack(snapshot)
        return int(tick), int(score), int(gauge), int(state_hash)

    @staticmethod
    def _encode(state: tuple[int, int, int, int]) -> bytes:
        return _payload(*state)

    @staticmethod
    def _observe(state: tuple[int, int, int, int], *, terminated: bool = False):
        return _observation(state[0], state[1], state[2], terminated=terminated)

    def reset(self, *, seed):
        self.seeds = list(seed)
        self.states = [(0, 0, 100, 10_000 + int(value)) for value in self.seeds]
        return [self._observe(value) for value in self.states], [
            {"seed": value, "config_hash": _CONFIG_HASH} for value in self.seeds
        ]

    def clone_state_many(self, indices):
        return tuple(self._encode(self.states[index]) for index in indices)

    def restore_many(self, indices, snapshots):
        lanes = tuple(indices)
        self.restore_many_calls.append(lanes)
        restored = [self._decode(snapshot) for snapshot in snapshots]
        for lane, state in zip(lanes, restored):
            self.states[lane] = state
        return [self._observe(value) for value in restored]

    def state_hash_many(self, indices):
        values = tuple(self.states[index][3] for index in indices)
        if self.bad_hash_once:
            self.bad_hash_once = False
            return (values[0] + 1, *values[1:])
        return values

    def config_hash_many(self, indices):
        return (_CONFIG_HASH,) * len(tuple(indices))

    def clone_state(self):
        return tuple(self._encode(value) for value in self.states)

    def restore_state(self, snapshots):
        self.full_restore_calls += 1
        restored = [self._decode(snapshot) for snapshot in snapshots]
        self.states = restored
        return [self._observe(value) for value in restored]

    def state_hash(self):
        return tuple(value[3] for value in self.states)

    def _step(self, indices, actions):
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = []
        for lane, action in zip(indices, actions):
            delta = int(action.wait_ticks) if action.kind == ActionKind.WAIT else 1
            tick, score, gauge, state_hash = self.states[lane]
            state = (
                tick + delta,
                score + delta,
                max(1, gauge - 10 * delta),
                state_hash + delta,
            )
            self.states[lane] = state
            is_terminal = lane in self.terminate_next
            observations.append(self._observe(state, terminated=is_terminal))
            rewards.append(delta)
            terminated.append(is_terminal)
            truncated.append(False)
            infos.append(
                {"events": (), "invalid_action": False, "config_hash": _CONFIG_HASH}
            )
        self.terminate_next.difference_update(indices)
        return observations, rewards, terminated, truncated, infos

    def step(self, actions):
        return self._step(range(self.num_envs), actions)

    def step_many(self, indices, actions):
        return self._step(indices, actions)

    def reset_many(self, indices, *, seeds):
        output = []
        for lane, seed in zip(indices, seeds):
            self.seeds[lane] = seed
            self.states[lane] = (0, 0, 100, 10_000 + int(seed))
            output.append(self._observe(self.states[lane]))
        return output


def _fixture() -> tuple[CurriculumSpec, dict[str, bytes]]:
    action_spec = ActionSpec()
    trace = (action_spec.serialize(SemanticAction.wait(1)).hex(),)
    definitions = (
        ("train-a", "train", "family-train-a", 10, 101, 1001),
        ("train-b", "train", "family-train-b", 20, 202, 2002),
        ("validation", "validation", "family-validation", 30, 303, 3003),
    )
    blobs = {
        name: _payload(tick, tick, gauge, state_hash)
        for name, _, _, tick, gauge, state_hash in definitions
    }
    recipes = tuple(
        SnapshotRecipe(
            name,
            "stage",
            split,
            family,
            _POOL,
            _CONFIG_SHA256,
            _CONFIG_HASH,
            100 + index,
            action_spec.sha256,
            trace,
            tick,
            tick,
            state_hash,
            hashlib.sha256(blobs[name]).hexdigest(),
            _RUNTIME_SHA256,
            "snapshot-test-v1",
        )
        for index, (name, split, family, tick, _gauge, state_hash) in enumerate(
            definitions
        )
    )
    library = SnapshotLibrary(recipes)
    stage = StageSpec(
        "stage",
        0,
        _POOL,
        ("train-a", "train-b"),
        ("validation",),
        (0,),
        (1,),
        1,
        1,
        1,
        1,
        1,
        20,
        RewardSchedule(
            "snapshot-weight-v1",
            (RewardKnot(0, 500_000), RewardKnot(5, 0)),
        ),
    )
    return CurriculumSpec("snapshot-test-v1", library, (stage,), 77), blobs


def _changing_learner_seed(spec: CurriculumSpec) -> int:
    for learner_seed in range(100):
        coordinator = CurriculumCoordinator(spec, 2, learner_seed=learner_seed)
        first = coordinator.reserve_assignments((0, 1))
        initial = first.assignments[0].snapshot_id
        coordinator.commit_assignments(first)
        second = coordinator.reserve_assignments((0,))
        following = second.assignments[0].snapshot_id
        coordinator.rollback_assignments(second)
        if initial != following:
            return learner_seed
    raise AssertionError("fixture could not find a changing lane assignment")


def _components():
    spec, blobs = _fixture()
    coordinator = CurriculumCoordinator(
        spec, 2, learner_seed=_changing_learner_seed(spec)
    )
    store = SnapshotBlobStore(spec.library, blobs)
    initializer = CurriculumSnapshotInitializer(
        coordinator,
        store,
        environment_pool=_POOL,
        runtime_identity_sha256=_RUNTIME_SHA256,
    )
    return spec, coordinator, store, initializer


class SnapshotInitializerTests(unittest.TestCase):
    def test_snapshot_library_manifest_round_trips_strictly(self) -> None:
        spec, _ = _fixture()
        manifest = spec.library.manifest()
        self.assertEqual(
            SnapshotLibrary.from_manifest(manifest).sha256,
            spec.library.sha256,
        )
        unknown = dict(manifest)
        unknown["unplanned"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            SnapshotLibrary.from_manifest(unknown)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/library.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            self.assertEqual(
                SnapshotLibrary.from_json(path).sha256,
                spec.library.sha256,
            )

    def test_assignment_stream_excludes_reward_schedule_identity(self) -> None:
        spec, _ = _fixture()
        zero_schedule = RewardSchedule("score-only-v1", (RewardKnot(0, 0),))
        score_only = replace(
            spec,
            stages=(replace(spec.stages[0], reward_schedule=zero_schedule),),
        )
        self.assertNotEqual(spec.sha256, score_only.sha256)
        self.assertEqual(spec.assignment_sha256, score_only.assignment_sha256)
        shaped = CurriculumCoordinator(spec, 2, learner_seed=31)
        control = CurriculumCoordinator(score_only, 2, learner_seed=31)
        self.assertEqual(
            shaped.reserve_assignments((0, 1)),
            control.reserve_assignments((0, 1)),
        )

    def test_blob_store_rejects_incomplete_corrupt_and_nonowned_payloads(self) -> None:
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        self.assertEqual(set(store.manifest()["blobs"]), set(blobs))
        self.assertEqual(store["train-a"], blobs["train-a"])

        incomplete = dict(blobs)
        incomplete.pop("validation")
        with self.assertRaisesRegex(ValueError, "blob set"):
            SnapshotBlobStore(spec.library, incomplete)

        corrupt = dict(blobs)
        corrupt["train-a"] += b"corrupt"
        with self.assertRaisesRegex(ValueError, "blob hash"):
            SnapshotBlobStore(spec.library, corrupt)

        nonowned = dict(blobs)
        nonowned["train-a"] = bytearray(nonowned["train-a"])  # type: ignore[assignment]
        with self.assertRaisesRegex(TypeError, "owned bytes"):
            SnapshotBlobStore(spec.library, nonowned)

    def test_full_initialization_commits_but_subset_commit_is_deferred(self) -> None:
        _, coordinator, _, initializer = _components()
        env = FakeSnapshotVector()
        env.reset(seed=(1, 2))

        initial = initializer.initialize(env, (0, 1), defer_commit=False)
        self.assertEqual(initial.lane_ids, (0, 1))
        self.assertEqual(coordinator.episode_ordinals, [1, 1])
        self.assertEqual(tuple(coordinator.lane_snapshot_id), initial.episode_labels)
        self.assertFalse(initializer.has_pending)

        old_labels = tuple(coordinator.lane_snapshot_id)
        pending = initializer.initialize(env, (1,), defer_commit=True)
        self.assertTrue(initializer.has_pending)
        self.assertEqual(coordinator.episode_ordinals, [1, 1])
        self.assertEqual(tuple(coordinator.lane_snapshot_id), old_labels)
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            initializer.initialize(env, (0,), defer_commit=True)
        with self.assertRaisesRegex(ValueError, "completed lanes"):
            initializer.commit_pending((0,))
        self.assertTrue(initializer.has_pending)

        initializer.commit_pending((1,))
        self.assertFalse(initializer.has_pending)
        self.assertEqual(coordinator.episode_ordinals, [1, 2])
        self.assertEqual(coordinator.lane_snapshot_id[1], pending.episode_labels[0])

    def test_identity_failure_rolls_back_state_and_cancels_assignment(self) -> None:
        spec, coordinator, _, initializer = _components()
        env = FakeSnapshotVector()
        env.reset(seed=(11, 12))
        before = env.clone_state()
        expected = CurriculumCoordinator(
            spec, 2, learner_seed=coordinator.learner_seed
        ).reserve_assignments((0, 1))

        env.bad_hash_once = True
        with self.assertRaisesRegex(ValueError, "snapshot identity"):
            initializer.initialize(env, (0, 1), defer_commit=False)
        self.assertEqual(env.clone_state(), before)
        self.assertEqual(coordinator.episode_ordinals, [0, 0])
        self.assertFalse(initializer.has_pending)

        actual = initializer.initialize(env, (0, 1), defer_commit=False)
        self.assertEqual(
            actual.episode_labels,
            tuple(value.snapshot_id for value in expected.assignments),
        )
        self.assertEqual(coordinator.episode_ordinals, [1, 1])
        self.assertEqual(env.restore_many_calls[-1], (0, 1))

    def test_encoder_failure_rolls_back_uncommitted_initial_assignment(self) -> None:
        class FailingEncoder:
            def encode(self, observations):
                del observations
                raise ValueError("synthetic encoder failure")

        spec, coordinator, _, initializer = _components()
        expected = CurriculumCoordinator(
            spec, 2, learner_seed=coordinator.learner_seed
        ).reserve_assignments((0, 1))
        env = FakeSnapshotVector()
        adapter = MacroVectorAdapter(
            env,
            encoder=FailingEncoder(),  # type: ignore[arg-type]
            episode_initializer=initializer,
        )
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            adapter.reset()
        self.assertEqual(coordinator.episode_ordinals, [0, 0])
        self.assertEqual(coordinator.lane_snapshot_id, ["", ""])
        self.assertFalse(initializer.has_pending)

        retry = initializer.initialize(env, (0, 1), defer_commit=False)
        self.assertEqual(
            retry.episode_labels,
            tuple(value.snapshot_id for value in expected.assignments),
        )

    def test_terminal_transition_keeps_old_identity_and_weight_until_commit(
        self,
    ) -> None:
        _, coordinator, _, initializer = _components()
        env = FakeSnapshotVector()
        adapter = MacroVectorAdapter(
            env,
            encoder=TeacherStateEncoder(),
            episode_initializer=initializer,
        )
        task = CurriculumTaskContract(
            coordinator,
            RewardComposer(shaping_spec=LinearGaugePotential()),
            capture_events=False,
            snapshot_initializer=initializer,
        )
        adapter.reset()
        old_label = coordinator.lane_snapshot_id[0]
        old_weight = coordinator.lane_shaping_weight_ppm[0]
        old_tick = int(adapter.current_observation.source_tick[0])
        for _ in range(5):
            coordinator.advance_update()

        env.terminate_next.add(0)
        transitions = adapter.step((SemanticAction.wait(1), SemanticAction.wait(1)))
        self.assertTrue(initializer.has_pending)
        self.assertTrue(transitions[0].terminated)
        self.assertEqual(transitions[0].episode_label, old_label)
        self.assertEqual(coordinator.lane_snapshot_id[0], old_label)
        self.assertEqual(
            int(transitions[0].final_observation.source_tick[0]), old_tick + 1
        )
        next_tick = int(transitions[0].next_policy_observation.source_tick[0])
        new_label = next(
            recipe.snapshot_id
            for recipe in coordinator.spec.library.recipes
            if recipe.expected_tick == next_tick
        )
        self.assertNotEqual(new_label, old_label)

        rewards = task.rewards(transitions)
        self.assertEqual(int(rewards.shaping_weight_ppm[0]), old_weight)
        self.assertEqual(coordinator.lane_shaping_weight_ppm[0], old_weight)
        with self.assertRaisesRegex(RuntimeError, "uncommitted"):
            adapter.checkpoint()

        task.after_transitions(transitions)
        self.assertFalse(initializer.has_pending)
        self.assertEqual(coordinator.lane_snapshot_id[0], new_label)
        self.assertEqual(coordinator.lane_shaping_weight_ppm[0], 0)
        checkpoint = adapter.checkpoint()
        self.assertEqual(checkpoint.episode_labels[0], new_label)

    def test_checkpoint_validates_active_snapshot_labels_before_restore(self) -> None:
        spec, coordinator, store, initializer = _components()
        env = FakeSnapshotVector()
        adapter = MacroVectorAdapter(
            env,
            encoder=TeacherStateEncoder(),
            episode_initializer=initializer,
        )
        adapter.reset()
        checkpoint = adapter.checkpoint()
        coordinator_state = coordinator.state_dict()

        restored_coordinator = CurriculumCoordinator(
            spec, 2, learner_seed=coordinator.learner_seed
        )
        restored_coordinator.load_state_dict(coordinator_state)
        restored_initializer = CurriculumSnapshotInitializer(
            restored_coordinator,
            store,
            environment_pool=_POOL,
            runtime_identity_sha256=_RUNTIME_SHA256,
        )
        restored_env = FakeSnapshotVector()
        restored_adapter = MacroVectorAdapter(
            restored_env,
            encoder=TeacherStateEncoder(),
            episode_initializer=restored_initializer,
        )
        restored_adapter.reset(disposable=True)
        bad_labels = ("validation", checkpoint.episode_labels[1])
        tampered = replace(checkpoint, episode_labels=bad_labels)
        before = restored_env.clone_state()
        with self.assertRaisesRegex(ValueError, "active snapshots"):
            restored_adapter.restore_checkpoint(tampered)
        self.assertEqual(restored_env.clone_state(), before)
        self.assertEqual(restored_env.full_restore_calls, 0)

        restored_adapter.restore_checkpoint(checkpoint)
        self.assertEqual(restored_env.full_restore_calls, 1)
        self.assertEqual(
            restored_adapter.checkpoint().episode_labels,
            checkpoint.episode_labels,
        )


if __name__ == "__main__":
    unittest.main()
