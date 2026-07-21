# IriSu Syndrome Superhuman Puzzle Agent Plan

## Objective

Build an agent that plays the normal puzzle mode of IriSu Syndrome at a demonstrably superhuman level while using only information and controls available to a human: puzzle-area pixels, mouse position, and legal left/right clicks.

The project will use a fast headless simulator for most training, then transfer the resulting policy to the original game and validate it there. A simulator-only high score is not sufficient evidence of success.

## Start Here

The repository begins with the headless clone. Its consolidated findings, recommended implementation workflow, first work package, requirements, interfaces, validation gates, and definition of done are in [clone.md](./clone.md). Treat that file as the authoritative clone handoff.

The next agent should:

1. Read `clone.md` in full, including its current-findings snapshot and ordered first work package.
2. Read [`reference/README.md`](./reference/README.md), [`reference/computer-use.md`](./reference/computer-use.md), [`reference/environment.md`](./reference/environment.md), [`reference/mechanics-evidence.md`](./reference/mechanics-evidence.md), [`reference/binary-analysis.md`](./reference/binary-analysis.md), [`reference/version-history.md`](./reference/version-history.md), and [`reference/manifest.md`](./reference/manifest.md). The local reference lab already contains a runnable v2.03 copy, pristine archives, replays, gameplay recordings, extracted shipped constants, recovered wrapper/replay/archive details, version and benchmark research, a recorded runtime environment, and a controlled computer-use protocol.
3. Inspect the repository and preserve any existing work.
4. Write a short implementation checklist that maps directly to the clone milestones.
5. Begin the deterministic simulation vertical slice and the reference-game measurement harness together.
6. Treat undocumented mechanics as explicit unknowns, not facts. Make them configurable until validated against the original game.

Do not begin large-scale RL training until the clone passes the readiness gates in `clone.md`.

## Success Criteria

The final deployed agent must:

- Observe only the rendered puzzle region or a causal object representation derived from it.
- Act through legal cursor coordinates and weak/strong clicks.
- Respect a fixed input-rate limit used by the human benchmark.
- Receive no RNG state, future spawn information, process memory, save states, or seed-selection privileges.
- Run at normal game speed during evaluation.
- Exceed an expert-human score distribution on a locked evaluation protocol.

Before training, record the exact definition of "superhuman," including the human cohort or replay set, input-rate rules, number of games, and statistical test. The preferred criterion is that the lower 95% confidence bound for the agent's paired mean or median score exceeds the corresponding expert-human result over a large held-out evaluation set. Also report lower-tail reliability; one exceptional run does not qualify.

## System Architecture

The intended system has four distinct layers:

1. **Reference harness**: runs the original Windows game, applies controlled inputs, captures frames, and preserves replay/video evidence.
2. **Headless clone**: deterministic, accelerated implementation of puzzle physics, spawning, scoring, gauge behavior, and termination.
3. **Teacher agent**: trains with privileged simulator object state and may use simulator search.
4. **Deployment agent**: receives only causal observations derived from original-game pixels and emits legal mouse actions.

Keep these boundaries explicit. In particular, simulator-only information must not leak into the deployment agent or final evaluation.

## Phase 1: Build and Validate the Clone

Follow [clone.md](./clone.md).

Important outputs include:

- A fixed-timestep, seeded, headless simulation.
- State snapshot, restore, and hash support.
- A vectorized Python training API.
- Configurable mechanics constants with documented provenance.
- Unit tests for state transitions and scoring.
- Golden tests derived from controlled original-game experiments.
- Replayable action traces and useful debug rendering.
- A fidelity report and a throughput benchmark.

The original executable, copyrighted assets, and personal replay collections must remain outside version control unless their licenses clearly permit inclusion.

## Phase 2: Establish Baselines

Create increasingly capable baselines before advanced training:

1. Random-action policy as an API and determinism smoke test.
2. Scripted policies for direct matching, side ejection, and obvious hazard removal.
3. A recurrent hybrid-action PPO or SAC policy trained on object state.
4. Evaluation on fixed simulator seeds and calibrated reference scenarios.

The policy action should be semi-Markov:

```text
wait(number_of_ticks)
fire(cursor_x, cursor_y, weak)
fire(cursor_x, cursor_y, strong)
```

This avoids forcing the model to emit uninformative no-ops at the render frame rate. Fast-forward is excluded initially and should only be introduced if it is part of the final benchmark.

## Phase 3: Curriculum

Train mechanics in a controlled progression:

1. Strike a single body toward a target region.
2. Redirect and juggle active bodies.
3. Eject dangerous bodies through legal side openings.
4. Match one fresh same-color pair.
5. Clear a fresh body against a same-color rotten body.
6. Use a bonus orb correctly.
7. Construct increasingly large direct chains.
8. Avoid wasting confirmed chains through excessive direct hits.
9. Play slow two-color boards.
10. Add colors, object variations, gauge pressure, and faster spawn phases.
11. Train complete games.
12. Restart from difficult states collected from policy failures.

Curriculum environments must use the same mechanics core as the full game. Do not create simplified behavior that silently differs from the real rules.

## Phase 4: Reward and Learning Objective

The permanent reward is the real score delta:

```text
reward_t = score_(t+1) - score_t
```

Normalize rewards for optimization without clipping away meaningful chain differences. Early curriculum stages may add potential-based shaping for gauge preservation, imminent rot risk, and terminal failure. Anneal this shaping to zero before final policy improvement.

Do not permanently reward elapsed survival time, raw match count, low click count, or avoiding side ejection. Those proxies can change the optimal policy or encourage stalling.

## Phase 5: Planning-Based Teacher

Once the model-free baseline is stable, improve it using the exact simulator:

1. A policy network proposes promising legal shots and waits.
2. CEM, sampled tree search, or another hybrid-action search evaluates short sequences using cloned simulator states.
3. A learned distributional value function evaluates states beyond the search horizon.
4. Search targets and realized returns train the policy and value networks.
5. Elite states and failure states receive prioritized replay.
6. Repeat search and distillation until held-out performance plateaus.

Search must not inspect the true hidden future RNG. Evaluate candidate actions over sampled possible future spawns. If planning is too expensive for deployment, distill the teacher into a real-time policy.

## Phase 6: Pixel-to-Action Transfer

Build a causal perception pipeline for the original game:

- Capture only the puzzle region.
- Detect object class, color, geometry, lifecycle state, and bonus objects.
- Track detections across frames to estimate linear and angular motion.
- Estimate global gauge, score, and level without process-memory access.
- Preserve uncertainty for missed, merged, or ambiguous detections.

Train the deployment policy with randomized rendering, observation noise, coordinate jitter, imperfect state estimates, dropped detections, and realistic input/vision latency. Begin with teacher distillation on noisy simulated observations, then use real-game recordings and failures for further adaptation.

Direct end-to-end pixel RL against the original executable is a fallback, not the primary training method.

## Phase 7: Evaluation

Maintain immutable training, validation, and final evaluation sets. Never select or reset unfavorable evaluation seeds.

Report at least:

- Mean and median score.
- 10th, 90th, and 99th percentile score.
- Confidence intervals.
- Survival time and highest level reached.
- Game-over rate.
- Gauge loss and rot events.
- Chain-length distribution.
- Performance with input latency and visual noise.
- Simulator score versus original-game score.

Compare the agent and expert humans under the same observation, timing, and input constraints whenever possible. Preserve action traces and videos for auditability.

## Main Risks

### Simulator mismatch

The policy may exploit inaccurate collision, boundary, scoring, or spawn behavior. Mitigate this through short-horizon original-game calibration, parameter randomization, continuous real-game validation, and regression tests for every discovered discrepancy.

### Long-horizon physical chaos

Exact replay trajectories will eventually diverge even with a strong clone. Judge fidelity primarily through short-horizon motion, discrete event agreement, score/gauge transitions, and policy transfer—not indefinite pixel synchronization.

### Reward hacking

Shaped rewards can teach stalling or safe but low-scoring behavior. Keep final optimization and model selection tied to real undiscounted score under the locked evaluation protocol.

### Privileged-information leakage

Teacher access to exact velocities and body state is allowed during training. Future RNG, hidden future spawns, and internal state unavailable from causal pixels must never reach the deployed policy or final evaluator.

## Project Completion

The project is complete only when:

1. The clone meets its documented fidelity and performance standards.
2. A policy trained primarily in the clone transfers to the original game.
3. The final agent follows the locked human-comparable interface.
4. The statistical superhuman criterion is met on the original game.
5. Results are reproducible from versioned code, configuration, model metadata, and evaluation traces.
