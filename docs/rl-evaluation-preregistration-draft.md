# Expert-human evaluation preregistration (draft)

Status: R0 draft, not locked. Target: authorized IriSu Syndrome v2.03 normal
mode at normal speed. This draft cannot authorize a final claim until every
provisional item below has measured evidence and the document is frozen before
evaluation.

## Primary question and metric

The primary comparison is per-run final human-visible/persisted score between
the locked agent checkpoint and an expert-human cohort under the same input,
display, version, timing, and restart rules. The primary statistic, uncertainty
interval, sample size, superiority/non-inferiority rule, and handling of tied or
aborted runs will be fixed after a blinded pilot and before final data exists.
No run may be retried or discarded after its outcome is visible.

## Information and controls

The agent receives only puzzle-region pixels, capture timestamps, and causal
state from its own input bridge. It receives no process memory, simulator state,
replay state, RNG, future frames, exact IDs, or hidden timers. It emits only
legal cursor movement and weak/strong clicks. Fast-forward and simultaneous
buttons are excluded unless the final protocol permits them equally for human
and agent participants.

## Data partition and locking

Training, validation, perception calibration, and final test data use the
immutable disjoint seed/data partitions in `configs/rl/seeds/v1.json`. Final
runs are generated and ordered before execution. The actor schema, perception
model, checkpoint, recurrent reset behavior, input bridge, coordinate transform,
and all hyperparameters are hashed and locked before the first final run.

## Reliability and reporting

Report every scheduled run, raw score distribution, median/mean and declared
intervals, termination mode, perception/input health failures, intervention or
abort, and wall-clock duration. Report simulator-only and original-game results
separately. A maximum score is secondary and cannot establish superiority.

## Open decisions that block locking

- authoritative terminal score (live HUD versus persisted result/replay);
- expert inclusion rule and cohort size;
- final run count and statistical test/effect margin;
- cursor travel, click-rate, press/release, capture-rate, and latency limits;
- hardware/display/window configuration and allowed recovery from technical
  failure;
- blinded pilot size and frozen perception-health exclusion criteria.

Owners are the evaluation lead (statistics/cohort), R4 input/perception lead
(timing and pixels), and fidelity lead (version/evidence). Each decision is due
before final checkpoint selection or access to final outcomes, whichever comes
first.
