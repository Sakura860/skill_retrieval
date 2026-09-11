# Task 5 latency decomposition — 2026-09-03

## Scope and measurement

This report is computed only from immutable saved JSON evidence. It does not issue DeepSeek requests and does not modify or rerun the consumed confirmation set.

Per-task orchestration residual is `end_to_end - sum(LLM calls) - handler execution`. Percentiles use the same nearest stored observation rule as the project benchmark (`round((n-1)*p)`). Historical one-stage calls were stored as `skill_selection` because phase inference matched text inside the hierarchical context; this report preserves that source label but normalizes the sole one-stage call to `one_stage_joint_planning`.

## end_to_end_dev

Sources: 3 file(s), 10 task(s) per file, split `dev`.

| Method | Observations | Mean E2E ms | P50 E2E ms | Mean LLM ms | Mean handler ms | Mean residual ms | Planner calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_stage | 30 | 2,465.79 | 2,217.49 | 2,460.10 | 2.44 | 3.24 | 1.00 |
| adaptive_signals | 30 | 3,955.45 | 3,271.72 | 3,950.76 | 1.94 | 2.74 | 2.00 |

Paired `adaptive_signals - one_stage` mean deltas over 30 observations:

- End-to-end: 1,489.66 ms
- LLM calls: 1,490.65 ms
- Handler execution: -0.50 ms
- Orchestration residual: -0.50 ms
- Planner calls: 1.00

Candidate phase means per task:

- Selection: 1,399.04 ms
- Planning: 2,551.72 ms
- Repair: 0.00 ms

## disclosure_dev

Sources: 3 file(s), 10 task(s) per file, split `dev`.

| Method | Observations | Mean E2E ms | P50 E2E ms | Mean LLM ms | Mean handler ms | Mean residual ms | Planner calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_full | 30 | 3,237.92 | 3,193.04 | 3,231.74 | 3.00 | 3.18 | 2.00 |
| adaptive_signals | 30 | 3,186.35 | 3,083.22 | 3,180.35 | 2.86 | 3.14 | 2.00 |

Paired `adaptive_signals - always_full` mean deltas over 30 observations:

- End-to-end: -51.58 ms
- LLM calls: -51.38 ms
- Handler execution: -0.15 ms
- Orchestration residual: -0.05 ms
- Planner calls: 0.00

Candidate phase means per task:

- Selection: 1,433.80 ms
- Planning: 1,746.55 ms
- Repair: 0.00 ms

## confirmation

Sources: 1 file(s), 10 task(s) per file, split `confirmation`.

| Method | Observations | Mean E2E ms | P50 E2E ms | Mean LLM ms | Mean handler ms | Mean residual ms | Planner calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_stage | 10 | 2,033.77 | 1,932.96 | 2,027.61 | 1.80 | 4.36 | 1.00 |
| adaptive_signals | 10 | 3,579.34 | 3,205.55 | 3,575.02 | 1.84 | 2.48 | 2.10 |

Paired `adaptive_signals - one_stage` mean deltas over 10 observations:

- End-to-end: 1,545.56 ms
- LLM calls: 1,547.41 ms
- Handler execution: 0.04 ms
- Orchestration residual: -1.88 ms
- Planner calls: 1.10

Candidate phase means per task:

- Selection: 1,587.29 ms
- Planning: 1,844.30 ms
- Repair: 143.42 ms

## Conclusion

On confirmation, adaptive-signals adds 1,545.56 ms mean end-to-end latency per task. The LLM component accounts for 1,547.41 ms (100.1% of the paired mean increase); handler change is 0.04 ms and orchestration residual change is -1.88 ms.

Therefore the measured latency penalty is primarily attributable to the second model call, a structural cost of two-stage planning on this provider. Local handler and orchestration overhead are comparatively negligible. Connection reuse or serialization work may still be profiled later, but the current evidence does not support presenting the 65.8% P50 increase as mainly a local implementation defect.

The disclosure-only dev comparison is also important: always-full and adaptive-signals both use two stages and have nearly identical latency. Adaptive disclosure saves tokens, but it does not remove the extra network round trip inherent in two-stage planning.

## Validity boundary

- Confirmation contains 10 tasks; means and empirical percentiles are descriptive rather than precise population estimates.
- Calls were executed sequentially against one provider; parallel or local models may have a different structural latency trade-off.
- Residual time is calculated rather than independently instrumented, so it combines all non-LLM, non-handler orchestration work.
- No post-confirmation optimization was evaluated, because the consumed confirmation protocol cannot be rerun.
