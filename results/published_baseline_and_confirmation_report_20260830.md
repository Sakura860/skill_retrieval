# Published baseline and frozen confirmation report

## Scope

This report closes two previously open items without changing the held-out
confirmation tasks during development:

1. Run a no-fallback published SkillRouter baseline on the local development
   benchmark under the same ranking metrics as the internal controls.
2. After freezing the final end-to-end protocol, consume the confirmation set
   exactly once and report the preregistered decision.

The local retrieval benchmark contains 24 skills and 10 development tasks. The
confirmation benchmark contains a separate 10 tasks from the same five task
families. Confirmation tasks were not used to construct or revise the
`adaptive_signals` policy.

## Published retrieval baseline

Protocol: `published_baselines_v02`; CPU, float32; official SkillRouter commit
`2f0c69fe6786bfee6312a1ab5d5f69abdc6bd245`; official 0.6B encoder and
reranker; strict top-20 reranking; no hash or substitute-model fallback.

| Method | MRR | Recall@10 | NDCG@10 | Mean query latency |
|---|---:|---:|---:|---:|
| BM25 brief | 0.5318 | 0.7500 | 0.5326 | 0.22 ms |
| BM25 all fields | 1.0000 | 1.0000 | 0.9695 | 0.35 ms |
| SkillRouter encoder 0.6B | 1.0000 | 1.0000 | 0.9832 | 268.93 ms |
| SkillRouter encoder + reranker 0.6B | 1.0000 | 1.0000 | 0.9920 | 24,143.80 ms |

The full pipeline improves NDCG@10 over encoder-only by 0.0088 absolute but is
89.8 times slower in mean online query latency on this CPU. It is therefore an
external accuracy ceiling, not the default route for this project. The encoder
is the primary published retrieval baseline.

This is a small-pool result with a clear ceiling effect: BM25 all fields and
both published variants already reach MRR 1.0 and Recall@10 1.0. It supports
component-level comparison but does not establish large-pool scalability or a
statistically precise superiority claim.

Latency v02 separates one-time model preparation, skill-pool indexing, and
online query latency. Two earlier pipeline files are retained but superseded:
v01 included lazy reranker download/loading in the first query and both earlier
files reranked 24 candidates despite declaring top 20.

## Frozen end-to-end confirmation

The real BM25 top-10 protocol was changed from
`development_preregistered` to `frozen` without changing controlled settings,
task IDs, data hashes, methods, or decision thresholds. A persistent registry
reserved one logical confirmation run; an interrupted run would only have been
resumable at the same output path.

| Method | Task success | Selection F1 | Sequence accuracy | Total tokens | P50 latency |
|---|---:|---:|---:|---:|---:|
| one-stage | 90% (9/10) | 0.9667 | 0.9000 | 8,469 | 1,932.96 ms |
| adaptive-signals | 100% (10/10) | 1.0000 | 1.0000 | 8,040 | 3,205.55 ms |

Against the frozen one-stage control, adaptive-signals has:

- exactly +10 percentage points task success;
- token ratio 0.9493, so it uses 5.1% fewer total tokens;
- P50 latency ratio 1.6584, below the preregistered maximum of 1.8.

All three required conditions pass. The resolved levels were stable with the
development pattern: 6 brief, 2 schema, and 2 full. One task used the permitted
hidden-Schema validation repair.

The only one-stage failure, `v2confirm05`, selected the correct two-step skill
sequence but supplied `input` where the handler Schema requires `data`.
Adaptive-signals disclosed Schema for this multi-skill dataflow case and passed
the deterministic verifier. This independently confirms the development-set
diagnosis that selection and parameter generation require different
information access.

## Decision audit

The immutable confirmation result initially stored the threshold check as
`fail` because Python represented the exact 1/10 gain as
`0.09999999999999998`. The result was not changed and the confirmation was not
rerun. A separate audit uses exact success counts and records the corrected
decision as `pass`; the comparison code now has a `1e-12` numerical tolerance
and a boundary regression test.

## Remaining external-validity work

The released SkillRouter public benchmark contains 75 scored tasks and
78,361/79,141 skills in its easy/hard pools. It has not been run locally: the
strict 0.6B encoder needed 13.79 seconds to index only 24 skills after model
preparation on this CPU, making the two-tier public run disproportionate here.
Run the unchanged official protocol on CUDA as a separate reciprocal
external-dataset validation. It is a limitation, not a reason to substitute a
hash baseline or to rerun the consumed confirmation set.

## Evidence files

- `results/published_baselines_controls_encoder_dev_v02_20260830.json`
- `results/published_baselines_skillrouter_pipeline_dev_v02_top20_20260830.json`
- `results/published_baseline_environment_20260830.json`
- `results/task5_confirmation_frozen_20260830.json`
- `results/task5_confirmation_registry.json`
- `results/task5_confirmation_decision_audit_20260830.json`
- `configs/task5_freeze_candidate_20260830.json`
