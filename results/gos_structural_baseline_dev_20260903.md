# Graph-of-Skills matched structural baseline

This dev-only diagnostic uses the pinned official reverse-aware PPR runtime with fixed task candidate seeds. It is not the full hybrid GoS pipeline.

- Protocol: `gos-structural-dev-diagnostic-20260903-v01`
- Official upstream: `203f60a2c689da055ce1ac351eb3cb9912a3bca7`
- Independent tasks: 2; model repetitions: 3
- Same task candidates, graph edges, maximum two additions, context budget, model, and Planner settings for every method.

| Method | Prereq recall | Correct add | Irrelevant add | Growth | Context tokens | LLM tokens | Task success | Mean e2e ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_completion | 0.000 | 0.00 | 0.00 | 0.00 | 1919.5 | 9948 | 0.000 | 3860.44 |
| typed_prerequisite_completion | 1.000 | 1.00 | 0.00 | 1.00 | 2122.5 | 10408 | 1.000 | 2233.53 |
| gos_reverse_ppr_prerequisite | 1.000 | 1.00 | 1.00 | 2.00 | 2338.0 | 11322 | 1.000 | 2424.13 |
| gos_reverse_ppr_prerequisite_plus_dataflow | 1.000 | 1.00 | 1.00 | 2.00 | 2362.0 | 11368 | 1.000 | 2176.34 |

## Per-task structural outputs

The added IDs below are deterministic and identical across the three model repetitions; task success is listed as successes/3.

| Method | Task | Added | Correct | Irrelevant | Successes/3 |
|---|---|---|---|---|---:|
| no_completion | v2graph01 | - | - | - | 0/3 |
| no_completion | v2graph02 | - | - | - | 0/3 |
| typed_prerequisite_completion | v2graph01 | sfile_copy | sfile_copy | - | 3/3 |
| typed_prerequisite_completion | v2graph02 | sfile_create | sfile_create | - | 3/3 |
| gos_reverse_ppr_prerequisite | v2graph01 | sfile_copy,scalc_int | sfile_copy | scalc_int | 3/3 |
| gos_reverse_ppr_prerequisite | v2graph02 | sfile_create,scalc_int | sfile_create | scalc_int | 3/3 |
| gos_reverse_ppr_prerequisite_plus_dataflow | v2graph01 | sfile_copy,scalc_int | sfile_copy | scalc_int | 3/3 |
| gos_reverse_ppr_prerequisite_plus_dataflow | v2graph02 | sfile_create,scalc_int | sfile_create | scalc_int | 3/3 |

## Interpretation

- Exact typed completion follows only prerequisite edges and stops after the required closure; it should therefore be read as a high-precision local operator, not a general graph ranker.
- Official reverse-PPR can recover a prerequisite through reverse dependency propagation, but a fixed top-N bundle may also hydrate unrelated zero/low-score nodes when the graph is sparse. The irrelevant-addition metric makes that cost explicit.
- Adding dataflow as GoS `workflow` edges produced no aggregate change on these two tasks. Each dataflow edge is collinear with its prerequisite edge, so this dataset provides no independent evidence that dataflow improves success.
- `co_use`, `alternative`, and `conflict` are not tested here and receive no effectiveness claim.

## Reproduction and validity boundary

- The adapter imports the exact pinned official `query.py` after checking both git commit and SHA-256; it does not copy or reimplement PPR.
- Full GoS hybrid semantic/lexical seeding was not run because no compatible embedding credential/workspace is configured. This missing result is not replaced with Hash embeddings.
- Only two independent dev tasks are used. Three DeepSeek repetitions assess runtime stability but are not six independent tasks; no held-out or broad superiority claim is made.
- No consumed Task 5 confirmation task or result was used to tune or run this baseline.
