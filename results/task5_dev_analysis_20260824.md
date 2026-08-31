# Task 5 dev analysis — 2026-08-24

## Scope

All reported model runs use DeepSeek V4-Pro with thinking disabled and temperature 0. The Planner comparison uses the same 18 `benchmark_v01` dev tasks, fixed brief-BM25 candidate fixtures, task context budgets, isolated initial environments, handlers and deterministic verifiers. The graph comparison is separate and uses only the two annotated `benchmark_v02` dev prerequisite cases. No held-out test task was executed.

## Body-aware retrieval on benchmark_v02 dev

| Indexed Skill fields | Hit@1 | Recall@3 | Recall@5 | MRR | Mean best-gold rank |
|---|---:|---:|---:|---:|---:|
| brief | 40.00% | 40.00% | 75.00% | 0.5318 | 4.40 |
| name + brief + detailed | 100.00% | 95.00% | 95.00% | 1.0000 | 1.00 |
| all-field body | 100.00% | 90.00% | 95.00% | 1.0000 | 1.00 |

Detailed functional boundaries provide the main gain. Adding every Schema/metadata field does not improve Hit@1 or MRR on this set and slightly reduces Recall@3 for multi-Skill tasks. Therefore the current evidence supports body-aware retrieval, but not the stronger claim that all-field indexing is always best.

## Planner comparison on benchmark_v01 dev

| Method | Task Success | Selection F1 | Sequence Accuracy | Avg. Skill Context | Total Tokens | Avg. Planner Calls | Avg. Repairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-stage | 72.22% | 100.00% | 100.00% | 807.50 | 14,513 | 1.000 | 0.000 |
| two-stage, no repair | 88.89% | 88.89% | 88.89% | 541.67 | 15,447 | 2.000 | 0.000 |
| two-stage, at most one repair | 100.00% | 100.00% | 100.00% | 541.67 | 15,795 | 2.056 | 0.056 |

Two-stage disclosure with one repair raises Task Success by 27.78 percentage points and reduces Skill Context by 32.92% relative to one-stage. It increases total model tokens by 8.83% because it normally makes two Planner calls. Only `tjson03` used the repair call in the final run.

One-stage failures were `tcalc06`, `tjson01`, `tjson03`, `tjson06` and `tdb03`, primarily due to wrong parameter names or binding the whole instruction to structured inputs. No-repair reduced the failures to `tjson03` and `tjson06`; both were rejected before execution instead of reaching handlers with invalid arguments.

Array item Schemas and `$last_output` compatibility checks also exposed an invalid `select_sqlite_rows → sort_json_records` chain for `tdb01`: SQLite returns arrays of rows, while the JSON sorter requires arrays of objects. The updated planning/repair prompt allows dropping this redundant step and expressing ordering in the SQL query.

## Typed graph prerequisite completion on benchmark_v02 dev

| Method | Task Success | Selection F1 | Sequence Accuracy | Avg. Skill Context | Total Tokens | Added Skills |
|---|---:|---:|---:|---:|---:|---:|
| no completion | 0.00% | 58.33% | 0.00% | 1,919.50 | 3,183 | 0 |
| prerequisite completion, max +2 | 100.00% | 100.00% | 100.00% | 2,122.50 | 3,479 | 1 per task |

The graph condition changes only prerequisite completion. It adds `sfile_copy` for `v2graph01` and `sfile_create` for `v2graph02`, increasing average Skill Context by 203 tokens and total tokens by 296. Alternative, conflict and co-use relations are not allowed to expand candidates.

## Validity boundary and next gate

These are single complete dev runs, not repeated estimates. Before freezing the method and touching the held-out test split, repeat the final Planner configurations, add a repeated graph run, and confirm that the result is stable rather than a single-run sampling effect.
