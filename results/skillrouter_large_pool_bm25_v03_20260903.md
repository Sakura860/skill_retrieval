# SkillRouter large-pool scaling — frozen protocol report

Protocol: `skillrouter-large-pool-20260903-v03`; status `frozen`. This is retrieval-only: no LLM Planner or Agent calls were issued.

Protocol history: v01 preflight failed before indexing because it hashed Git-bundled metadata copies. v02 completed BM25, then audit found that 554 relevance-map IDs include degraded IDs absent from both released pools; the official evaluator intersects relevance with each pool, leaving 196 present IDs. v03 corrects only this declaration and adds an enforcement check. Slice seed, membership algorithm, methods, and metrics are unchanged; the v02 result remains preserved as superseded evidence.

## Frozen questions

- Quality: The 24-skill MRR/Recall ceiling will weaken as the candidate pool grows; measure each method's ranking degradation rather than claiming parity from the local ceiling.
- Cost: BM25, the official 0.6B encoder, and strict top-20 reranking have materially different preparation, indexing, memory, and online-query scaling curves.

The local 24-skill result is context only and is not plotted as an official-dataset scale point. Official subsets contain every graded relevance ID and use a frozen relevance-blind hash order for distractors and score ties.

## easy tier

Pool records: 78,361; graded IDs forced into every slice: 196.

| Method | Pool | MRR@10 | Recall@10 | NDCG@10 | FullCoverage@10 | Index ms | Query mean ms | Query P95 ms | Traced peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_brief | 1,000 | 0.7143 | 0.6442 | 0.5799 | 0.4533 | 104.20 | 2.942 | 3.880 | 2.43 |
| bm25_brief | 10,000 | 0.5945 | 0.5776 | 0.4901 | 0.3733 | 878.60 | 28.596 | 37.435 | 10.67 |
| bm25_brief | 78,361 | 0.5142 | 0.4460 | 0.3834 | 0.2667 | 7,170.10 | 299.596 | 391.053 | 40.75 |
| bm25_all_fields | 1,000 | 0.8298 | 0.7988 | 0.7369 | 0.6667 | 1,295.67 | 14.292 | 20.392 | 18.55 |
| bm25_all_fields | 10,000 | 0.7500 | 0.7076 | 0.6433 | 0.5467 | 13,747.89 | 146.482 | 205.511 | 99.54 |
| bm25_all_fields | 78,361 | 0.6523 | 0.6085 | 0.5350 | 0.4667 | 106,877.53 | 1,359.744 | 1,946.307 | 456.51 |

- `skillrouter_encoder`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

- `skillrouter_pipeline`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

## hard tier

Pool records: 79,141; graded IDs forced into every slice: 196.

| Method | Pool | MRR@10 | Recall@10 | NDCG@10 | FullCoverage@10 | Index ms | Query mean ms | Query P95 ms | Traced peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_brief | 1,000 | 0.7195 | 0.6612 | 0.5879 | 0.4667 | 115.85 | 2.817 | 3.735 | 2.28 |
| bm25_brief | 10,000 | 0.6231 | 0.5743 | 0.4935 | 0.3600 | 930.53 | 29.422 | 37.334 | 10.64 |
| bm25_brief | 79,141 | 0.4942 | 0.4460 | 0.3746 | 0.2667 | 8,062.66 | 289.438 | 387.000 | 40.89 |
| bm25_all_fields | 1,000 | 0.8198 | 0.7944 | 0.7412 | 0.6533 | 1,289.62 | 16.129 | 23.242 | 14.64 |
| bm25_all_fields | 10,000 | 0.7616 | 0.6867 | 0.6411 | 0.5200 | 13,932.89 | 166.603 | 243.143 | 96.62 |
| bm25_all_fields | 79,141 | 0.6368 | 0.6085 | 0.5191 | 0.4667 | 116,208.38 | 1,430.610 | 2,070.776 | 457.81 |

- `skillrouter_encoder`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

- `skillrouter_pipeline`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

## BM25 findings

- `easy`: from 1k to 78,361, brief MRR@10 drops 0.7143 → 0.5142; all-field drops 0.8298 → 0.6523. The small-pool ceiling does not persist.
- `easy` full pool: all-field improves MRR@10 from 0.5142 to 0.6523 and Recall@10 from 0.4460 to 0.6085, but costs 4.8× mean query latency, 17.2× cumulative index time, and 11.2× traced peak index memory.
- `hard`: from 1k to 79,141, brief MRR@10 drops 0.7195 → 0.4942; all-field drops 0.8198 → 0.6368. The small-pool ceiling does not persist.
- `hard` full pool: all-field improves MRR@10 from 0.4942 to 0.6368 and Recall@10 from 0.4460 to 0.6085, but costs 4.9× mean query latency, 15.2× cumulative index time, and 11.2× traced peak index memory.

The evidence supports body-aware retrieval under large distractor pools, but it also shows that richer lexical indexing is not free. Official SkillRouter neural curves remain unmeasured until a CUDA environment can run the unchanged frozen protocol.

## Interpretation boundary

- Quality comparisons across pool sizes are valid within this official dataset protocol; they must not be merged numerically with the local 24-skill benchmark.
- BM25 traced memory covers Python allocations made while building the index, not total process RSS or compressed dataset storage.
- A missing official neural result is reported as missing. CPU projections and Hash embeddings are not substituted for official 0.6B runs.
- The 1k/10k slices include all graded items by construction; they measure distractor-scale sensitivity, not random task coverage loss.
