# SkillRouter large-pool scaling — frozen protocol report

Protocol: `skillrouter-large-pool-20260903-v02`; status `frozen`. This is retrieval-only: no LLM Planner or Agent calls were issued.

## Frozen questions

- Quality: The 24-skill MRR/Recall ceiling will weaken as the candidate pool grows; measure each method's ranking degradation rather than claiming parity from the local ceiling.
- Cost: BM25, the official 0.6B encoder, and strict top-20 reranking have materially different preparation, indexing, memory, and online-query scaling curves.

The local 24-skill result is context only and is not plotted as an official-dataset scale point. Official subsets contain every graded relevance ID and use a frozen relevance-blind hash order for distractors and score ties.

## easy tier

Pool records: 78,361; graded IDs forced into every slice: 196.

| Method | Pool | MRR@10 | Recall@10 | NDCG@10 | FullCoverage@10 | Index ms | Query mean ms | Query P95 ms | Traced peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_brief | 1,000 | 0.7143 | 0.6442 | 0.5799 | 0.4533 | 138.88 | 2.464 | 3.358 | 2.43 |
| bm25_brief | 10,000 | 0.5945 | 0.5776 | 0.4901 | 0.3733 | 850.94 | 25.008 | 34.772 | 10.67 |
| bm25_brief | 78,361 | 0.5142 | 0.4460 | 0.3834 | 0.2667 | 6,830.34 | 259.401 | 356.760 | 40.75 |
| bm25_all_fields | 1,000 | 0.8298 | 0.7988 | 0.7369 | 0.6667 | 1,146.73 | 12.293 | 17.828 | 18.55 |
| bm25_all_fields | 10,000 | 0.7500 | 0.7076 | 0.6433 | 0.5467 | 12,148.04 | 133.671 | 185.668 | 99.54 |
| bm25_all_fields | 78,361 | 0.6523 | 0.6085 | 0.5350 | 0.4667 | 117,628.73 | 1,257.197 | 1,911.797 | 456.51 |

- `skillrouter_encoder`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

- `skillrouter_pipeline`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

## hard tier

Pool records: 79,141; graded IDs forced into every slice: 196.

| Method | Pool | MRR@10 | Recall@10 | NDCG@10 | FullCoverage@10 | Index ms | Query mean ms | Query P95 ms | Traced peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_brief | 1,000 | 0.7195 | 0.6612 | 0.5879 | 0.4667 | 94.81 | 2.479 | 3.374 | 2.28 |
| bm25_brief | 10,000 | 0.6231 | 0.5743 | 0.4935 | 0.3600 | 824.21 | 24.623 | 33.474 | 10.64 |
| bm25_brief | 79,141 | 0.4942 | 0.4460 | 0.3746 | 0.2667 | 7,146.84 | 275.487 | 364.599 | 40.89 |
| bm25_all_fields | 1,000 | 0.8198 | 0.7944 | 0.7412 | 0.6533 | 1,315.66 | 15.678 | 22.592 | 14.64 |
| bm25_all_fields | 10,000 | 0.7616 | 0.6867 | 0.6411 | 0.5200 | 13,888.69 | 145.343 | 202.745 | 96.62 |
| bm25_all_fields | 79,141 | 0.6368 | 0.6085 | 0.5191 | 0.4667 | 108,866.27 | 1,359.834 | 1,894.290 | 457.81 |

- `skillrouter_encoder`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

- `skillrouter_pipeline`: **not_run_hardware_gate** — CUDA unavailable; frozen full-pool hardware gate prohibits a formal CPU or Hash substitute

## Interpretation boundary

- Quality comparisons across pool sizes are valid within this official dataset protocol; they must not be merged numerically with the local 24-skill benchmark.
- BM25 traced memory covers Python allocations made while building the index, not total process RSS or compressed dataset storage.
- A missing official neural result is reported as missing. CPU projections and Hash embeddings are not substituted for official 0.6B runs.
- The 1k/10k slices include all graded items by construction; they measure distractor-scale sensitivity, not random task coverage loss.
