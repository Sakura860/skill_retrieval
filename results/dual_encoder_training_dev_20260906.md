# Dual-encoder dev engineering diagnostic

- Protocol: `dual-encoder-dev-engineering-20260906-v01`
- Protocol SHA-256: `4bcbe86b46b0edf89ad82d891eb60ad1b37ac4c53ad24966840bee420f8e7f36`
- Data: benchmark_v01 dev only (18 tasks, 20 positive samples)
- Claim boundary: training-set engineering diagnostic; no test/confirmation task was selected or evaluated.

## Result

- Loss: 4.003416 -> 0.078032
- MRR: 0.126455 -> 1.000000
- Recall@1: 0.000000 -> 0.944444
- Recall@5: 0.305556 -> 1.000000
- Checkpoint SHA-256: `303cc209f3db9ca76493170f3d6a2d9b7a73486ac618688891514e59ebfc6cf2`
- Checkpoint reload max score delta: 0.000000000000
- Two-run state hash exact: True
- Two-run loss history exact: True
- Overall acceptance: **True**

## Interpretation boundary

This run verifies optimization, deterministic repetition, stale-index invalidation, and portable weight reload. Because the same dev tasks supply training and evaluation, the metric gain is not evidence of held-out quality or generalization.
