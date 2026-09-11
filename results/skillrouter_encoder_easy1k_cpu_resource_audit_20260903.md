# SkillRouter official encoder easy-1k CPU resource audit

This was a deliberately limited smoke of the official SkillRouter 0.6B encoder on the frozen `easy-1k` membership from protocol `skillrouter-large-pool-20260903-v03`. It is **not** a completed retrieval-quality result and is **not** a substitute for the CUDA full-pool run.

## What happened

- The official model weights loaded successfully, so model resolution and the upstream adapter path were reached.
- The run started at approximately `2026-09-03T10:33:15+08:00` and was terminated at approximately `10:50:50`, after about 17 minutes without completing metrics.
- The last resource observation showed 7,307.58 aggregate CPU seconds, a 19.36 GB working set, and 55.27 GB private memory while the process remained responsive.
- The host has about 29.7 GB physical RAM, so the attempt was already paging heavily. Continuing would test swap tolerance rather than a credible online retrieval setup.
- No success JSON was written. The attempt is preserved only as this resource audit.

## Interpretation and boundary

The smoke verifies that the official encoder can be loaded through the adapter, but it does not establish encoder quality or latency. It strengthens the protocol's predeclared hardware gate: formal 78k/79k encoder and strict top-20 pipeline measurements require CUDA. The completed BM25 scale curves remain valid and separate.

Do not extrapolate full-pool cost from this attempt, do not rerun it with altered batch or membership under the same protocol ID, and do not substitute a hash encoder. A future formal run must preserve protocol v03 and record GPU model, VRAM, software versions, and peak memory.
