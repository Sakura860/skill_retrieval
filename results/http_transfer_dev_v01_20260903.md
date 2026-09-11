# HTTP disclosure transfer — dev

Protocol: `http-disclosure-transfer-20260903-v01`; status `development`.

The existing `signal-disclosure-dev-v01` policy was used unchanged. Every task ran against an isolated loopback HTTP service and the verifier checked the exact request trace as well as the output.

| Method | Success | Selection F1 | Sequence | Tokens | P50 ms | API errors |
|---|---:|---:|---:|---:|---:|---:|
| one_stage | 0.6667 | 0.8333 | 0.8333 | 2588 | 1820.01 | 0 |
| always_full | 0.8333 | 0.8333 | 0.8333 | 5212 | 3106.16 | 0 |
| adaptive_signals | 0.8333 | 0.8333 | 0.8333 | 4737 | 2820.50 | 0 |

## Frozen-rule result

- Classification: **cross_domain_transfer**
- Observed: `{"adaptive_api_errors": 0, "adaptive_disclosure_levels": {"brief": 3, "full": 2, "schema": 1}, "adaptive_success_count": 5, "always_full_success_count": 5}`
- Adaptive token change vs always-full: -9.11% (4737 vs 5212).

## Failure layers

- `one_stage`: `{"selection": 1, "verification_or_behavior": 1}`
- `always_full`: `{"selection": 1}`
- `adaptive_signals`: `{"selection": 1}`

## Adaptive per-task evidence

| Task | Planned IDs | Level | Markers | Success | Repairs | Tokens | Latency ms | Failure |
|---|---|---|---|---:|---:|---:|---:|---|
| thttpd01 | shttp01 | brief | - | true | 0 | 651 | 2820.50 | - |
| thttpd02 | shttp03 | brief | - | true | 0 | 668 | 2926.76 | - |
| thttpd03 | shttp04,shttp05 | schema | - | true | 0 | 922 | 4327.15 | - |
| thttpd04 | shttp06 | full | 仅当,原样,不自动 | true | 0 | 1019 | 3224.10 | - |
| thttpd05 | shttp07 | brief | - | true | 0 | 727 | 2642.59 | - |
| thttpd06 | shttp03 | full | 原样,不自动 | false | 0 | 750 | 2221.87 | http_state_mismatch |

## Validity boundary

- Six tasks per split are an engineering transfer check, not a broad statistical generalization claim.
- Loopback HTTP preserves methods, headers, status codes, bodies, and request counts but excludes public-network nondeterminism.
- Confirmation results may not change the policy, task family, markers, or thresholds.
