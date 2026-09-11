# HTTP disclosure transfer — confirmation

Protocol: `http-disclosure-transfer-20260903-v01`; status `frozen`.

The existing `signal-disclosure-dev-v01` policy was used unchanged. Every task ran against an isolated loopback HTTP service and the verifier checked the exact request trace as well as the output.

| Method | Success | Selection F1 | Sequence | Tokens | P50 ms | API errors |
|---|---:|---:|---:|---:|---:|---:|
| one_stage | 0.6667 | 0.8333 | 0.8333 | 2636 | 1908.43 | 0 |
| always_full | 0.8333 | 0.8333 | 0.8333 | 5263 | 3414.99 | 0 |
| adaptive_signals | 0.6667 | 0.6667 | 0.6667 | 5330 | 3627.11 | 0 |

## Frozen-rule result

- Classification: **partial_transfer**
- Observed: `{"adaptive_api_errors": 0, "adaptive_disclosure_levels": {"brief": 2, "full": 2, "schema": 2}, "adaptive_success_count": 4, "always_full_success_count": 5}`
- Adaptive token change vs always-full: +1.27% (5330 vs 5263).

## Failure layers

- `one_stage`: `{"selection": 1, "verification_or_behavior": 1}`
- `always_full`: `{"selection": 1}`
- `adaptive_signals`: `{"argument_contract": 1, "selection": 1}`

## Adaptive per-task evidence

| Task | Planned IDs | Level | Markers | Success | Repairs | Tokens | Latency ms | Failure |
|---|---|---|---|---:|---:|---:|---:|---|
| thttpc01 | shttp01 | brief | - | true | 0 | 661 | 3398.32 | - |
| thttpc02 | shttp03 | brief | - | true | 0 | 678 | 3199.94 | - |
| thttpc03 | shttp04,shttp05 | schema | - | true | 0 | 948 | 3975.13 | - |
| thttpc04 | shttp06 | full | 仅当,原样,不自动 | true | 0 | 1040 | 3627.11 | - |
| thttpc05 | shttp08 | schema | - | false | 1 | 1224 | 5114.13 | schema_validation_failed |
| thttpc06 | shttp03 | full | 原样,不自动 | false | 0 | 779 | 24855.24 | http_state_mismatch |

## Interpretation

- The frozen cross-domain threshold required at least 5/6 adaptive successes. The observed 4/6 therefore cannot be called full cross-domain transfer; it meets only the preregistered partial-transfer rule.
- `thttpc04` shows positive behavioral transfer: the old markers triggered full disclosure, the planner supplied strict accepted-status/error-body behavior, and the verifier observed exactly one POST.
- `thttpc05` selected the correct PATCH Skill but serialized an object as a string; hidden-Schema escalation and the single repair did not recover. This is an argument-contract failure after correct selection.
- `thttpc06` selected HEAD for a task requiring an error body. Full disclosure cannot repair a wrong Skill selected in stage one, so this is a selection-layer failure.
- Adaptive did not save tokens on confirmation because the failed PATCH case added a repair call. Cost savings from dev do not transfer reliably with six tasks.

## Validity boundary

- Six tasks per split are an engineering transfer check, not a broad statistical generalization claim.
- Loopback HTTP preserves methods, headers, status codes, bodies, and request counts but excludes public-network nondeterminism.
- Confirmation results may not change the policy, task family, markers, or thresholds.
