# LIS Verify Model-Free Demo

- Milestone: Pass 5 M2
- Status: Implemented Alpha boundary
- Command: `lis-verify demo`
- Expected semantic result: `REGRESSION` (exit 4)

The demo is the first immediately testable LIS Verify workflow. It requires no
model, LIS binary, private asset, network access, telemetry, or optional Textual
dependency. Its mismatch is intentional: success means the verifier detects
and localizes the seeded difference, not that the command exits zero.

## Install and run

From a clean checkout:

```bash
python -m pip install --no-deps .

set +e
lis-verify demo --out .lis/verify
demo_status=$?
set -e
test "$demo_status" -eq 4
```

The source-tree equivalent is:

```bash
PYTHONPATH=tools python -m lis_verify demo --out .lis/verify
```

Each run creates a new private `attempt-<id>` directory containing:

- `verification_report.json`, the canonical source of truth;
- `summary.md`, a deterministic report rendering; and
- `attempt.jsonl`, the append-only lifecycle ledger.

The internal replay workspace is removed after observed cleanup. Supplying
`--debug-retain` preserves that bounded directory and reports the retention in
both the report and ledger.

## What the demo exercises

The installed package contains a versioned manifest, four bounded run-report
templates, an intra-layer trace template, and a generation profile. Every
resource is byte-bound by SHA-256 before use. The demo constructs separate
discovery and authoritative generations and calls the production:

1. Pass 0 calibration preflight;
2. Pass 1 selected-token localization;
3. Pass 2 independent prefix/policy reproduction validation;
4. Pass 3A discovery localization;
5. bounded authoritative recapture and Pass 3B revalidation; and
6. Pass 4 intra-layer localization.

The frozen result is a selected-token mismatch at generated step 17, the layer
interval `(4, 8]`, and the intra-layer interval
`(rope_key_output, attention_scores]`. These are bounded synthetic replay
facts. The report explicitly warns that no customer model or binary was run.

## Evidence limits and failure behavior

Digest equality is not tensor equality. A digest mismatch is not numeric
confirmation, and neither suspect interval confirms the first numeric or
operation-level divergence. Numeric confirmation remains `not_performed`, and
all frozen nonclaims remain false.

The replay computation runs in a shell-free bounded worker. Timeout produces
`INCONCLUSIVE` (exit 3); handled SIGINT/SIGTERM preserve exits 130/143; fixture,
worker-output, or binding corruption produces `HARNESS_ERROR` (exit 2). No
failure path promotes partial evidence to the seeded `REGRESSION`, and there is
no silent retry. Real `backend` and `runtime` workflows remain unavailable
until M3.
