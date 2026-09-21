# LIS Verify Customer-Validation Kit

- Milestone: Pass 5 M5 local preparation only
- Status: Implemented; human observation and M5 acceptance remain pending
- Session schema: `lis.usability_session/v1`
- Protocol: `lis.usability.protocol/v1`
- Aggregate schema: `lis.usability_aggregate/v1`

This kit lets an authorized study operator validate pseudonymous, bounded
customer-observation records and compute the frozen M5 metrics locally. It does
not recruit or contact participants, record telemetry, access external
repositories, or declare M5 accepted. A threshold-satisfying aggregate can
reach only `READY_FOR_HUMAN_REVIEW`; the report permanently keeps
`m5_accepted`, `m6_authorized`, and `m7_authorized` false.

## Authority boundary

Do not begin counted sessions until a human owner has approved participant
recruitment, consent and withdrawal terms, access to the observed workflows,
the four-week window, and private data retention/deletion. Synthetic personas,
the project author, Codex, and repeated records for one participant cannot be
used to satisfy the 8–12-person cohort.

Session work and UX correction are `development_debugging`. A final metrics
run is `verification_acceptance` only when `LIS_VERIFY_ACCEPTANCE_MANIFEST`
points to a private, clean-source authority and `--source-root` is supplied.
Debugging output is never promoted into acceptance evidence.

## Private input boundary

The command accepts one directory containing zero through 64 canonical session
records. The directory must be owned by the current user and mode `0700`.
Every record must be an owned mode-`0600` regular file named:

```text
record-<32 lowercase hexadecimal characters>.json
```

The filename suffix must equal the suffix of the record's random `usr1:` ID.
Symlinks, unexpected entries, duplicate record/participant/enrollment IDs,
oversized data, duplicate JSON keys, noncanonical JSON, and files that change
while being read fail closed.

The schema accepts only fixed enums, booleans, bounded integers, canonical
SHA-256 identities, and random pseudonymous IDs. It has no free-text, name,
email, employer, repository URL, filesystem path, prompt, generated text, model
path, or tensor field. Consent records and the participant-ID map stay in a
different private location and are never inputs to this command.

The normative packaged schema is
`lis_verify.usability_contract/session_record_v1.schema.json`. The strict
standard-library validator is authoritative for relational rules such as
consent/status coherence, exact setup-time arithmetic, report identity
binding, paired investigation durations, observable retained-CI evidence, and
residue-check completeness. Public model-free examples live in
`tools/test_fixtures/lis_verify_usability/session_examples_v1.json`; they are
test material, not customer evidence.

## Standard session procedure

The facilitator uses the same sequence for every counted participant:

1. Start from the documented clean-clone boundary without a product
   walkthrough.
2. Observe installation and `lis-verify demo`.
3. Ask for the verdict meaning, evidence ceiling, and bounded next action.
4. Observe explicit preparation of the pinned public model.
5. Observe the first model-backed backend comparison.
6. Present the five standardized verdict examples, including strict
   `UNSUPPORTED`.
7. Observe normal and handled-error cleanup for sensitive tensor residue.
8. Record facilitator help and undocumented interventions; do not silently
   convert an assisted or abandoned attempt into success.

One final metrics record represents one participant. Retries and UX debugging
remain in the separate append-only session ledger. An eligible incomplete
record remains in metric denominators instead of disappearing.

## Timing and scoring

`hands_on_seconds` must exactly equal:

```text
wall_seconds
  - model_acquisition_wait_seconds
  - inference_wait_seconds
```

The aggregate emits these nine Beta metrics and one separate scope-control
metric:

| Metric | Required threshold |
|---|---:|
| clean-clone demo success | at least 90% |
| median hands-on setup | at most 600 seconds |
| manual intermediate artifact inputs | zero |
| actionable verification rate in supported environments | at least 90% |
| seeded-regression false passes | zero |
| verdict/evidence-ceiling/next-action comprehension | at least 80% |
| median paired mismatch-investigation reduction | at least 50% |
| retained CI use among the first eight eligible design partners | at least 5 |
| sensitive tensor residue events | zero |
| monthly verification-problem frequency (scope control) | at least 50% |

Every metric preserves its numerator, denominator, missing count, rational
value, exact target, and status. No floating-point rounding is used. The M5
cohort gate additionally requires 8–12 eligible participants, at least three
outside the normal LIS workflow, and 3–5 distinct pseudonymous real workflows.
Missing setup, comprehension, residue, paired timing, or four-week follow-up
evidence yields `incomplete`, never a pass.

## Local command

Use separate private input and output directories:

```bash
install -d -m 700 /private/m5-records /private/m5-output

lis-verify-usability \
  --records /private/m5-records \
  --out /private/m5-output/aggregate.json
```

The output is canonical, mode `0600`, bounded to 128 KiB, and never
overwritten. The command performs no network access and emits no telemetry.

Successful validation returns exit 0 regardless of metric status. For a final
operator gate, add `--require-beta-ready`:

| Aggregate result | Strict exit |
|---|---:|
| `READY_FOR_HUMAN_REVIEW` | 0 |
| `NOT_EVALUATED` | 3 |
| `M5_NOT_ACCEPTED` | 4 |
| malformed/unsafe input or publication failure | 2 |

The `READY_FOR_HUMAN_REVIEW` state is not M5 acceptance. It means only that the
frozen threshold arithmetic is complete and passing; an authorized human must
still review consent authority, privacy, exclusions, real-workflow evidence,
and dataset provenance.

For an acceptance-classified run, first freeze the dataset identity from a
separate debugging validation, then supply both authorities:

```bash
LIS_VERIFY_ACCEPTANCE_MANIFEST=/private/source-acceptance.json \
lis-verify-usability \
  --records /private/m5-records \
  --out /private/m5-output/final-aggregate.json \
  --source-root . \
  --expected-dataset-sha256 sha256:<64 lowercase hexadecimal characters> \
  --require-beta-ready
```

Acceptance mode fails closed when either the clean-source authority or the
pre-frozen dataset identity is absent or has changed.

## Four-week follow-up

Retained use requires observable canonical run evidence at the fixed follow-up
checkpoint. `pending`, `not_due`, stopped, unavailable, retained, and withdrawn
states remain distinct. Stated intention to keep using the workflow cannot be
substituted for an observed retained run.

## Cleanup and retention

The aggregator retains no runtime workspace and writes only the requested
aggregate report. The operator owns deletion of participant records according
to the approved retention policy. Do not commit session records, consent
material, ID maps, raw notes, external repository facts, or generated aggregate
reports to this repository.
