# LIS Inspect

Post-execution TUI inspector for LIS runs. Reads the canonical
`--report-json` artifact (`lis.execution_artifact/v1`, kind `run_report`) and,
optionally, captured stderr from a `--perf` run, then renders a small Textual
TUI with `Overview`, `Perf`, `Per-Token`, `Artifact`, `Raw`, and `Issues`
tabs.
With exactly two `--report-json` inputs it launches a two-run compare view.

JSON is canonical. stderr is supplementary: when both speak to the same value,
JSON wins.

This is **not** a live monitor, telemetry backend, prompt/output viewer, or
benchmark database.

## Status

**Core Capabilities**

- CLI argument parsing
- JSON artifact load + schema/kind validation
- stderr parser for `lis: perf-stage`, `lis: perf-summary`, `lis: perf-per-token`
- merged `InspectSession` model
- centralized semantic palette
- Overview tab
- Perf tab (compact bar + numeric table)
- derived metrics: `dominant_stage`, `run_profile`, `artifact_completeness`,
  `selected_equals_emitted`

**Report and Artifact Views**

- Per-Token tab — count / min / max / mean / p50 / p90 / p95 / p99,
  compact per-step trend, top-5 slow steps, explicit degraded message when
  no per-token data was captured
- Artifact tab — Identity, Retention Policy, Token Accounting (with
  `selected == emitted`), bounded token preview (head 8 + tail 8 IDs, or
  count+digest only when the JSON omitted the ID list), Fingerprints,
  canonical-JSON note
- Current derivations: percentile stats, top-N slow steps, bounded token
  preview — all stdlib, deterministic, unit-tested

**Raw Data and Parser Feedback**

- Raw tab — source availability, parser status, completeness classification,
  compact warning summary, and bounded JSON/stderr previews
- Lightweight parser-warning surface — Overview shows a warning badge and
  compact notes when parsing is missing, partial, malformed, or degraded
- Reload/path polish — reload refreshes all tabs including Raw, and missing or
  unreadable stderr produces bounded UI feedback instead of silent absence

**Compare Mode and UI Refinement**

- Two-run compare mode — exactly two reports, optional stderr per side,
  compact overview/perf/per-token/artifact deltas
- Lightweight interpretation — rule-based run profile, warmup effect,
  tail spread, completeness, and compact compare summary
- Issues tab — structured source availability, parser completeness, issue
  count, unavailable data, and normalized issue bullets
- UI refinement — clearer badges, semantic `focus`/`partial` palette roles,
  compare/issue visual consistency
- Header branding — `LIS Inspect` remains primary header text
- Overlay polish — command palette and theme picker now render as bounded
  centered panels with de-emphasised background, coherent input+list
  layout, and compact footer hints

**Deferred / Not Supported in the Initial Public Release**

- Live mode
- History/DB
- Prompt text viewing
- Generated text viewing
- Multi-run compare beyond exactly two runs
- Telemetry/dashboard expansion

## Install

Recommended:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/requirements.txt
```

The only dependency is `textual` (Rich is pulled in transitively).

## Invoke

The package lives at `tools/lis_inspect/`. Add `tools/` to `PYTHONPATH` and
invoke as a module:

```bash
PYTHONPATH=tools python3 -m lis_inspect \
  --report-json /tmp/lis_run.json \
  --stderr-log /tmp/lis_run.stderr
```

Two-run compare mode accepts exactly two report JSON paths. stderr logs are
optional and, when present, match the report order:

```bash
PYTHONPATH=tools python3 -m lis_inspect \
  --report-json /tmp/lis_run_a.json /tmp/lis_run_b.json \
  --stderr-log /tmp/lis_run_a.stderr /tmp/lis_run_b.stderr
```

Producing the inputs:

```bash
./srcs/libs/lis \
  --model "$MODEL" \
  --config "$MODEL/config.json" \
  --hf-tokenizer "$MODEL/tokenizer.json" \
  --prompt "Write one short sentence about the sea." \
  --context 128 --batch 1 --generate 32 --threads 4 \
  --perf --perf-per-token \
  --report-json /tmp/lis_run.json \
  2> /tmp/lis_run.stderr
```

## Headless smoke check

`--no-tui` skips Textual entirely and prints a one-line summary. Useful for CI
parser checks and for verifying inputs before launching the TUI:

```bash
PYTHONPATH=tools python3 -m lis_inspect \
  --report-json tools/test_fixtures/qwen3_1p7b_run.json \
  --stderr-log tools/test_fixtures/qwen3_1p7b_run.stderr \
  --no-tui
```

## Keys

| Key | Action          |
|-----|-----------------|
| `1` | Overview tab    |
| `2` | Perf tab        |
| `3` | Per-Token tab   |
| `4` | Artifact tab    |
| `5` | Raw tab         |
| `6` | Issues tab      |
| `r` | Reload inputs   |
| `q` | Quit            |

In compare mode, key `1` opens the Compare tab and key `6` opens Issues.

## Raw tab and warnings

`Raw` is for parser transparency: it shows whether JSON/stderr sources were
available, which parser sections were successfully parsed, the current
completeness classification (`json_only`, `json_perf`, `json_perf_per_token`,
or `degraded`), parser warnings, and bounded previews of the JSON and stderr
inputs.

Previews are deliberately limited and redacted. Raw prompt text, generated
text, and absolute paths are not surfaced. stderr remains supplementary; JSON
continues to be canonical.

Warnings are intentionally compact and non-alarmist. They flag inspection
limits such as missing stderr, missing perf sections, malformed perf lines,
partial per-token data, or unreadable secondary input. There is no dedicated
warning screen yet.

## Compare mode

Compare mode is exactly two runs and remains post-execution only. Each side
uses canonical JSON first; stderr-derived stage/per-token data is supplementary
and may be absent on either side.

The Compare tab shows:

- status, stop reason, model/backend, context, threads, and completeness
- TTFT / mean ITL / steady TPS / end-to-end TPS deltas
- stage timing deltas when both sides have stage data
- per-token count and percentile deltas when available
- retention, runtime, fingerprint, and token-accounting differences

Incomplete sides are marked partial or degraded instead of being filled in with
guesses.

## Interpretation and Issues

Interpretation is compact and rule-based. It reports only labels derived from
available metrics: run profile, warmup effect, tail spread, completeness, and
a compare summary such as `faster decode` or `no significant change`.

The Issues tab is an inspection-completeness panel, not telemetry. It explains
source availability, parser completeness, issue count, unavailable data, and
normalized issue bullets for single-run and compare mode.

## Retention and safety

LIS Inspect deliberately does not surface:

- absolute filesystem paths (when omitted by JSON retention policy)
- raw prompt text
- raw generated text

If a field is omitted by the JSON artifact's retention policy, the TUI does not
reconstruct or infer it from any other source.

## Semantic color policy

Color encodes meaning, not decoration. All colors flow through
`tools/lis_inspect/palette.py`:

| Role       | Used for                                                   |
|------------|------------------------------------------------------------|
| `success`  | OK status / healthy run                                    |
| `focus`    | Active focus / primary run emphasis                        |
| `partial`  | Partial but usable input coverage                          |
| `warning`  | Degraded / missing secondary input / parse warning         |
| `error`    | Hard failure / parse failure / incompatible artifact       |
| `identity` | Schema / kind / model family / backend                     |
| `perf`     | Dominant stage / active perf metric                        |
| `artifact` | Retention / fingerprints                                   |
| `muted`    | Secondary labels / footer notes                            |

Views never hard-code style strings; they reference roles. Update the role's
style once and every view picks it up.

## Tests

```bash
PYTHONPATH=tools python3 -m unittest discover tools/tests
```

Tests cover JSON parse / schema validation, stderr `perf-stage`,
`perf-summary`, and `perf-per-token` parsing including malformed-line warnings,
derivation of dominant stage / run profile / artifact completeness, Raw tab
bounded previews and warning rendering, compare/interpretation/issues, and the
semantic palette mapping.
Tests use stdlib `unittest`.

## Reference fixtures

- `tools/test_fixtures/qwen3_1p7b_run.{json,stderr,md}` — minimal baseline
  (no per-token, no token ID list, retention-minimized).
- `tools/test_fixtures/qwen3_1p7b_run_full.{json,stderr,md}` — fuller fixture
  adding 10 `perf-per-token` lines and explicit
  `selected_token_ids` / `emitted_token_ids` so the bounded preview and
  percentile views can be exercised end-to-end.
- `tools/test_fixtures/qwen3_1p7b_run_compare.{json,stderr}` — second run for
  two-run compare validation.

All fixtures are synthetic but schema-valid, used purely to exercise the
parser, derivation, and UI rendering paths.
