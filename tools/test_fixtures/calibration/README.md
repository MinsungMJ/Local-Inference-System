# Pass 0 Calibration Fixtures

Synthetic, minimized fixtures for the model-free P1 Pass 0 (Calibration
Preflight) tests in `tools/tests/`.

## Provenance

- These fixtures are **synthetic and/or minimized**; they are not captured
  production runs.
- They contain **no private data**, **no private prompt text**, and **no
  confidential generated text** (only token counts and `fnv1a64` digests).
- Model-family labels are **compatibility labels only** and do not imply model
  endorsement or quality claims.

## Files

| File | Purpose |
|---|---|
| `lis_build_profile.json` | Default LIS build calibration profile (F1/F3 facts: penalty 1.2 always-on, structural suppression on, Llama `rms_norm_eps` unbound / Qwen3 bound, KV write not verified round-to-nearest-even). |
| `run_llama_reference.json` | Minimal nested-shape `run_report` for a Llama Mode A reference side (`backend: reference`, `kv: bf16`). |
| `run_llama_avx2.json` | Matching candidate side (`backend: avx2`); identical identity, different backend. |
| `golden/calibration_preflight_llama_mode_a.json` | Golden serialized `calibration_preflight` artifact for the live-Llama Mode A scenario (`limited_comparison_allowed`). |

All run-report fixtures are schema-valid against `lis.execution_artifact/v1`.
