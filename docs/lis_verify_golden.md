# LIS Verify Golden Workflows

- Milestone: Pass 5 M4
- Status: Implemented
- Golden manifest: `lis.public_model_golden/v1`
- Expected public-model result: `PASS` under strict policy

M4 provides two repeatable gates. Pull requests run the offline, model-free
seeded demonstration and require the expected `REGRESSION`/exit 4 result. A
separate scheduled, manual, or release workflow runs a real scalar-versus-
optimized backend comparison over a pinned public model and requires
`PASS`/exit 0.

## Public model identity

The repository stores no model weights. The packaged canonical manifest names
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
whose upstream model page declares Apache-2.0, at immutable revision:

```text
28e66ca6931668447a3bac213f23d990ad3b0e2b
```

Required material is exactly:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 704 | `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| `model.safetensors` | 269,060,552 | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |

The configuration is `LlamaForCausalLM`, model type `llama`, uniform BF16,
plain RoPE, one merged safetensors file, tied embeddings, and an 8192-token
maximum context. The gate uses direct token ID `[1]`, context 128, batch one,
eight generated tokens, and one runtime thread. No tokenizer download or chat
template is involved.

The model-readiness run validated this material against LIS revision
`b9a779221ea3f3696e17b27f29e3b5c3c050215c`, version `0.2.0a1`, and production
source identity
`sha256:7d6ac0be7753ffd91839269d3ec7b755782a39cb0e227955d8db4e7d2cfa0006`.
The packaged manifest identity is
`sha256:c9c8230064d6e3b5d80409a0e1b28d04525a183d218a0a70b4a111063e69d53a`.

## Explicit local preparation

Acquisition is always a visible operator or CI step. Neither `lis-verify` nor
`make verify-diff` accesses the network.

```bash
MODEL_DIR=/path/to/frozen-smollm2-135m
mkdir -p "$MODEL_DIR"

curl --fail --location \
  --output "$MODEL_DIR/config.json" \
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/28e66ca6931668447a3bac213f23d990ad3b0e2b/config.json

curl --fail --location \
  --output "$MODEL_DIR/model.safetensors" \
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/28e66ca6931668447a3bac213f23d990ad3b0e2b/model.safetensors

PYTHONPATH=tools python -m lis_verify.golden --model "$MODEL_DIR"
```

The validator rejects mutable URLs in the manifest, unknown fields,
noncanonical JSON, unsafe paths, symlinks, owner mismatch, wrong size or hash,
and configuration drift. Missing material fails clearly; it is never resolved
through a private fallback path.

## Run the strict gate

```bash
make verify-diff \
  VERIFY_MODEL="$MODEL_DIR" \
  VERIFY_DIFF_OUT_DIR=.lis/verify \
  VERIFY_STAGE_TIMEOUT_SECONDS=600
```

The reference side must resolve to the scalar `reference` backend. This v1
manifest requires the candidate to resolve to AVX2. If the host lacks that path,
the canonical semantic verdict remains `UNSUPPORTED`, while strict policy exits
6. Identical reference and candidate backend identities can never pass.

`PASS` means the two paths selected the same eight token IDs in this bounded
case. It is not tensor equality, whole-runtime equivalence, numeric
confirmation, or proof of all model behavior.

## CI evidence and limits

Both workflows freeze a private acceptance manifest after proving the tracked
source clean. That authority binds the source revision/tree, dependency input,
and workflow/Make command files. Each LIS Verify report is marked
`verification_acceptance`; a debug-classified report cannot satisfy the CI
consumer.

The consumer validates the canonical report, its deterministic Markdown
projection, the append-only ledger, verdict/exit policy, cleanup, and artifact
identities. The golden path additionally checks exact model/config/input
identities and distinct backend identities. It writes a bounded Actions step
summary and uploads the report, summary, ledger, and acceptance manifest.

The PR job is limited to 15 minutes. The public-model job is limited to 30
minutes, requires at least 2 GiB free disk before acquisition, bounds each
download to its exact manifest size and time window, uses one inference thread,
sets a 600-second stage timeout, and retains uploaded evidence for 30 days.
Model acquisition has no silent retry.

## Baseline updates

A baseline change is never automatic. A reviewed change must update the
canonical manifest and the immutable URLs in the public-model workflow, then
re-run material validation, focused adversarial tests, the full test suite, and
a clean one-shot golden acceptance. The diff must explain the license,
revision, file hashes and sizes, configuration boundary, expected semantic
result, and newly validated LIS identity. Updating only an expected verdict or
weakening the optimized-backend condition is prohibited.
