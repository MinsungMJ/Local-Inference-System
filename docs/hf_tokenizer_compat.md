# HuggingFace tokenizer.json Compatibility Scope

This document defines the supported subset of HuggingFace `tokenizer.json`
files for BPE tokenizer import.

## Supported Structure

LIS imports BPE vocabulary and merge data from the `model` object within a `tokenizer.json` file. It also imports root-level `added_tokens` entries marked as special tokens so Llama chat/control tokens can be encoded atomically.

### Required Fields

| JSON Path | Type | Constraint |
|---|---|---|
| `model.type` | string | Must be `"BPE"`. All other values are rejected with `LIS_STATUS_UNSUPPORTED`. |
| `model.vocab` | object | Maps token strings (UTF-8) to integer token IDs (0-based). |
| `model.merges` | array | Each entry is either a string `"token_a token_b"` (legacy space-delimited pair) or an array of two strings `["token_a", "token_b"]` (modern array pair). Both formats resolve token strings via the vocab and populate the merge table by array order (index 0 = highest priority). |

### Optional Fields

| JSON Path | Type | Handling |
|---|---|---|
| `added_tokens` | array of objects | Entries with integer `id`, string `content`, and no `special:false` marker are inserted into the vocabulary and matched as whole tokens before byte-level BPE. |

### Rejected Tokenizer Types

If `model.type` is any value other than `"BPE"` — including `"Unigram"`, `"WordPiece"`, `"WordLevel"`, or an unrecognized string — the import function returns `LIS_STATUS_UNSUPPORTED` without populating any tokenizer state.

## Token String Encoding: Byte-to-Unicode Mapping

HuggingFace byte-level BPE tokenizers (as used by GPT-2, Llama 3.x, and related models) represent raw byte values as Unicode characters using a fixed bijective mapping originally defined in GPT-2.

### Mapping Definition

The 256 byte values are partitioned into two sets:

**Direct-mapped bytes (188 values):** Bytes whose values are also their Unicode codepoints.

- 33–126 (printable ASCII: `!` through `~`)
- 161–172 (`¡` through `¬`)
- 174–255 (`®` through `ÿ`)

**Indirect-mapped bytes (68 values):** Bytes that map to codepoints 256–323.

- Bytes 0–32 → codepoints 256–288 (Ā through Ġ)
- Bytes 127–160 → codepoints 289–322
- Byte 173 → codepoint 323

### Reversal During Import

When LIS imports a vocabulary entry from `model.vocab`, each Unicode codepoint in the token string is reversed to its original byte value:

1. If the codepoint is in the direct-mapped set, the byte value equals the codepoint.
2. If the codepoint is in range 256–323, it maps to the corresponding indirect byte.
3. If the codepoint is outside both sets (above 323, or in gaps like 0–32 as raw codepoints), the token is rejected as invalid.

The resulting raw byte string is stored in `lis_tokenizer.token_bytes` for the token's ID.

### Merge Entry Parsing

Each merge rule may appear as either a space-delimited string or an array of two strings. Both are parsed so the two token strings can be looked up in the vocab and concatenated to find the resulting merge token. The import path supports the classic `"token_a token_b"` format and the modern `["token_a", "token_b"]` array format used by some newer artifacts. To insert a merge:

1. Parse the two token strings from the merge entry.
2. Look up the vocab ID for `token_a`.
3. Look up the vocab ID for `token_b`.
4. Look up the vocab ID for the concatenation `token_a + token_b`.
5. Insert `(first_id, second_id, result_id, rank)` into the merge table.

If any of these lookups fail, the merge is rejected.

## Limitations

The following HuggingFace tokenizer.json features are **not supported** by this import path:

| Feature | Handling |
|---|---|
| `pre_tokenizer` (regex splitting, whitespace rules) | Ignored. LIS applies its own byte-level BPE directly to raw input bytes. |
| `normalizer` (NFKC, lowercase, strip accents) | Ignored. Input text is not normalized before encoding. |
| `post_processor` (template processing, special token insertion) | Ignored. |
| `decoder` (byte-level decoder configuration) | Ignored. LIS uses its own vocabulary-based decode path. |
| `model.byte_fallback` | Ignored. LIS relies on the byte_to_token lookup built from single-byte vocabulary entries. |
| `model.dropout`, `model.unk_token`, `model.continuing_subword_prefix`, `model.end_of_word_suffix`, `model.fuse_unk` | Ignored. These do not affect vocabulary or merge table extraction. |

### Behavioral Differences

Because LIS does not execute the pre-tokenizer, its BPE encoding may produce different tokenizations than HuggingFace's full tokenizer pipeline for the same input text. The vocabulary and merge table are correct, but the token boundary decisions may differ for inputs where the pre-tokenizer would have split text before BPE (e.g., whitespace-delimited words in GPT-2/Llama).

Root-level special `added_tokens` are matched with a longest-match pass before
normal byte-level BPE. LIS still does not execute full HuggingFace
pre-tokenizer or post-processor templates. In the supported Llama Instruct CLI
path, `--hf-tokenizer PATH --prompt TEXT` treats `TEXT` as the single user
message and wraps it with exactly
`<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{TEXT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
before tokenizer encoding. This is a narrow Llama Instruct prompt-construction
helper, not a general tokenizer post-processor or chat-template framework.

During tokenizer-backed CLI generation, known structural Llama chat/control tokens are not emitted as visible assistant text. `<|eot_id|>`, `<|end_of_text|>`, and `<|eom_id|>` are treated as stop controls; `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|finetune_right_pad_id|>`, `<|step_id|>`, `<|python_tag|>`, and `<|reserved_special_token_*|>` are excluded from greedy candidate selection. This is a narrow output hygiene policy, not a sampling framework or tokenizer post-processor implementation.

When `--diagnostics` is provided, the CLI writes one minimal generation
diagnostic record per selected step to stderr. The record includes selected
token ID, selected token text or `<unavailable>`, stop reason, whether
structural suppression affected candidate selection, whether the repetition
penalty changed selection, and whether the selected token was already
generated. Diagnostics are opt-in and do not alter default stdout generation
output.

## Error Behavior

| Condition | Status |
|---|---|
| File cannot be read | `LIS_STATUS_IO` |
| JSON is malformed | `LIS_STATUS_FORMAT` |
| `model` field is missing or not an object | `LIS_STATUS_FORMAT` |
| `model.type` is missing or not `"BPE"` | `LIS_STATUS_UNSUPPORTED` |
| `model.vocab` is missing or not an object | `LIS_STATUS_FORMAT` |
| `model.merges` is missing or not an array | `LIS_STATUS_FORMAT` |
| `added_tokens` is present but malformed | `LIS_STATUS_FORMAT` |
| Vocab entry value is not a non-negative integer | `LIS_STATUS_FORMAT` |
| Token string contains invalid codepoints | `LIS_STATUS_FORMAT` |
| Merge entry is malformed (legacy string missing space separator, or array entry not exactly two strings) | `LIS_STATUS_FORMAT` |
| Merge token not found in vocab | `LIS_STATUS_FORMAT` |
| Memory allocation failure | `LIS_STATUS_NO_MEMORY` |
