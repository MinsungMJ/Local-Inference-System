# Pass 1 Token-Localization Fixtures

Synthetic, minimized, model-free `run_report` fixtures for P1 Pass 1 tests.
They use the current nested manifest and object-valued fingerprint shape. Token
IDs are synthetic; the fixtures contain no prompt text, generated text, model
data, paths, logits, tensors, or checkpoint values.

`run_prefix_64.json` and `run_prefix_65.json` provide long exact reference
sequences. Tests derive a candidate by changing the final selected token, so
the matched prefix lengths are 64 and 65 respectively.
