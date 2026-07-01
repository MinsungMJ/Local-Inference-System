# Pass 2 Prefix/Policy Reproduction Fixtures

Synthetic, minimized, model-free `run_report` fixtures for P1 Pass 2 tests.
The original reports carry only the identity and runtime metadata needed for
source binding, build continuity, exact-prefix checks, and context derivation.
Token IDs are synthetic. The fixtures contain no prompt text, generated text,
model data, paths, logits, tensors, checkpoint binaries, or numeric checkpoint
values.

The bound reference/candidate pair first differs at generated token step 17.
Reproduction fixtures use the same model/config/input identities while varying
only the field named by each fixture.
