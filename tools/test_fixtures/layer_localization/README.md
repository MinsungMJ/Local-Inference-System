# Coverage-scoped layer-localization fixtures

This directory contains bounded metadata and digest fixtures for Pass 3. It
must not contain model weights, tensor payloads, prompts, logits arrays, model
paths, or checkpoint binaries.

`producer_contract/` freezes the producer-vNext schema and digest byte stream
before either the C producer or the model-free Pass 3 implementation. The
producer-vNext example is synthetic contract evidence, not a claim of C
runtime provenance. The legacy example remains a valid execution artifact but
is intentionally unsupported as a Pass 3 checkpoint layout.

`golden/` contains bounded `layer_localization` output artifacts emitted by
the model-free Pass 3 core. They retain canonical source identities and
coverage/interval evidence but never tensor payloads or confirmation claims.
