import unittest
import os
import tempfile
import json
from lis_inspect.parse_stderr import parse_stderr_text
from lis_inspect.parse_json import load_artifact, ArtifactParseError

class InspectCompatibilitySmokeTest(unittest.TestCase):
    def test_inspect_stderr_ignores_additive_lines(self):
        text = """lis: simd backend=avx2
lis: precision path=f32_accum weights=f32 kv=f32
lis: perf-stage tag=none name=model_load ns=123 ms=0.123 tokens=0
lis: perf-summary tag=none threads=1 prompt_tokens=3 generated_tokens=2 ttft_ms=1.0 itl_ms=2.0 tps_steady=10.0 tps_end_to_end=8.0
lis: perf-per-token tag=none step=0 ns=10 ms=0.01
lis: generation-diagnostic step=0 phase=decode ...
lis: generation-diagnostic-reasoning step=0 phase=decode decision_class=greedy ...
lis: generation-diagnostic-candidate step=0 phase=decode rank=1 ...
lis: layer-checkpoint step=0 phase=prefill name=embedding shape=[1] min=0 max=0 mean=0 l2=0 nan=0 inf=0"""
        log = parse_stderr_text(text)
        self.assertIsNotNone(log.summary)
        self.assertIn("model_load", log.stages)
        self.assertEqual(len(log.per_token), 1)
        self.assertEqual(len(log.warnings), 0)

    def test_inspect_rejects_decode_trace_kind(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('{"schema":"lis.execution_artifact/v1","kind":"decode_trace"}')
            path = f.name
        
        try:
            with self.assertRaises(ArtifactParseError) as context:
                load_artifact(path)
            self.assertIn("unsupported kind", str(context.exception))
            self.assertIn("decode_trace", str(context.exception))
        finally:
            os.unlink(path)

    def test_inspect_rejects_layer_trace_kind(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('{"schema":"lis.execution_artifact/v1","kind":"layer_trace"}')
            path = f.name
        
        try:
            with self.assertRaises(ArtifactParseError) as context:
                load_artifact(path)
            self.assertIn("unsupported kind", str(context.exception))
            self.assertIn("layer_trace", str(context.exception))
        finally:
            os.unlink(path)
