"""Bounded, private execution adapter for real LIS comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Sequence

from .execution import BoundedExecutor, ExecutionResult
from .model_profile import ModelExecutionProfile, ResolvedModel
from .pass1_inputs import CanonicalRunReport
from .pass3_inputs import CanonicalLayerTrace
from .product_contract import (
    MAX_IN_MEMORY_ARTIFACT_BYTES,
    canonical_json_bytes,
)
from .provenance import (
    BuildProvenance,
    ProvenanceUnavailableError,
    hash_regular_file,
    load_build_provenance,
)


MAX_RUN_REPORT_BYTES = 4 * 1024 * 1024
MAX_LAYER_TRACE_BYTES = 16 * 1024 * 1024


class RealExecutionError(RuntimeError):
    """A real binary or one of its source-bound artifacts failed closed."""

    def __init__(
        self,
        message: str,
        *,
        classification: str = "harness_error",
        execution: ExecutionResult | None = None,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.execution = execution


@dataclass(frozen=True)
class ResolvedBinary:
    path: Path
    provenance: BuildProvenance


@dataclass(frozen=True)
class RunCapture:
    report: CanonicalRunReport
    layer_trace: CanonicalLayerTrace | None
    execution: ExecutionResult


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _no_symlink_regular(path: Path, *, executable: bool) -> Path:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise RealExecutionError(
                "binary path is missing or inaccessible",
                classification="unsupported",
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RealExecutionError(
                "symlink binary path components are prohibited",
                classification="unsupported",
            )
    info = absolute.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RealExecutionError(
            "binary input is not a regular file",
            classification="unsupported",
        )
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RealExecutionError(
            "binary input is not owned by the current user",
            classification="unsupported",
        )
    if executable and not os.access(absolute, os.X_OK):
        raise RealExecutionError(
            "binary input is not executable",
            classification="unsupported",
        )
    return absolute


def resolve_binary(path: Path) -> ResolvedBinary:
    binary = _no_symlink_regular(Path(path), executable=True)
    try:
        provenance = load_build_provenance(binary)
    except ProvenanceUnavailableError as exc:
        raise RealExecutionError(
            "binary-adjacent build provenance is unavailable",
            classification="inconclusive",
        ) from exc
    except (OSError, ValueError) as exc:
        raise RealExecutionError(
            "binary-adjacent build provenance is invalid",
            classification="harness_error",
        ) from exc
    return ResolvedBinary(binary, provenance)


def resolve_backend_binary() -> ResolvedBinary:
    checkout_candidates = (
        Path.cwd() / "srcs" / "libs" / "lis",
        Path(__file__).resolve().parents[2] / "srcs" / "libs" / "lis",
    )
    seen: set[Path] = set()
    for checkout in checkout_candidates:
        absolute = checkout.absolute()
        if absolute in seen:
            continue
        seen.add(absolute)
        if checkout.is_file() and not checkout.is_symlink():
            return resolve_binary(checkout)
    discovered = shutil.which("lis", path=os.environ.get("PATH", os.defpath))
    if discovered is None:
        raise RealExecutionError(
            "no eligible LIS binary was found",
            classification="unsupported",
        )
    return resolve_binary(Path(discovered))


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RealExecutionError("cannot create private runtime input") from exc
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RealExecutionError("private runtime input write stalled")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_runtime_output(path: Path, maximum: int) -> bytes:
    if maximum <= 0 or maximum > MAX_IN_MEMORY_ARTIFACT_BYTES:
        raise ValueError("runtime artifact bound is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RealExecutionError("required runtime artifact is missing") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RealExecutionError("runtime artifact is not a regular file")
        if hasattr(os, "getuid") and before.st_uid != os.getuid():
            raise RealExecutionError("runtime artifact owner is invalid")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(65_536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RealExecutionError("runtime artifact changed while reading")
        if len(data) > maximum:
            raise RealExecutionError("runtime artifact exceeds its byte bound")
        return bytes(data)
    finally:
        os.close(fd)


def _fingerprint(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        algorithm = value.get("algorithm")
        payload = value.get("hex")
        if isinstance(algorithm, str) and isinstance(payload, str):
            return f"{algorithm}:{payload}"
    return None


def _validate_report(
    report: CanonicalRunReport,
    *,
    binary_fingerprint: str | None,
    model: ResolvedModel,
    profile: ModelExecutionProfile,
    expected_backend: str | None,
) -> None:
    raw = report.materialize()
    if raw.get("schema") != "lis.execution_artifact/v1" or raw.get("kind") != "run_report":
        raise RealExecutionError("LIS emitted an unsupported run report")
    manifest = raw.get("manifest")
    body = raw.get("report")
    if not isinstance(manifest, dict) or not isinstance(body, dict):
        raise RealExecutionError("LIS run report is structurally incomplete")
    if body.get("execution_status") != "ok":
        raise RealExecutionError("LIS run report did not record a successful execution")
    runtime = manifest.get("runtime")
    model_manifest = manifest.get("model")
    backend = manifest.get("backend")
    if not all(isinstance(item, dict) for item in (runtime, model_manifest, backend)):
        raise RealExecutionError("LIS run report manifest is incomplete")
    if (
        runtime.get("configured_context") != profile.context_length
        or runtime.get("batch_size") != profile.batch_size
        or runtime.get("thread_count") != profile.thread_count
        or runtime.get("generation_limit") != profile.generation_limit
        or model_manifest.get("family") != "llama3_decoder"
    ):
        raise RealExecutionError("LIS run report disagrees with the frozen profile")
    if expected_backend is not None and backend.get("name") != expected_backend:
        raise RealExecutionError(
            "LIS selected an unexpected backend",
            classification="unsupported",
        )
    if binary_fingerprint is not None:
        actual = _fingerprint((manifest.get("binary") or {}).get("fingerprint"))
        if actual != binary_fingerprint:
            raise RealExecutionError("LIS binary identity drifted across executions")
    prompt_sequences = body.get("prompt_sequences")
    if (
        not isinstance(prompt_sequences, list)
        or len(prompt_sequences) != profile.batch_size
        or any(
            not isinstance(item, dict)
            or item.get("token_count") != len(profile.direct_token_ids)
            for item in prompt_sequences
        )
    ):
        raise RealExecutionError("LIS prompt boundary disagrees with the frozen profile")


class RealLISExecutor:
    def __init__(
        self,
        *,
        runtime_dir: Path,
        model: ResolvedModel,
        profile: ModelExecutionProfile,
        timeout_seconds: int,
        executor: BoundedExecutor | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.model = model
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.executor = BoundedExecutor() if executor is None else executor
        self._sequence = 0
        self._seen_artifact_sets: set[str] = set()
        self.token_path = self.runtime_dir / "direct_tokens.txt"
        token_text = " ".join(str(token) for token in profile.direct_token_ids) + "\n"
        _write_private(self.token_path, token_text.encode("ascii"))
        self._token_sha256 = hashlib.sha256(token_text.encode("ascii")).hexdigest()

    @property
    def input_sha256(self) -> str:
        return _digest(
            {
                "domain": "lis.verify.direct_token_input/v1",
                "profile_id": self.profile.profile_id,
                "token_ids": list(self.profile.direct_token_ids),
            }
        )

    def _validate_frozen_inputs(self, binary: ResolvedBinary) -> None:
        try:
            observed = load_build_provenance(binary.path)
            model_sha256, _ = hash_regular_file(self.model.model_path)
            config_sha256, _ = hash_regular_file(self.model.config_path)
        except (OSError, ValueError) as exc:
            raise RealExecutionError("frozen execution input is unavailable") from exc
        if (
            observed.identity_sha256 != binary.provenance.identity_sha256
            or observed.binary_sha256 != binary.provenance.binary_sha256
            or observed.source_sha256 != binary.provenance.source_sha256
            or model_sha256 != self.model.model_sha256
            or config_sha256 != self.model.config_sha256
        ):
            raise RealExecutionError("frozen execution input identity changed")
        token_data = _read_runtime_output(self.token_path, 4096)
        if hashlib.sha256(token_data).hexdigest() != self._token_sha256:
            raise RealExecutionError("private direct-token input identity changed")

    def run(
        self,
        binary: ResolvedBinary,
        *,
        role: str,
        environment: Mapping[str, str] | None = None,
        expected_backend: str | None = None,
        expected_binary_fingerprint: str | None = None,
        forced_prefix: Sequence[int] | None = None,
        forced_binding: bytes | None = None,
        checkpoint_step: int | None = None,
        intra_layer: int | None = None,
    ) -> RunCapture:
        self._validate_frozen_inputs(binary)
        self._sequence += 1
        stem = f"{self._sequence:02d}-{role}"
        report_path = self.runtime_dir / f"{stem}-report.json"
        layer_path = self.runtime_dir / f"{stem}-layer.json"
        argv = [
            os.fspath(binary.path),
            "--model",
            os.fspath(self.model.directory),
            "--config",
            os.fspath(self.model.config_path),
            "--tokens",
            os.fspath(self.token_path),
            "--context",
            str(self.profile.context_length),
            "--batch",
            str(self.profile.batch_size),
            "--generate",
            str(self.profile.generation_limit),
            "--threads",
            str(self.profile.thread_count),
            "--report-json",
            os.fspath(report_path),
        ]
        if checkpoint_step is not None:
            argv.extend(
                [
                    "--layer-checkpoints",
                    str(checkpoint_step),
                    "--layer-trace-json",
                    os.fspath(layer_path),
                ]
            )
        if intra_layer is not None:
            if checkpoint_step is None:
                raise ValueError("intra-layer capture requires a checkpoint step")
            argv.extend(["--intra-layer-checkpoints", str(intra_layer)])
        if forced_prefix is not None or forced_binding is not None:
            if forced_prefix is None or forced_binding is None:
                raise ValueError("forced prefix and binding must be paired")
            binding_path = self.runtime_dir / f"{stem}-forced-binding.json"
            _write_private(binding_path, forced_binding)
            argv.extend(
                [
                    "--forced-prefix",
                    " ".join(str(token) for token in forced_prefix),
                    "--forced-prefix-binding-json",
                    os.fspath(binding_path),
                ]
            )
        child_env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        if environment is not None:
            child_env.update(environment)
        result = self.executor.run(
            argv,
            cwd=self.runtime_dir,
            environment=child_env,
            timeout_seconds=self.timeout_seconds,
        )
        if not result.ok:
            classification = (
                "inconclusive"
                if result.status in {"timeout", "interrupted"}
                else "harness_error"
            )
            raise RealExecutionError(
                "LIS execution did not complete successfully",
                classification=classification,
                execution=result,
            )
        self._validate_frozen_inputs(binary)
        try:
            report = CanonicalRunReport.from_json(
                _read_runtime_output(report_path, MAX_RUN_REPORT_BYTES).decode("utf-8")
            )
        except (UnicodeError, ValueError) as exc:
            raise RealExecutionError("LIS emitted a malformed run report") from exc
        _validate_report(
            report,
            binary_fingerprint=expected_binary_fingerprint,
            model=self.model,
            profile=self.profile,
            expected_backend=expected_backend,
        )
        artifact_set_id = report.materialize().get("artifact_set_id")
        if not isinstance(artifact_set_id, str) or not artifact_set_id:
            raise RealExecutionError("LIS artifact-set identity is unavailable")
        if artifact_set_id in self._seen_artifact_sets:
            raise RealExecutionError(
                "LIS reused an artifact set across independent executions"
            )
        self._seen_artifact_sets.add(artifact_set_id)
        layer_trace = None
        if checkpoint_step is not None:
            try:
                layer_trace = CanonicalLayerTrace.from_json(
                    _read_runtime_output(layer_path, MAX_LAYER_TRACE_BYTES).decode("utf-8")
                )
            except (UnicodeError, ValueError) as exc:
                raise RealExecutionError("LIS emitted a malformed layer trace") from exc
        return RunCapture(report, layer_trace, result)


def role_identity(
    binary: ResolvedBinary,
    model: ResolvedModel,
    profile: ModelExecutionProfile,
    input_sha256: str,
    report: CanonicalRunReport,
) -> dict[str, str]:
    raw = report.materialize()
    manifest = raw["manifest"]
    return {
        "source_sha256": binary.provenance.source_sha256,
        "binary_sha256": binary.provenance.binary_sha256,
        "model_sha256": model.model_sha256,
        "config_sha256": model.config_sha256,
        "input_sha256": input_sha256,
        "runtime_sha256": _digest(
            {
                "domain": "lis.verify.runtime_identity/v1",
                "profile_sha256": profile.identity_sha256,
                "runtime": manifest["runtime"],
            }
        ),
        "backend_sha256": _digest(
            {
                "domain": "lis.verify.backend_identity/v1",
                "backend": manifest["backend"],
            }
        ),
    }
