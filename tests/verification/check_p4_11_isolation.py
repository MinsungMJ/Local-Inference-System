#!/usr/bin/env python3
"""Fail closed if test controls leak into the production build."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


AFFECTED_OBJECTS = (
    "srcs/cli/driver.o",
    "srcs/core/checkpoint_digest.o",
    "srcs/runtime/runtime.o",
    "srcs/runtime/llama.o",
)

FORBIDDEN_PRODUCTION_TOKENS = (
    b"lis_cli_test_",
    b"lis_test_control_",
    b"test_observation_perturbation",
    b"LIS_TESTING",
    b"LIS_TEST_SELECTED_TOKEN",
    b"LIS_TEST_LAYER_OBSERVATION",
    b"LIS_TEST_INTRA_LAYER_OBSERVATION",
)

REQUIRED_TEST_SYMBOLS = (
    "lis_cli_test_injection_reset",
    "lis_cli_test_override_selected_token",
    "lis_cli_test_perturb_layer_observation",
    "lis_cli_test_perturb_intra_layer_observation",
)

FORBIDDEN_CLI_FLAGS = (
    "--test-selected-token",
    "--test-layer-observation",
    "--test-intra-layer-observation",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_required(path: Path) -> bytes:
    if not path.is_file():
        fail(f"required file is missing: {path}")
    return path.read_bytes()


def nm_text(path: Path) -> str:
    result = subprocess.run(
        ["nm", "-a", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        fail(f"nm failed for {path}: {result.stderr.strip()}")
    return result.stdout


def assert_production_clean(path: Path) -> None:
    blob = read_required(path)
    symbols = nm_text(path).encode("utf-8", errors="replace")

    for token in FORBIDDEN_PRODUCTION_TOKENS:
        if token in blob or token in symbols:
            fail(
                f"production artifact {path} contains forbidden test token "
                f"{token.decode('ascii')}"
            )


def assert_test_symbols(path: Path) -> None:
    symbols = nm_text(path)

    for symbol in REQUIRED_TEST_SYMBOLS:
        if symbol not in symbols:
            fail(f"test binary {path} is missing test-only symbol {symbol}")


def assert_cli_surface(production_binary: Path) -> None:
    help_result = subprocess.run(
        [str(production_binary), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if help_result.returncode != 0:
        fail("production --help failed during isolation verification")
    help_text = help_result.stdout + help_result.stderr
    for flag in FORBIDDEN_CLI_FLAGS:
        if flag in help_text:
            fail(f"production help exposes forbidden test flag {flag}")

        rejected = subprocess.run(
            [str(production_binary), flag],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rejected.returncode != 2:
            fail(
                f"production CLI did not reject {flag} as an unknown "
                f"argument (status {rejected.returncode})"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-bin", required=True, type=Path)
    parser.add_argument("--test-cli-bin", required=True, type=Path)
    parser.add_argument("--test-runtime-bin", required=True, type=Path)
    parser.add_argument("--production-obj-dir", required=True, type=Path)
    parser.add_argument("--testing-obj-dir", required=True, type=Path)
    args = parser.parse_args()

    assert_production_clean(args.production_bin)
    assert_test_symbols(args.test_cli_bin)
    assert_test_symbols(args.test_runtime_bin)
    assert_cli_surface(args.production_bin)

    for relative in AFFECTED_OBJECTS:
        production_object = args.production_obj_dir / relative
        testing_object = args.testing_obj_dir / relative
        production_blob = read_required(production_object)
        testing_blob = read_required(testing_object)

        assert_production_clean(production_object)
        if production_object.resolve() == testing_object.resolve():
            fail(f"production and testing object paths alias: {relative}")
        if production_blob == testing_blob:
            fail(f"production and testing object bytes are identical: {relative}")

    print("P4-11 production/test isolation checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"P4-11 isolation failure: {exc}", file=sys.stderr)
        raise SystemExit(1)
