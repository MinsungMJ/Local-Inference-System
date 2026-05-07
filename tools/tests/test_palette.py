"""Unit tests for the semantic palette."""

from __future__ import annotations

import unittest

from lis_inspect.palette import (
    SEMANTIC,
    status_role,
    stop_reason_role,
    style_for,
)


class PaletteTest(unittest.TestCase):
    def test_all_required_roles_present(self) -> None:
        for role in (
            "success",
            "warning",
            "error",
            "identity",
            "perf",
            "artifact",
            "muted",
            "neutral",
        ):
            self.assertIn(role, SEMANTIC, f"missing semantic role: {role}")

    def test_style_for_known_role(self) -> None:
        self.assertEqual(style_for("success"), SEMANTIC["success"].style)

    def test_style_for_unknown_role_falls_back(self) -> None:
        self.assertEqual(style_for("not_a_real_role"), "default")

    def test_status_role_ok_is_success(self) -> None:
        self.assertEqual(status_role("OK"), "success")
        self.assertEqual(status_role("ok"), "success")

    def test_status_role_error_branch(self) -> None:
        self.assertEqual(status_role("error"), "error")
        self.assertEqual(status_role("FAILED"), "error")

    def test_status_role_none_is_muted(self) -> None:
        self.assertEqual(status_role(None), "muted")

    def test_stop_reason_role_neutral_for_decode_limit(self) -> None:
        self.assertEqual(stop_reason_role("decode_limit"), "neutral")

    def test_stop_reason_role_error_for_failure_word(self) -> None:
        self.assertEqual(stop_reason_role("internal_error"), "error")
        self.assertEqual(stop_reason_role("decode_failure"), "error")

    def test_stop_reason_role_none_is_muted(self) -> None:
        self.assertEqual(stop_reason_role(None), "muted")


if __name__ == "__main__":
    unittest.main()
