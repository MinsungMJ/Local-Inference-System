import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tools" / "test_fixtures" / "lis_verify_contract"
AUDIT = FIXTURE_ROOT / "documentation_status_audit_v1.json"
CONTRACT = FIXTURE_ROOT / "product_contract_v1.json"
PUBLIC_DOCS = (
    ROOT / "docs" / "differential_verification.md",
    ROOT / "docs" / "calibration_preflight.md",
    ROOT / "docs" / "repro_execution_artifacts.md",
    ROOT / "docs" / "verification_framework.md",
    ROOT / "docs" / "lis_verify_contract.md",
)


class TestDocumentationStatusAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(AUDIT.read_text())

    def test_audit_has_no_unresolved_or_stale_entry(self):
        self.assertEqual(self.audit["audit_status"], "resolved")
        self.assertEqual(self.audit["unresolved_entries"], [])
        allowed = set(self.audit["classifications"])
        self.assertEqual(allowed, {"current", "historical_frozen"})
        for entry in self.audit["capabilities"]:
            self.assertIn(entry["classification"], allowed)
            self.assertNotEqual(entry["status"], "unknown")

    def test_all_document_assertions_are_present(self):
        for entry in self.audit["document_assertions"]:
            path = ROOT / entry["document"]
            self.assertTrue(path.is_file(), entry["document"])
            document = " ".join(path.read_text().split())
            required = " ".join(entry["required_text"].split())
            self.assertIn(required, document, entry)

    def test_known_stale_pass_status_statement_is_absent(self):
        calibration = (ROOT / "docs" / "calibration_preflight.md").read_text()
        self.assertNotIn("Passes 2–6 remain planned", calibration)

    def test_forced_prefix_design_is_not_claimed_as_implemented(self):
        contract = " ".join(
            (ROOT / "docs" / "lis_verify_contract.md").read_text().split()
        )
        self.assertIn("remains unimplemented until M3", contract)
        self.assertIn("artifact_supported = false", contract)
        self.assertIn("current C CLI rejection", contract)

    def test_pass3_and_pass4_nonclaims_remain_visible(self):
        differential = " ".join(
            (ROOT / "docs" / "differential_verification.md").read_text().split()
        )
        for text in (
            "do not prove tensor equality",
            "confirm the first numeric or operation-level divergence",
            "does not provide the layout required for layer or intra-layer localization",
        ):
            self.assertIn(text, differential)
        product = " ".join(
            (ROOT / "docs" / "lis_verify_contract.md").read_text().split()
        )
        for text in (
            "bounded digest equality is not tensor equality",
            "bounded digest mismatch is not numeric confirmation",
            "partial coverage is not whole-runtime equivalence",
        ):
            self.assertIn(text, product)

    def test_public_files_do_not_reference_private_archive(self):
        forbidden = re.compile(r"(?:/home/mj/Repos/)?temp_docs" r"_LIS")
        for path in PUBLIC_DOCS:
            self.assertIsNone(forbidden.search(path.read_text()), path)
        self.assertFalse(self.audit["public_repository_references_private_archive"])


class TestMarkdownFixtureParity(unittest.TestCase):
    def test_contract_index_matches_machine_readable_fixture(self):
        markdown = (ROOT / "docs" / "lis_verify_contract.md").read_text()
        match = re.search(
            r"<!-- LIS-VERIFY-CONTRACT-INDEX-BEGIN -->\s*"
            r"```json\s*(.*?)\s*```\s*"
            r"<!-- LIS-VERIFY-CONTRACT-INDEX-END -->",
            markdown,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        index = json.loads(match.group(1))
        fixture = json.loads(CONTRACT.read_text())
        self.assertGreater(len(index), 0)
        for key, value in index.items():
            self.assertIn(key, fixture)
            self.assertEqual(value, fixture[key], key)


if __name__ == "__main__":
    unittest.main()
