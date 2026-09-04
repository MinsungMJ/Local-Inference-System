import sys
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TestPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    def test_project_is_dependency_free_and_has_console_entry(self):
        project = self.pyproject["project"]
        self.assertEqual(project["dependencies"], [])
        self.assertNotIn("version", project)
        self.assertEqual(project["dynamic"], ["version"])
        self.assertEqual(
            self.pyproject["tool"]["setuptools"]["dynamic"]["version"],
            {"attr": "lis_verify.__version__"},
        )
        self.assertEqual(project["scripts"]["lis-verify"], "lis_verify.cli:main")
        self.assertEqual(project["requires-python"], ">=3.10")

    def test_textual_is_only_an_optional_inspect_dependency(self):
        project = self.pyproject["project"]
        self.assertEqual(project["optional-dependencies"]["inspect"], ["textual>=0.60"])

    def test_importing_cli_does_not_import_textual(self):
        before = set(sys.modules)
        import lis_verify.cli  # noqa: F401

        imported = set(sys.modules) - before
        self.assertFalse(any(name == "textual" or name.startswith("textual.") for name in imported))

    def test_only_product_packages_are_discovered(self):
        config = self.pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(config["where"], ["tools"])
        self.assertEqual(config["include"], ["lis_verify*", "lis_inspect*"])


if __name__ == "__main__":
    unittest.main()
