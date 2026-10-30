import subprocess
import sys
import unittest


class TestPackage(unittest.TestCase):
    """Test cases for the top-level termxtract package."""

    def test_top_level_exports(self):
        """Test that the public classes are importable from the package root."""
        import termxtract
        from termxtract.supervised.term_extractor import SupervisedTermExtractor
        from termxtract.unsupervised.term_extractor import TermExtractor
        from termxtract.utils import ATEResults

        self.assertIs(termxtract.TermExtractor, TermExtractor)
        self.assertIs(termxtract.SupervisedTermExtractor, SupervisedTermExtractor)
        self.assertIs(termxtract.ATEResults, ATEResults)
        self.assertEqual(
            sorted(termxtract.__all__),
            ["ATEResults", "SupervisedTermExtractor", "TermExtractor"],
        )

    def test_unknown_attribute(self):
        """Test that unknown attributes raise AttributeError."""
        import termxtract

        with self.assertRaises(AttributeError):
            termxtract.DoesNotExist

    def test_import_is_lazy(self):
        """Test that importing the package does not load heavy dependencies."""
        code = (
            "import sys, termxtract; "
            "from termxtract import TermExtractor; "
            "heavy = [m for m in ('torch', 'transformers', 'wn') if m in sys.modules]; "
            "print(','.join(heavy))"
        )
        # Run in a fresh interpreter, since other tests have already imported these modules.
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "")

    def test_supervised_without_transformers_does_not_load_torch(self):
        """Test that non-transformer supervised methods do not need the optional extra."""
        code = (
            "import sys; "
            "from termxtract import SupervisedTermExtractor; "
            "SupervisedTermExtractor(method='adaboost'); "
            "SupervisedTermExtractor(method='roger'); "
            "heavy = [m for m in ('torch', 'transformers', 'sacremoses') if m in sys.modules]; "
            "print(','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "")

    def test_token_classification_without_extra_raises_clear_error(self):
        """Test that a missing 'transformers' extra gives an install hint."""
        # Simulate torch not being installed. (Setting sys.modules['torch'] = None is not
        # enough: scipy inspects sys.modules['torch'] and fails on None.)
        code = (
            "import sys\n"
            "class BlockTorch:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'torch':\n"
            "            raise ImportError('No module named torch')\n"
            "sys.meta_path.insert(0, BlockTorch())\n"
            "from termxtract import SupervisedTermExtractor\n"
            "SupervisedTermExtractor(method='token-classification')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pip install 'termxtract[transformers]'", result.stderr)


if __name__ == "__main__":
    unittest.main()
