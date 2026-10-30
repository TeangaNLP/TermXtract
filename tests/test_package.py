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


if __name__ == "__main__":
    unittest.main()
