import importlib.util
import unittest


class ImportBoundaryTests(unittest.TestCase):
    def test_core_modules_import_without_initializing_judge(self):
        import riser
        from riser.primitives.filtering import LLMJudgeFilter

        self.assertIsNotNone(riser)
        self.assertIsNotNone(LLMJudgeFilter)

    def test_judge_dependency_is_optional_until_instantiation(self):
        from riser.primitives.filtering import LLMJudgeFilter

        if importlib.util.find_spec("anthropic") is None:
            with self.assertRaises(ImportError):
                LLMJudgeFilter()
        else:
            self.skipTest("anthropic is installed; optional-import failure path is not applicable")


if __name__ == "__main__":
    unittest.main()
