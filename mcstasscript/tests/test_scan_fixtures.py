import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


FIXTURES = Path(__file__).resolve().parent / "scan_fixtures"


class TestScanFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "scan_fixture_capture", FIXTURES / "capture.py")
        cls.capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.capture)
        manifest = json.loads((FIXTURES / "expected.json").read_text())
        cls.cases = {case["name"]: case for case in manifest["cases"]}

    def test_verify_without_native_toolchain(self):
        output = io.StringIO()
        with patch("subprocess.run", side_effect=AssertionError(
                "Fixture verification must not execute the native toolchain")) as run:
            with redirect_stdout(output):
                self.capture.verify()
        run.assert_not_called()
        for name in self.cases:
            self.assertIn("Verified " + name + "\n", output.getvalue())

    def test_generator_plans_match_frozen_metadata(self):
        self.assertEqual(self.capture.WORK_ROOT, Path(tempfile.gettempdir()))
        generated = self.capture.cases()
        self.assertEqual({case["name"] for case in generated}, set(self.cases))
        for case in generated:
            with self.subTest(case=case["name"]):
                frozen = self.cases[case["name"]]
                self.assertEqual(case["intended_points"], frozen["intended_points"])
                self.assertEqual(self.capture.native_arguments(case), frozen["command"])

    def test_newer_version_fixtures_verify_without_subprocesses(self):
        spec = importlib.util.spec_from_file_location(
            "scan_fixture_verify_310", FIXTURES / "mcstas_3_10" / "verify.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        output = io.StringIO()
        with patch("subprocess.run", side_effect=AssertionError("No native execution")) as run:
            with redirect_stdout(output):
                verifier.main()
        run.assert_not_called()
        self.assertIn("Verified 29 native files, 15 point/entry records", output.getvalue())

    def test_categorical_summary_uses_indices_not_declared_coordinates(self):
        case = self.cases["ascii_categorical_repeated"]
        planned = case["intended_points"]
        observed = case["observed"]
        self.assertEqual(len(observed["summary_variables"].split()), 5)
        self.assertEqual(len(observed["summary_rows"]), len(planned))
        for intended, native, row in zip(
                planned, observed["points"], observed["summary_rows"]):
            with self.subTest(point_id=intended["point_id"]):
                columns = row.split()
                self.assertEqual(len(columns), 3)
                self.assertEqual(int(columns[0]), intended["point_id"])
                self.assertEqual(native["parameters"]["category"],
                                 intended["parameters"]["category"])
                self.assertNotIn(intended["parameters"]["category"], columns)
        self.assertEqual(planned[0]["parameters"], planned[2]["parameters"])
        self.assertNotEqual(planned[0]["point_id"], planned[2]["point_id"])

    def test_partial_scans_distinguish_incomplete_and_unattempted_ids(self):
        for name in ("ascii_partial", "nexus_partial"):
            with self.subTest(case=name):
                case = self.cases[name]
                planned = {point["point_id"]: point
                           for point in case["intended_points"]}
                observed = case["observed"]
                complete = {point["point_id"] for point in observed["points"]}
                incomplete = {point["point_id"]
                              for point in observed["incomplete_points"]}
                self.assertEqual({key: point["expected_status"]
                                  for key, point in planned.items()},
                                 {0: "completed", 1: "failed",
                                  2: "not_attempted", 3: "not_attempted"})
                self.assertEqual(complete, {0})
                self.assertEqual(incomplete, {1})
                self.assertEqual(set(planned) - complete - incomplete, {2, 3})
                self.assertEqual(set(observed["missing_point_ids"]),
                                 set(planned) - complete)
                failed = observed["incomplete_points"][0]
                self.assertEqual(int(failed["parameters"]["fail"]),
                                 planned[1]["parameters"]["fail"])


if __name__ == "__main__":
    unittest.main()
