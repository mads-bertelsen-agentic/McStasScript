import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from mcstasscript import load_data, load_scan
from mcstasscript.data.collection import McStasDataCollection
from mcstasscript.data.scan import ScanDataError
from mcstasscript.data.scan_ascii import ScanFormatError


FIXTURES = Path(__file__).resolve().parent / "scan_fixtures"


class TestAsciiScanLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((FIXTURES / "expected.json").read_text())
        cls.cases = {case["name"]: case for case in manifest["cases"]}

    def test_native_ascii_cases_load_without_subprocesses(self):
        names = ("ascii_1d", "ascii_cartesian_2x3", "ascii_parallel",
                 "ascii_categorical_repeated", "ascii_paired")
        for name in names:
            with self.subTest(case=name):
                case = self.cases[name]
                with patch("subprocess.run", side_effect=AssertionError(
                        "ASCII loading must not execute a subprocess")) as run:
                    scan = load_scan(FIXTURES / name)
                    monitor = scan.dataset("spectrum")
                    data = scan.points[0].data
                run.assert_not_called()

                expected_points = case["observed"]["points"]
                self.assertEqual(len(scan.points), len(expected_points))
                for point, expected in zip(scan.points, expected_points):
                    self.assertEqual(point.point_id, expected["point_id"])
                    self.assertEqual(point.metadata["seed"], expected["seed"])
                    for key, value in expected["parameters"].items():
                        actual = point.parameters[key]
                        try:
                            expected_value = int(value)
                        except ValueError:
                            try:
                                expected_value = float(value)
                            except ValueError:
                                expected_value = value
                        self.assertEqual(actual, expected_value)
                expected_totals = [float(row.split()[-2])
                                   for row in case["observed"]["summary_rows"]]
                np.testing.assert_allclose(monitor.total_I, expected_totals)
                self.assertIsInstance(data, McStasDataCollection)
                self.assertEqual(data[0].name, "spectrum")
                self.assertEqual(data[0].Intensity.shape, (4,))
                self.assertEqual(scan.metadata["format"], "ascii")
                self.assertEqual(scan.execution, {})

    def test_point_metadata_wins_over_native_summary_coordinate_header(self):
        scan = load_scan(FIXTURES / "ascii_cartesian_2x3")
        np.testing.assert_array_equal(
            scan.dataset("spectrum").coordinates["theta"], [-1, 1, -1, 1, -1, 1]
        )
        np.testing.assert_array_equal(
            scan.dataset("spectrum").coordinates["wavelength"], [2, 2, 4, 4, 6, 6]
        )

    def test_partial_ascii_scan_preserves_metadata_without_inventing_unobserved_points(self):
        scan = load_scan(FIXTURES / "ascii_partial")
        self.assertEqual([point.point_id for point in scan.points], [0, 1])
        self.assertEqual(scan.points[1].parameters["theta"], 1)
        self.assertEqual(scan.points[1].parameters["fail"], 1)
        self.assertEqual(scan.execution, {})
        self.assertEqual(scan.metadata["declared_numpoints"], "4")
        self.assertTrue(scan.metadata["diagnostics"])
        self.assertEqual(scan.points[1].status, "unknown")
        with self.assertRaises(ScanDataError) as caught:
            _ = scan.points[1].data
        self.assertEqual(caught.exception.data_status, "missing")
        self.assertEqual(scan.points[1].data_status, "missing")
        np.testing.assert_allclose(scan.dataset("spectrum").total_I,
                                   [1.25659e-6, np.nan], equal_nan=True)

    def test_categorical_values_and_repeated_coordinates_are_not_reconstructed(self):
        scan = load_scan(FIXTURES / "ascii_categorical_repeated")
        self.assertEqual([point.parameters["category"] for point in scan.points],
                         ["vanadium", "nickel", "vanadium"])
        self.assertEqual(scan.points[0].point_id, 0)
        self.assertEqual(scan.points[2].point_id, 2)
        self.assertEqual(scan.points[0].parameters, scan.points[2].parameters)
        self.assertNotEqual(scan.points[0].point_id, scan.points[2].point_id)

    def test_standalone_and_custom_summary_files(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            standalone = directory / "summary.dat"
            standalone.write_text(
                "# Numpoints: 2\n"
                "# xvars: theta, category\n"
                "# yvars: (spectrum_I,spectrum_ERR)\n"
                "# variables: theta category spectrum_I spectrum_ERR\n"
                "0 \"nickel,vanadium\" 1.0 0.1\n"
                "1 \"vanadium, copper\" 2.0 0.2\n"
            )
            scan = load_scan(standalone)
            self.assertEqual(scan.points[0].parameters["category"], "nickel,vanadium")
            self.assertEqual(scan.points[1].parameters["category"], "vanadium, copper")
            self.assertIsNone(scan.points[0].source)
            self.assertIsNone(scan.dataset("spectrum").monitor.filename)

            shutil.copyfile(FIXTURES / "ascii_1d" / "mccode.dat",
                            directory / "custom-summary.dat")
            sim_text = (FIXTURES / "ascii_1d" / "mccode.sim").read_text()
            (directory / "mccode.sim").write_text(
                sim_text.replace("filename: mccode.dat", "filename: custom-summary.dat")
            )
            custom = load_scan(directory)
            self.assertEqual(len(custom.points), 3)
            self.assertEqual(custom.metadata["summary"], str(directory / "custom-summary.dat"))

    def test_malformed_and_unsupported_inputs_are_rejected(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "broken.dat"
            summary.write_text(
                "# variables: theta spectrum_I spectrum_ERR\n"
                "0 1.0\n"
            )
            with self.assertRaises(ScanFormatError):
                load_scan(summary)

        with self.assertRaises(FileNotFoundError):
            load_scan(FIXTURES / "nonexistent")

    def test_single_run_loader_explains_recognized_scan_root(self):
        with self.assertRaisesRegex(RuntimeError, "load_scan"):
            load_data(FIXTURES / "ascii_1d")


if __name__ == "__main__":
    unittest.main()
