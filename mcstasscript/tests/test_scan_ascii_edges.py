from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mcstasscript import load_scan
from mcstasscript.data.scan import ScanDataError
from mcstasscript.data.scan_ascii import ScanFormatError, _parse_value, _point_records


FIXTURES = Path(__file__).resolve().parent / "scan_fixtures"


def copy_fixture(relative_path, destination):
    source = FIXTURES / relative_path
    target = destination / source.name
    shutil.copytree(source, target)
    return target


def write_point(path, declarations, parameters, monitors, seed=1):
    path.mkdir(parents=True)
    lines = [
        "begin instrument: generated",
        "  Parameters: " + " ".join(
            "{}({})".format(name, kind) for name, kind in declarations),
        "end instrument",
        "",
        "begin simulation: generated",
        "  Seed: {}".format(seed),
    ]
    lines.extend("  Param: {}={}".format(name, value) for name, value in parameters)
    lines.extend(("end simulation", ""))
    for monitor in monitors:
        lines.extend([
            "begin data",
            "  type: array_1d({})".format(monitor["bins"]),
            "  component: {}".format(monitor["component"]),
            "  filename: {}".format(monitor["filename"]),
            "  values: {}".format(monitor["values"]),
            "  variables: L I I_err N",
            "end data",
            "",
        ])
    (path / "mccode.sim").write_text("\n".join(lines))
    for monitor in monitors:
        if monitor["data"] is not None:
            (path / monitor["filename"]).write_text(monitor["data"])


class TestAsciiScanEdgeCases(unittest.TestCase):
    def test_native_string_apostrophes_survive_shared_ascii_and_nexus_parser(self):
        self.assertEqual(_parse_value("'single'", "string"), "'single'")
        self.assertEqual(_parse_value(r'\"single\"', "string"), '"single"')
        self.assertEqual(_parse_value('"two words"', "string"), "two words")

        with TemporaryDirectory() as directory:
            point_path = Path(directory) / "0"
            write_point(
                point_path,
                [("category", "string")],
                [("category", "'single'")],
                [{"component": "spectrum", "filename": "spectrum.dat",
                  "bins": 1, "values": "1 0 1", "data": "0 1 0 1\n"}],
            )

            point = _point_records(point_path)

        self.assertEqual(point.parameters["category"], "'single'")

    def test_native_string_parameters_unescape_quotes_and_backslashes(self):
        with TemporaryDirectory() as directory:
            point_path = Path(directory) / "0"
            write_point(
                point_path,
                [("category", "string")],
                [("category", r'quote\"mark\\slash')],
                [{"component": "spectrum", "filename": "spectrum.dat",
                  "bins": 1, "values": "1 0 1", "data": "0 1 0 1\n"}],
            )

            point = _point_records(point_path)

        self.assertEqual(point.parameters["category"], r'quote"mark\slash')

    def test_numbered_points_load_without_summary(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture("ascii_1d", Path(directory))
            (root / "mccode.dat").unlink()

            scan = load_scan(root)

        self.assertEqual([point.point_id for point in scan.points], [0, 1, 2])
        self.assertEqual([point.parameters["theta"] for point in scan.points], [-1, 0, 1])
        self.assertEqual(scan.execution, {})
        np.testing.assert_allclose(
            scan.dataset("spectrum").total_I,
            [6.28296e-7, 1.25659e-6, 1.88489e-6],
        )

    def test_custom_summary_preserves_row_order_and_unknown_counts(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "custom.txt"
            summary.write_text(
                "# xvars: wavelength, theta\n"
                "# yvars: (spectrum_I,spectrum_ERR)\n"
                "# variables: wavelength theta spectrum_I spectrum_ERR\n"
                "6.25 -1.0 3.5 0.5\n"
                "1.25 2.0 1.25 0.125\n"
            )

            scan = load_scan(summary)

        self.assertEqual(scan.points[0].point_id, 0)
        self.assertEqual(scan.points[1].point_id, 1)
        self.assertEqual(scan.parameter_names, ("wavelength", "theta"))
        self.assertEqual(scan.coordinates["wavelength"].tolist(), [6.25, 1.25])
        self.assertEqual(scan.coordinates["theta"].tolist(), [-1.0, 2.0])
        view = scan.dataset("spectrum")
        np.testing.assert_allclose(view.total_I, [3.5, 1.25])
        np.testing.assert_allclose(view.total_E, [0.5, 0.125])
        self.assertTrue(np.isnan(view.total_N).all())
        self.assertEqual(view.availability["total_N"].tolist(), [False, False])
        self.assertIsNone(scan.metadata["declared_numpoints"])
        self.assertEqual(scan.execution, {})

    def test_old_categorical_summary_preserves_explicit_ids(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "categorical.txt"
            shutil.copyfile(
                FIXTURES / "ascii_categorical_repeated" / "mccode.dat", summary)
            header = "\n".join(
                line for line in summary.read_text().splitlines()
                if line.lstrip().startswith("#")
            )
            summary.write_text(
                header + "\n"
                "7 1.0 0.1\n"
                "2 2.0 0.2\n"
                "11 3.0 0.3\n"
            )

            scan = load_scan(summary)

        self.assertEqual([point.point_id for point in scan.points], [7, 2, 11])
        self.assertEqual(
            [point.metadata["encoded_point_id"] for point in scan.points], [7, 2, 11]
        )
        np.testing.assert_allclose(scan.dataset("spectrum").total_I, [1.0, 2.0, 3.0])
        self.assertEqual(scan.execution, {})

    def test_truncated_numeric_row_is_not_treated_as_categorical(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "truncated.txt"
            summary.write_text(
                "# Param: theta = 0, category = vanadium\n"
                "# list: ['0']\n"
                "# xvars: theta, category\n"
                "# yvars: (spectrum_I,spectrum_ERR)\n"
                "# variables: theta category spectrum_I spectrum_ERR\n"
                "0 0 1.0 0.1\n"
                "1 2.0 0.2\n"
            )

            with self.assertRaisesRegex(ScanFormatError, "line"):
                load_scan(summary)

    def test_missing_point_metadata_keeps_unmatched_summary_useful(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture("ascii_1d", Path(directory))
            (root / "1" / "mccode.sim").unlink()

            scan = load_scan(root)

        self.assertEqual(len(scan.points), 3)
        summary_only = [
            point for point in scan.points if point.parameters.get("theta") == 0
        ]
        self.assertEqual(len(summary_only), 1)
        point = summary_only[0]
        self.assertIsNone(point.source)
        self.assertEqual(point.status, "unknown")
        self.assertEqual(point.data_status, "missing")
        self.assertIsNone(point.metadata["encoded_point_id"])
        monitor_summary = next(iter(point.summaries.values()))
        self.assertAlmostEqual(monitor_summary.total_I, 1.25659e-6)
        self.assertEqual(scan.execution, {})

    def test_duplicate_summary_does_not_attach_success_to_failed_duplicate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(FIXTURES / "ascii_partial" / "1", root / "0")
            shutil.copytree(FIXTURES / "ascii_partial" / "0", root / "1")
            failed_metadata = root / "0" / "mccode.sim"
            failed_metadata.write_text(
                failed_metadata.read_text()
                .replace("Param: theta=1", "Param: theta=0")
                .replace("Param: fail=1", "Param: fail=0")
            )
            (root / "mccode.dat").write_text(
                "# xvars: fail, theta\n"
                "# yvars: (spectrum_I,spectrum_ERR)\n"
                "# variables: fail theta spectrum_I spectrum_ERR\n"
                "0 0 1.25659e-6 1.57074e-7\n"
                "0 0 1.25659e-6 1.57074e-7\n"
            )

            scan = load_scan(root)

        failed = next(point for point in scan.points if point.point_id == 0)
        successful = next(point for point in scan.points if point.point_id == 1)
        self.assertEqual(failed.summaries, {})
        successful_summary = next(iter(successful.summaries.values()))
        self.assertAlmostEqual(successful_summary.total_I, 1.25659e-6)
        preserved = scan.metadata["unassociated_summary_rows"]
        self.assertTrue(
            preserved
            or any("Ambiguous" in issue for issue in scan.metadata["diagnostics"])
            or any(
                point.source is None and point.metadata.get("summary_line")
                for point in scan.points
            )
        )

    def test_string_parameters_survive_lazy_dataset_metadata(self):
        parameters = {
            "colon": "alpha:beta",
            "equals": "left=right",
            "comma": "one,two",
            "spaced": "two words",
            "numeric": "001",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_point(
                root / "0",
                [(name, "string") for name in parameters],
                [("spaced", '"two words"')] + [
                    (name, value) for name, value in parameters.items()
                    if name != "spaced"
                ],
                [{
                    "component": "spectrum",
                    "filename": "spectrum.dat",
                    "bins": 2,
                    "values": "3.0 0.3 2",
                    "data": "1.0 2.0 0.2 1\n2.0 1.0 0.1 1\n",
                }],
            )
            scan = load_scan(root)
            point = scan.points[0]
            self.assertEqual(point.parameters, parameters)
            self.assertFalse(point.is_loaded)
            dataset = point.data[0]

        self.assertEqual(dataset.metadata.info["Parameters"], parameters)
        self.assertEqual(dataset.metadata.parameters, parameters)

    def test_one_bin_detector_keeps_one_dimensional_arrays(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_point(
                root / "0",
                [("theta", "double")],
                [("theta", "0")],
                [{
                    "component": "spectrum",
                    "filename": "spectrum.dat",
                    "bins": 1,
                    "values": "2.0 0.2 1",
                    "data": "1.0 2.0 0.2 1\n",
                }],
            )
            dataset = load_scan(root).points[0].data[0]

        self.assertEqual(dataset.Intensity.shape, (1,))
        self.assertEqual(dataset.Error.shape, (1,))
        self.assertEqual(dataset.Ncount.shape, (1,))
        np.testing.assert_allclose(dataset.Intensity, [2.0])

    def test_partial_monitor_loading_warns_and_all_invalid_reports_diagnostics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            monitor_metadata = [
                {
                    "component": "good",
                    "filename": "good.dat",
                    "bins": 2,
                    "values": "3.0 0.3 2",
                    "data": "1.0 2.0 0.2 1\n2.0 1.0 0.1 1\n",
                },
                {
                    "component": "missing",
                    "filename": "missing.dat",
                    "bins": 2,
                    "values": "4.0 0.4 2",
                    "data": None,
                },
                {
                    "component": "corrupt",
                    "filename": "corrupt.dat",
                    "bins": 2,
                    "values": "5.0 0.5 2",
                    "data": "not numeric\n",
                },
            ]
            write_point(root / "0", [("theta", "double")], [("theta", "0")],
                        monitor_metadata)
            all_invalid = [dict(monitor) for monitor in monitor_metadata[1:]]
            write_point(root / "1", [("theta", "double")], [("theta", "1")],
                        all_invalid)
            scan = load_scan(root)

            partial = scan.points[0]
            with self.assertWarns(UserWarning):
                data = partial.data
            self.assertEqual([dataset.name for dataset in data], ["good"])
            self.assertEqual(partial.data_status, "partial")
            self.assertEqual(
                {monitor.component for monitor in partial.diagnostics},
                {"missing", "corrupt"},
            )
            for diagnostic in partial.diagnostics.values():
                self.assertIn(str(partial.source.path), diagnostic)

            failed = scan.points[1]
            with self.assertRaises(ScanDataError) as caught:
                _ = failed.data

        message = str(caught.exception)
        self.assertIn(str(failed.source.path), message)
        self.assertIn("missing.dat", message)
        self.assertIn("corrupt.dat", message)
        self.assertTrue(failed.diagnostics)

    def test_native_310_hashseed_case_keeps_point_metadata_and_no_guessed_ids(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture("mcstas_3_10/ascii_hashseed_1", Path(directory))
            scan = load_scan(root)

        expected = [
            {"theta": -1.0, "wavelength": 2.0, "category": "vanadium", "fail": 0},
            {"theta": 0.0, "wavelength": 4.0, "category": "vanadium", "fail": 0},
            {"theta": 1.0, "wavelength": 6.0, "category": "vanadium", "fail": 0},
        ]
        self.assertEqual([point.parameters for point in scan.points], expected)
        self.assertEqual([point.metadata["seed"] for point in scan.points],
                         [314159, 315183, 316207])
        self.assertEqual([point.point_id for point in scan.points], [0, 1, 2])
        self.assertTrue(all(point.status == "unknown" for point in scan.points))
        self.assertEqual(scan.execution, {})
        self.assertNotIn("unobserved_point_ids", scan.execution)
        if scan.metadata["unassociated_summary_rows"]:
            self.assertTrue(scan.metadata["diagnostics"])
        else:
            np.testing.assert_allclose(
                scan.dataset("spectrum").total_I,
                [6.28298e-7, 1.25659e-6, 1.88488e-6],
            )

    def test_native_310_categorical_case_keeps_authoritative_parameters(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture("mcstas_3_10/ascii_categorical_repeated", Path(directory))
            scan = load_scan(root)

        self.assertEqual(
            [point.parameters["category"] for point in scan.points],
            ["vanadium", "nickel", "vanadium"],
        )
        self.assertEqual([point.parameters["theta"] for point in scan.points], [0.0, 1.0, 0.0])
        self.assertEqual([point.point_id for point in scan.points], [0, 1, 2])
        self.assertEqual([point.metadata["seed"] for point in scan.points],
                         [314159, 315183, 316207])
        self.assertNotEqual(scan.points[0].point_id, scan.points[2].point_id)
        self.assertEqual(scan.points[0].parameters, scan.points[2].parameters)
        self.assertTrue(all(point.status == "unknown" for point in scan.points))
        self.assertEqual(scan.execution, {})

    def test_native_310_partial_case_keeps_failed_metadata_without_unobserved_ids(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture("mcstas_3_10/ascii_partial", Path(directory))
            scan = load_scan(root)

        expected = [
            (0.0, 0),
            (1.0, 1),
            (2.0, 0),
            (3.0, 0),
        ]
        self.assertEqual(
            [(point.parameters["theta"], point.parameters["fail"])
             for point in scan.points],
            expected,
        )
        self.assertTrue(all(
            point.parameters["category"] == "vanadium" and
            point.parameters["wavelength"] == 3.0
            for point in scan.points
        ))
        self.assertEqual([point.point_id for point in scan.points], [0, 1, 2, 3])
        self.assertEqual([point.metadata["seed"] for point in scan.points],
                         [314159, 315183, 316207, 317231])
        self.assertTrue(all(point.status == "unknown" for point in scan.points))
        self.assertEqual(scan.execution, {})
        self.assertNotIn("missing_point_ids", scan.execution)
        self.assertNotIn("unobserved_point_ids", scan.execution)


if __name__ == "__main__":
    unittest.main()
