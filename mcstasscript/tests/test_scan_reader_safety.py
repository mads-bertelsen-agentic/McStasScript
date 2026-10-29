from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from mcstasscript import load_scan
from mcstasscript.data.scan import ScanDataError
from mcstasscript.data.scan_ascii import ScanFormatError
from mcstasscript.helper import managed_mcrun
from mcstasscript.tests.test_scan_ascii_edges import copy_fixture, write_point
from mcstasscript.tests.test_scan_nexus import _monitor, _write_entry


FIXTURES = Path(__file__).resolve().parent / "scan_fixtures"
VERSIONED_FIXTURES = FIXTURES / "mcstas_3_10"


def _spectrum_arrays(intensity):
    return {
        "Wavelength__AA_": [0.0, 1.0],
        "data": intensity,
        "errors": [0.0, 0.0] if len(intensity) == 2 else [0.0] * len(intensity),
        "ncount": [1.0] * len(intensity),
    }


def _write_nexus_entries(path, entries):
    with h5py.File(path, "w") as handle:
        for number, parameters, seed, monitors in entries:
            _write_entry(
                handle,
                number,
                parameters=parameters,
                parameter_types={
                    "theta": "double",
                    "wavelength": "double",
                    "category": "string",
                    "fail": "int",
                },
                monitors=monitors,
                seed=seed,
            )


class TestScanReaderSafety(unittest.TestCase):
    def test_competing_directory_sources_require_explicit_hdf5_path(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture("ascii_1d", Path(directory))
            shutil.copyfile(FIXTURES / "nexus_1d" / "mccode.h5", root / "mccode.h5")

            with self.assertRaisesRegex(ScanFormatError, "Competing full ASCII"):
                load_scan(root)

            scan = load_scan(root / "mccode.h5")
            self.assertEqual(scan.execution, {})
            self.assertEqual(scan.metadata["format"], "nexus")
            self.assertEqual(len(scan.points), 3)
            self.assertTrue(all(point.source.format == "nexus" for point in scan.points))
            self.assertEqual(len(scan.points[0].data), 1)

    def test_root_ascii_summary_complements_hdf5_arrays_and_conflicts_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "combined"
            root.mkdir()
            shutil.copyfile(FIXTURES / "nexus_1d" / "mccode.h5", root / "mccode.h5")
            for filename in ("mccode.sim", "mccode.dat"):
                shutil.copyfile(FIXTURES / "ascii_1d" / filename, root / filename)

            scan = load_scan(root)
            self.assertEqual(scan.execution, {})
            np.testing.assert_allclose(
                scan.dataset("spectrum").total_I,
                [6.28296e-7, 1.25659e-6, 1.88489e-6],
            )
            np.testing.assert_allclose(
                scan.dataset("spectrum").total_E,
                [7.8537e-8, 1.57074e-7, 2.35611e-7],
            )
            with h5py.File(FIXTURES / "nexus_1d" / "mccode.h5", "r") as handle:
                expected_array = handle["entry1/data/spectrum_dat/data"][()]
            np.testing.assert_array_equal(scan.points[0].data[0].Intensity, expected_array)

            summary = root / "mccode.dat"
            summary.write_text(summary.read_text().replace("6.28296e-07", "9.0", 1))
            with self.assertRaises(ScanFormatError):
                load_scan(root)

    def test_metadata_only_ascii_points_merge_unique_coordinate_and_seed_matches(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture(
                "mcstas_3_10/ascii_categorical_repeated", Path(directory))
            for point in root.iterdir():
                if point.is_dir() and point.name.isdecimal():
                    (point / "spectrum.dat").unlink()
            (root / "mccode.sim").unlink()
            (root / "mccode.dat").unlink()

            parameters = [
                {"theta": "0", "wavelength": "3", "category": "vanadium", "fail": "0"},
                {"theta": "1", "wavelength": "5", "category": "nickel", "fail": "0"},
                {"theta": "0", "wavelength": "3", "category": "vanadium", "fail": "0"},
            ]
            entries = []
            for number, (values, seed, intensity) in enumerate(
                    zip(parameters, (314159, 315183, 316207), (1.0, 2.0, 3.0)), 1):
                entries.append((
                    number,
                    values,
                    seed,
                    (_monitor(
                        "spectrum",
                        _spectrum_arrays([intensity, intensity + 1.0]),
                        dimension=2,
                        values=None,
                    ),),
                ))
            _write_nexus_entries(root / "mccode.h5", entries)
            scan = load_scan(root)
            self.assertEqual(scan.execution, {})
            self.assertEqual(len(scan.points), 3)
            self.assertEqual(
                [point.parameters["category"] for point in scan.points],
                ["vanadium", "nickel", "vanadium"],
            )
            for point, intensity in zip(scan.points, (1.0, 2.0, 3.0)):
                self.assertEqual(point.source.format, "nexus")
                np.testing.assert_array_equal(
                    point.data[0].Intensity, [intensity, intensity + 1.0])

    def test_duplicate_metadata_points_without_distinguishable_nexus_match_reject(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            monitor = {
                "component": "spectrum",
                "filename": "spectrum.dat",
                "bins": 2,
                "values": "1.0 0.1 1",
                "data": None,
            }
            for number in (0, 1):
                write_point(
                    root / str(number),
                    [("theta", "double")],
                    [("theta", "0")],
                    [monitor],
                    seed=7,
                )
            _write_nexus_entries(
                root / "mccode.h5",
                [(1, {"theta": "0"}, 7, (_monitor(
                    "spectrum",
                    _spectrum_arrays([1.0, 1.0]),
                    dimension=2,
                    values=None,
                ),))],
            )

            with self.assertRaises(ScanFormatError):
                load_scan(root)

    def test_known_ascii_summary_quantities_survive_unknown_nexus_quantities(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_point(
                root / "0",
                [("theta", "double")],
                [("theta", "0")],
                [{
                    "component": "spectrum",
                    "filename": "spectrum.dat",
                    "bins": 2,
                    "values": "1.0 2.0",
                    "data": None,
                }],
                seed=11,
            )
            monitor = _monitor(
                "spectrum",
                _spectrum_arrays([4.0, 5.0]),
                dimension=2,
                values=None,
            )
            monitor["metadata"].info["values"] = "1.0"
            _write_nexus_entries(
                root / "mccode.h5",
                [(1, {"theta": "0"}, 11, (monitor,))],
            )

            scan = load_scan(root)

        self.assertEqual(scan.execution, {})
        view = scan.dataset("spectrum")
        self.assertEqual(view.total_I.tolist(), [1.0])
        self.assertEqual(view.total_E.tolist(), [2.0])
        self.assertTrue(np.isnan(view.total_N[0]))
        self.assertEqual(view.availability["total_I"].tolist(), [True])
        self.assertEqual(view.availability["total_E"].tolist(), [True])
        self.assertEqual(view.availability["total_N"].tolist(), [False])

    def test_summary_only_new_native_categorical_rows_keep_numeric_codes(self):
        with TemporaryDirectory() as directory:
            root = copy_fixture(
                "mcstas_3_10/ascii_categorical_repeated", Path(directory))
            for point in root.iterdir():
                if point.is_dir() and point.name.isdecimal():
                    shutil.rmtree(point)

            scan = load_scan(root)

        self.assertEqual(scan.execution, {})
        self.assertEqual(
            [point.metadata["encoded_parameters"]["category"] for point in scan.points],
            [0.0, 1.0, 0.0],
        )
        for point in scan.points:
            self.assertNotIn("category", point.parameters)
            self.assertNotIn("vanadium", point.parameters.values())
            self.assertNotIn("nickel", point.parameters.values())

    def test_unknown_monitor_type_is_a_scan_format_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_point(
                root / "0",
                [("theta", "double")],
                [("theta", "0")],
                [{
                    "component": "spectrum",
                    "filename": "spectrum.dat",
                    "bins": 2,
                    "values": "1.0 0.1 1",
                    "data": "1.0 1.0 0.1 1\n2.0 1.0 0.1 1\n",
                }],
            )
            sim = root / "0" / "mccode.sim"
            sim.write_text(sim.read_text().replace(
                "type: array_1d(2)", "type: unsupported_monitor_type"))

            with self.assertRaises(ScanFormatError):
                load_scan(root)

    def test_malformed_nexus_arrays_keep_good_monitors_and_report_failures(self):
        good = _monitor(
            "good",
            {
                "Wavelength__AA_": [0.0, 1.0],
                "data": [1.0, 2.0],
                "errors": [0.1, 0.2],
                "ncount": [1.0, 2.0],
            },
            dimension=2,
            values="3.0 0.3 3",
        )
        wrong_shape = _monitor(
            "wrong_shape",
            {
                "Wavelength__AA_": [0.0, 1.0],
                "data": [1.0, 2.0, 3.0],
                "errors": [0.1, 0.2],
                "ncount": [1.0, 2.0],
            },
            dimension=2,
            values="4.0 0.4 4",
        )
        nonnumeric = _monitor(
            "nonnumeric",
            {
                "Wavelength__AA_": [0.0, 1.0],
                "data": [b"bad", b"array"],
                "errors": [0.1, 0.2],
                "ncount": [1.0, 2.0],
            },
            dimension=2,
            values="5.0 0.5 5",
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            _write_nexus_entries(
                path,
                [
                    (1, {"theta": "0"}, 1, (good, wrong_shape, nonnumeric)),
                    (2, {"theta": "1"}, 2, (wrong_shape, nonnumeric)),
                ],
            )
            scan = load_scan(path)
            self.assertEqual(scan.execution, {})

            partial = scan.points[0]
            with self.assertWarns(UserWarning):
                data = partial.data
            self.assertEqual([dataset.name for dataset in data], ["good"])
            self.assertEqual({monitor.component for monitor in partial.diagnostics},
                             {"wrong_shape", "nonnumeric"})
            self.assertEqual(
                partial.array_availability[
                    next(monitor for monitor in partial.array_availability
                         if monitor.component == "good")],
                {"Intensity": True, "Error": True, "Ncount": True},
            )
            for monitor, diagnostic in partial.diagnostics.items():
                self.assertIn("/data/", diagnostic)
                self.assertFalse(partial.array_availability[monitor]["Intensity"])

            failed = scan.points[1]
            with self.assertRaises(ScanDataError) as caught:
                _ = failed.data

        self.assertTrue(caught.exception.diagnostics)
        self.assertEqual(
            {monitor.component for monitor in caught.exception.diagnostics},
            {"wrong_shape", "nonnumeric"},
        )
        self.assertEqual(
            {monitor.component for monitor in caught.exception.availability},
            {"wrong_shape", "nonnumeric"},
        )
        self.assertEqual(
            {monitor.component for monitor in failed.diagnostics},
            {"wrong_shape", "nonnumeric"},
        )

    def test_all_missing_ascii_files_retain_failure_diagnostics_and_availability(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_point(
                root / "0",
                [("theta", "double")],
                [("theta", "0")],
                [
                    {
                        "component": "missing_one",
                        "filename": "missing_one.dat",
                        "bins": 2,
                        "values": "1.0 0.1 1",
                        "data": None,
                    },
                    {
                        "component": "missing_two",
                        "filename": "missing_two.dat",
                        "bins": 2,
                        "values": "2.0 0.2 2",
                        "data": None,
                    },
                ],
            )
            scan = load_scan(root)
            point = scan.points[0]
            self.assertEqual(scan.execution, {})
            with self.assertRaises(ScanDataError) as caught:
                _ = point.data

        self.assertEqual(caught.exception.data_status, "missing")
        self.assertEqual(
            {monitor.component for monitor in point.diagnostics},
            {"missing_one", "missing_two"},
        )
        for monitor, diagnostic in point.diagnostics.items():
            self.assertIn(monitor.filename, diagnostic)
            self.assertEqual(
                point.array_availability[monitor],
                {"Intensity": False, "Error": False, "Ncount": False},
            )

    def test_shared_single_run_loader_rejects_a_scan_root(self):
        with self.assertRaisesRegex(RuntimeError, "load_scan"):
            managed_mcrun.load_results(FIXTURES / "ascii_1d")


if __name__ == "__main__":
    unittest.main()
