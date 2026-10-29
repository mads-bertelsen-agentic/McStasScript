import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from mcstasscript import load_data, load_scan
from mcstasscript.data.collection import McStasDataCollection
from mcstasscript.data.data import McStasDataBinned, McStasDataEvent, McStasMetaData
from mcstasscript.data.scan import ScanDataError


FIXTURES = Path(__file__).resolve().parent / "scan_fixtures"
VERSIONED_FIXTURES = FIXTURES / "mcstas_3_10"


def _bytes(value):
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _monitor_metadata(component, *, filename=None, dimension=2, values=None,
                      variables="L I I_err N", xlabel="Wavelength [AA]"):
    """Build the same metadata that native McStas output stores on NXdata."""
    metadata = McStasMetaData()
    if dimension == 0:
        data_type = "array_0d"
    elif isinstance(dimension, int):
        data_type = "array_1d({})".format(dimension)
    else:
        data_type = "array_2d({},{})".format(*dimension)
    metadata.add_info("type", data_type)
    metadata.add_info("component", component)
    metadata.add_info("filename", filename or component + ".dat")
    metadata.add_info("variables", variables)
    if dimension != 0:
        metadata.add_info("xlabel", xlabel)
        metadata.add_info("xlimits", "0 1")
    if values is not None:
        metadata.add_info("values", values)
    metadata.extract_info()
    return metadata


def _monitor(component, arrays, *, field=None, filename=None, dimension=2,
             values=None, variables="L I I_err N", xlabel="Wavelength [AA]"):
    return {
        "field": field or component + "_dat",
        "metadata": _monitor_metadata(
            component,
            filename=filename,
            dimension=dimension,
            values=values,
            variables=variables,
            xlabel=xlabel,
        ),
        "arrays": arrays,
    }


def _write_entry(handle, number, *, parameters=None, parameter_types=None,
                 monitors=(), seed=314159):
    parameters = parameters or {"theta": b"0"}
    parameter_types = parameter_types or {name: "double" for name in parameters}
    declarations = " ".join(
        "{}({})".format(name, parameter_types[name]) for name in parameters
    )

    entry = handle.create_group("entry{}".format(number))
    instrument = entry.create_group("instrument")
    instrument.attrs["NX_class"] = b"NXinstrument"
    instrument.attrs["Parameters"] = _bytes(" " + declarations)

    simulation = entry.create_group("simulation")
    simulation.attrs["Seed"] = _bytes(seed)
    parameter_group = simulation.create_group("Param")
    parameter_group.attrs["NX_class"] = b"NXparameters"
    for name, value in parameters.items():
        raw_value = _bytes(value)
        parameter_group.attrs[name] = raw_value
        parameter_group.create_dataset(
            name,
            data=np.asarray([raw_value], dtype="S{}".format(max(1, len(raw_value)))),
        )

    data_group = entry.create_group("data")
    data_group.attrs["NX_class"] = b"NXdetector"
    for monitor in monitors:
        metadata = monitor["metadata"]
        monitor_group = data_group.create_group(monitor["field"])
        monitor_group.attrs["NX_class"] = b"NXdata"
        for name, value in metadata.info.items():
            monitor_group.attrs[name] = _bytes(value)
        for name, values in monitor["arrays"].items():
            monitor_group.create_dataset(name, data=np.asarray(values))


def _assert_entry_order_basis(testcase, scan):
    testcase.assertEqual(scan.execution, {})
    testcase.assertEqual(scan.metadata["format"], "nexus")
    testcase.assertEqual(scan.metadata["point_id_basis"], "entry order")
    testcase.assertIn("diagnostics", scan.metadata)
    testcase.assertIn("declared_numpoints", scan.metadata)
    testcase.assertIsNone(scan.plan)


class TestNexusScanLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((FIXTURES / "expected.json").read_text())
        cls.cases = {case["name"]: case for case in manifest["cases"]}

    def test_combined_and_per_point_cases_load_without_subprocesses(self):
        names = ("nexus_1d", "nexus_cartesian_2x3", "nexus_parallel",
                 "nexus_paired_categorical_repeated", "nexus_paired")
        for name in names:
            with self.subTest(case=name):
                case = self.cases[name]
                with patch("subprocess.run", side_effect=AssertionError(
                        "NeXus loading must not execute a subprocess")) as run:
                    scan = load_scan(FIXTURES / name)
                    monitor = scan.dataset("spectrum")
                    data = scan.points[0].data
                run.assert_not_called()
                expected_count = (len(case["observed"]["points"]) +
                                   len(case["observed"]["incomplete_points"]))
                self.assertEqual(len(scan.points), expected_count)
                self.assertEqual(scan.execution, {})
                self.assertEqual(scan.metadata["format"], "nexus")
                self.assertIn(scan.metadata["point_id_basis"],
                              ("entry order", "numbered directories"))
                self.assertIn("diagnostics", scan.metadata)
                self.assertIn("declared_numpoints", scan.metadata)
                if scan.metadata["point_id_basis"] == "entry order":
                    self.assertEqual([point.point_id for point in scan.points],
                                     list(range(expected_count)))
                self.assertIsInstance(data, McStasDataCollection)
                self.assertEqual(data[0].name, "spectrum")
                self.assertEqual(data[0].Intensity.shape, (4,))
                self.assertEqual(len(monitor.total_I), expected_count)
                for point, expected in zip(scan.points, case["observed"]["points"]):
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

    def test_entry_qualified_sources_preserve_combined_and_parallel_identity(self):
        combined = load_scan(FIXTURES / "nexus_cartesian_2x3")
        self.assertEqual([point.source.entry for point in combined.points],
                         ["/entry1", "/entry2", "/entry3", "/entry4",
                          "/entry5", "/entry6"])
        parallel = load_scan(FIXTURES / "nexus_parallel")
        self.assertEqual([point.source.path.parent.name for point in parallel.points],
                         ["0", "1"])
        self.assertEqual([point.source.entry for point in parallel.points],
                         ["/entry1", "/entry1"])
        np.testing.assert_array_equal(
            combined.dataset("spectrum").coordinates["theta"], [-1, 1, -1, 1, -1, 1]
        )
        np.testing.assert_array_equal(
            combined.dataset("spectrum").coordinates["wavelength"], [2, 2, 4, 4, 6, 6]
        )

    def test_nexus_arrays_match_the_equivalent_ascii_captures(self):
        for nexus_name, ascii_name in (
                ("nexus_1d", "ascii_1d"),
                ("nexus_cartesian_2x3", "ascii_cartesian_2x3"),
                ("nexus_paired", "ascii_paired"),
                ("nexus_paired_categorical_repeated", "ascii_categorical_repeated")):
            with self.subTest(case=nexus_name):
                nexus = load_scan(FIXTURES / nexus_name)
                ascii_scan = load_scan(FIXTURES / ascii_name)
                self.assertEqual(len(nexus.points), len(ascii_scan.points))
                for nexus_point, ascii_point in zip(nexus.points, ascii_scan.points):
                    np.testing.assert_allclose(
                        nexus_point.data[0].Intensity, ascii_point.data[0].Intensity
                    )
                    np.testing.assert_allclose(
                        nexus_point.data[0].Error, ascii_point.data[0].Error
                    )
                    np.testing.assert_array_equal(
                        nexus_point.data[0].Ncount, ascii_point.data[0].Ncount
                    )

    def test_repeated_categorical_entries_remain_distinct_and_equal_only_in_arrays(self):
        scan = load_scan(FIXTURES / "nexus_paired_categorical_repeated")
        self.assertEqual(scan.points[0].parameters, scan.points[2].parameters)
        self.assertNotEqual(scan.points[0].point_id, scan.points[2].point_id)
        np.testing.assert_array_equal(scan.points[0].data[0].Intensity,
                                      scan.points[2].data[0].Intensity)

    def test_partial_entry_is_retained_as_missing_data(self):
        scan = load_scan(FIXTURES / "nexus_partial")
        self.assertEqual(scan.execution, {})
        self.assertEqual([point.point_id for point in scan.points], [0, 1])
        point = scan.points[1]
        self.assertEqual(point.parameters["fail"], 1)
        self.assertEqual(point.summaries, {})
        self.assertEqual(point.status, "unknown")
        self.assertEqual(point.data_status, "unknown")
        with self.assertRaises(ScanDataError) as caught:
            _ = point.data
        self.assertEqual(caught.exception.data_status, "missing")
        self.assertEqual(point.data_status, "missing")
        np.testing.assert_allclose(scan.dataset("spectrum").total_I,
                                   [1.25659e-6, np.nan], equal_nan=True)
        self.assertEqual(scan.dataset("spectrum").availability["total_I"].tolist(),
                         [True, None])
        np.testing.assert_array_equal(scan.dataset("spectrum").valid, [True, False])

    def test_mcstas_3_10_partial_keeps_storage_records_without_original_plan_ids(self):
        scan = load_scan(VERSIONED_FIXTURES / "nexus_partial")

        self.assertEqual(scan.execution, {})
        self.assertIsNone(scan.plan)
        self.assertEqual(scan.metadata["format"], "nexus")
        self.assertEqual(scan.metadata["point_id_basis"], "entry order")
        self.assertEqual(scan.metadata["declared_numpoints"], "4")
        self.assertEqual(len(scan.points), 3)
        self.assertEqual([point.point_id for point in scan.points], [0, 1, 2])
        self.assertEqual([point.source.entry for point in scan.points],
                         ["/entry1", "/entry2", "/entry3"])
        np.testing.assert_array_equal(scan.coordinates["theta"], [0, 2, 3])
        self.assertEqual([point.parameters["fail"] for point in scan.points], [0, 0, 0])
        self.assertEqual([point.metadata["seed"] for point in scan.points],
                         [314159, 316207, 317231])
        self.assertTrue(any("original point IDs" in diagnostic
                            for diagnostic in scan.metadata["diagnostics"]))
        self.assertTrue(any("no missing plan IDs inferred" in diagnostic
                            for diagnostic in scan.metadata["diagnostics"]))

        for point in scan.points:
            self.assertEqual(point.status, "unknown")
            self.assertEqual(point.data_status, "unknown")
            self.assertEqual(len(point.summaries), 1)
            self.assertEqual(point.data[0].name, "spectrum")
            self.assertEqual(point.data_status, "available")

    def test_mcstas_3_10_parallel_uses_shifted_seeds_and_entry_order_identity(self):
        scan = load_scan(VERSIONED_FIXTURES / "nexus_parallel")

        self.assertEqual(scan.execution, {})
        self.assertIsNone(scan.plan)
        self.assertEqual(scan.metadata["format"], "nexus")
        self.assertEqual(scan.metadata["point_id_basis"], "entry order")
        self.assertEqual(scan.metadata["declared_numpoints"], "2")
        self.assertEqual([point.point_id for point in scan.points], [0, 1])
        self.assertEqual([point.source.entry for point in scan.points],
                         ["/entry1", "/entry2"])
        np.testing.assert_array_equal(scan.coordinates["theta"], [-1, 1])
        self.assertEqual([point.metadata["seed"] for point in scan.points],
                         [314159, 315183])
        self.assertTrue(any("original point IDs" in diagnostic
                            for diagnostic in scan.metadata["diagnostics"]))

    def test_missing_middle_entries_use_synthetic_combined_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                for number, theta, intensity in ((1, b"0", 1), (3, b"2", 3)):
                    _write_entry(
                        handle,
                        number,
                        parameters={"theta": theta},
                        parameter_types={"theta": "double"},
                        monitors=(
                            _monitor(
                                "spectrum",
                                {
                                    "Wavelength__AA_": [0.0],
                                    "data": [intensity],
                                    "errors": [0.0],
                                    "ncount": [intensity],
                                },
                                values="{} 0 {}".format(intensity, intensity),
                                dimension=1,
                            ),
                        ),
                    )

            scan = load_scan(path)
            self.assertEqual(scan.execution, {})
            self.assertIsNone(scan.plan)
            self.assertEqual(scan.metadata["format"], "nexus")
            self.assertEqual(scan.metadata["point_id_basis"], "entry order")
            self.assertIsNone(scan.metadata["declared_numpoints"])
            self.assertEqual([point.point_id for point in scan.points], [0, 1])
            self.assertEqual([point.source.entry for point in scan.points],
                             ["/entry1", "/entry3"])
            np.testing.assert_array_equal(scan.coordinates["theta"], [0, 2])
            self.assertTrue(any("original point IDs" in diagnostic
                                for diagnostic in scan.metadata["diagnostics"]))
            for point, intensity in zip(scan.points, (1, 3)):
                np.testing.assert_array_equal(point.data[0].Intensity, [intensity])

    def test_combined_file_and_single_run_loader_diagnostic(self):
        scan = load_scan(FIXTURES / "nexus_1d" / "mccode.h5")
        self.assertEqual(len(scan.points), 3)
        with self.assertRaisesRegex(RuntimeError, "load_scan"):
            load_data(FIXTURES / "nexus_1d")

    def test_numeric_entry_ordering_loads_distinct_arrays_after_entry9(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                for number in range(1, 13):
                    _write_entry(
                        handle,
                        number,
                        parameters={"theta": str(number)},
                        parameter_types={"theta": "double"},
                        monitors=(
                            _monitor(
                                "spectrum",
                                {
                                    "Wavelength__AA_": [0.0, 1.0],
                                    "data": [number, number + 0.5],
                                    "errors": [0.0, 0.0],
                                    "ncount": [number, number],
                                },
                                values="{} 0 {}".format(number, number),
                            ),
                        ),
                    )

            scan = load_scan(path)
            self.assertEqual(
                [point.source.entry for point in scan.points],
                ["/entry{}".format(number) for number in range(1, 13)],
            )
            np.testing.assert_array_equal(
                scan.dataset("spectrum").total_I,
                np.arange(1, 13, dtype=float),
            )
            for point, number in zip(scan.points, range(1, 13)):
                np.testing.assert_array_equal(
                    point.data[0].Intensity,
                    [number, number + 0.5],
                )
            _assert_entry_order_basis(self, scan)

    def test_repeated_coordinate_entries_remain_distinct(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                for number, theta in ((1, b"0"), (2, b"1"), (3, b"0")):
                    _write_entry(
                        handle,
                        number,
                        parameters={"theta": theta},
                        parameter_types={"theta": "double"},
                        monitors=(
                            _monitor(
                                "spectrum",
                                {
                                    "Wavelength__AA_": [0.0],
                                    "data": [number],
                                    "errors": [0.0],
                                    "ncount": [number],
                                },
                                values="{} 0 {}".format(number, number),
                                dimension=1,
                            ),
                        ),
                    )

            scan = load_scan(path)
            self.assertEqual(scan.points[0].parameters, scan.points[2].parameters)
            self.assertNotEqual(scan.points[0].point_id, scan.points[2].point_id)
            self.assertNotEqual(scan.points[0].source.entry, scan.points[2].source.entry)
            np.testing.assert_array_equal(scan.points[0].data[0].Intensity, [1])
            np.testing.assert_array_equal(scan.points[2].data[0].Intensity, [3])
            _assert_entry_order_basis(self, scan)

    def test_loading_and_summary_access_are_lazy_and_close_hdf5_handles(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                _write_entry(
                    handle,
                    1,
                    monitors=(
                        _monitor(
                            "spectrum",
                            {
                                "Wavelength__AA_": [0.0, 1.0],
                                "data": [4.0, 5.0],
                                "errors": [0.0, 0.0],
                                "ncount": [2.0, 2.0],
                            },
                            values="9 0 4",
                        ),
                    ),
                )

            opened = []
            detector_reads = []
            real_file = h5py.File
            real_getitem = h5py.Dataset.__getitem__
            real_array = h5py.Dataset.__array__

            def spy_file(*args, **kwargs):
                handle = real_file(*args, **kwargs)
                opened.append(handle)
                return handle

            def spy_getitem(dataset, *args, **kwargs):
                if "/data/" in dataset.name:
                    detector_reads.append((dataset.name, "getitem"))
                return real_getitem(dataset, *args, **kwargs)

            def spy_array(dataset, *args, **kwargs):
                if "/data/" in dataset.name:
                    detector_reads.append((dataset.name, "array"))
                return real_array(dataset, *args, **kwargs)

            with patch("mcstasscript.data.scan_nexus.h5py.File",
                       side_effect=spy_file) as file_spy:
                with patch.object(h5py.Dataset, "__getitem__", spy_getitem):
                    with patch.object(h5py.Dataset, "__array__", spy_array):
                        scan = load_scan(path)
                        summary = scan.dataset("spectrum")
                        _ = summary.total_I
                        _ = summary.total_E
                        _ = summary.total_N
                        _ = summary.availability
                        self.assertEqual(detector_reads, [])
                        self.assertTrue(all(not handle.id.valid for handle in opened))

                        np.testing.assert_array_equal(
                            scan.points[0].data[0].Intensity,
                            [4.0, 5.0],
                        )

            self.assertGreater(file_spy.call_count, 0)
            self.assertTrue(detector_reads)
            self.assertTrue(all(not handle.id.valid for handle in opened))

    def test_byte_parameters_follow_declared_instrument_types(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                _write_entry(
                    handle,
                    1,
                    parameters={
                        "label": b"001",
                        "count": b"7",
                        "gain": b"1.25",
                    },
                    parameter_types={
                        "label": "string",
                        "count": "int",
                        "gain": "double",
                    },
                    monitors=(
                        _monitor(
                            "spectrum",
                            {
                                "Wavelength__AA_": [0.0],
                                "data": [1.0],
                                "errors": [0.0],
                                "ncount": [7.0],
                            },
                            values="1 0 7",
                            dimension=1,
                        ),
                    ),
                )

            point = load_scan(path).points[0]

        self.assertEqual(point.parameters["label"], "001")
        self.assertIs(type(point.parameters["label"]), str)
        self.assertEqual(point.parameters["count"], 7)
        self.assertIs(type(point.parameters["count"]), int)
        self.assertEqual(point.parameters["gain"], 1.25)
        self.assertIs(type(point.parameters["gain"]), float)

    def test_event_monitors_remain_pointwise_and_unbinned(self):
        expected_events = {}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                for number in (1, 2):
                    events = np.asarray(
                        [[number, 0.1], [number + 1, 0.2]],
                        dtype=float,
                    )
                    expected_events[number] = events
                    _write_entry(
                        handle,
                        number,
                        monitors=(
                            _monitor(
                                "events",
                                {"events": events},
                                dimension=(2, 2),
                                values=None,
                                variables="p x",
                            ),
                        ),
                    )

            scan = load_scan(path)
            for point, number in zip(scan.points, (1, 2)):
                dataset = point.data[0]
                self.assertIsInstance(dataset, McStasDataEvent)
                self.assertNotIsInstance(dataset, McStasDataBinned)
                self.assertEqual(dataset.data_type, "Events")
                np.testing.assert_array_equal(dataset.Events, expected_events[number])

    def test_varying_monitor_inventory_preserves_point_and_summary_alignment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                _write_entry(
                    handle,
                    1,
                    monitors=(
                        _monitor(
                            "spectrum",
                            {
                                "Wavelength__AA_": [0.0],
                                "data": [1.0],
                                "errors": [0.0],
                                "ncount": [1.0],
                            },
                            values="1 0 1",
                            dimension=1,
                        ),
                        _monitor(
                            "beam",
                            {
                                "Wavelength__AA_": [0.0],
                                "data": [10.0],
                                "errors": [0.0],
                                "ncount": [10.0],
                            },
                            values="10 0 10",
                            dimension=1,
                        ),
                    ),
                )
                _write_entry(
                    handle,
                    2,
                    monitors=(
                        _monitor(
                            "spectrum",
                            {
                                "Wavelength__AA_": [0.0],
                                "data": [2.0],
                                "errors": [0.0],
                                "ncount": [2.0],
                            },
                            values="2 0 2",
                            dimension=1,
                        ),
                    ),
                )
                _write_entry(
                    handle,
                    3,
                    monitors=(
                        _monitor(
                            "beam",
                            {
                                "Wavelength__AA_": [0.0],
                                "data": [30.0],
                                "errors": [0.0],
                                "ncount": [30.0],
                            },
                            values="30 0 30",
                            dimension=1,
                        ),
                    ),
                )
                _write_entry(handle, 4)

            scan = load_scan(path)
            point_data_names = [
                [dataset.name for dataset in scan.points[index].data]
                for index in range(3)
            ]
            with self.assertRaises(ScanDataError) as caught:
                _ = scan.points[3].data

        self.assertEqual({monitor.component for monitor in scan.monitors},
                         {"spectrum", "beam"})
        np.testing.assert_allclose(
            scan.dataset("spectrum").total_I,
            [1.0, 2.0, np.nan, np.nan],
            equal_nan=True,
        )
        self.assertEqual(
            scan.dataset("spectrum").availability["total_I"].tolist(),
            [True, True, None, None],
        )
        np.testing.assert_allclose(
            scan.dataset("beam").total_I,
            [10.0, np.nan, 30.0, np.nan],
            equal_nan=True,
        )
        self.assertEqual(set(point_data_names[0]), {"spectrum", "beam"})
        self.assertEqual(point_data_names[1], ["spectrum"])
        self.assertEqual(point_data_names[2], ["beam"])
        self.assertEqual(caught.exception.data_status, "missing")
        _assert_entry_order_basis(self, scan)

    def test_missing_monitor_arrays_are_diagnostic_and_known_zero_is_available(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mccode.h5"
            with h5py.File(path, "w") as handle:
                _write_entry(
                    handle,
                    1,
                    monitors=(
                        _monitor(
                            "known_zero",
                            {
                                "Wavelength__AA_": [0.0, 1.0],
                                "data": [0.0, 0.0],
                                "errors": [0.0, 0.0],
                                "ncount": [0.0, 0.0],
                            },
                            values="0 0 0",
                        ),
                        _monitor(
                            "missing_arrays",
                            {
                                "Wavelength__AA_": [0.0, 1.0],
                                "data": [3.0, 4.0],
                            },
                            values=None,
                        ),
                    ),
                )

            scan = load_scan(path)
            known_zero = scan.dataset("known_zero")
            missing = scan.dataset("missing_arrays")
            self.assertEqual(known_zero.total_I.tolist(), [0.0])
            self.assertEqual(known_zero.total_E.tolist(), [0.0])
            self.assertEqual(known_zero.total_N.tolist(), [0.0])
            self.assertEqual(known_zero.availability["total_E"].tolist(), [True])
            self.assertEqual(known_zero.availability["total_N"].tolist(), [True])
            self.assertEqual(missing.availability["total_E"].tolist(), [False])
            self.assertEqual(missing.availability["total_N"].tolist(), [False])

            point = scan.points[0]
            with self.assertWarns(UserWarning):
                data = point.data

        self.assertEqual([dataset.name for dataset in data], ["known_zero"])
        self.assertEqual(point.data_status, "partial")

        available = {
            monitor.component: (monitor, availability)
            for monitor, availability in point.array_availability.items()
        }
        self.assertEqual(
            available["known_zero"][1],
            {"Intensity": True, "Error": True, "Ncount": True},
        )
        self.assertEqual(
            available["missing_arrays"][1],
            {"Intensity": True, "Error": False, "Ncount": False},
        )
        self.assertIn("errors", str(point.diagnostics[available["missing_arrays"][0]]))

    def test_single_entry_load_data_ignores_zero_byte_scan_summary(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "mccode.h5"
            with h5py.File(path, "w") as handle:
                _write_entry(
                    handle,
                    1,
                    monitors=(
                        _monitor(
                            "spectrum",
                            {
                                "Wavelength__AA_": [0.0, 1.0],
                                "data": [2.0, 3.0],
                                "errors": [0.0, 0.0],
                                "ncount": [1.0, 1.0],
                            },
                            values="5 0 2",
                        ),
                    ),
                )
            (directory / "mccode.dat").write_bytes(b"")

            data = load_data(directory)

        self.assertIsInstance(data, McStasDataCollection)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].name, "spectrum")
        np.testing.assert_array_equal(data[0].Intensity, [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
