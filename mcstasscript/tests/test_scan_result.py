from copy import deepcopy
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import warnings

import h5py
import numpy as np

from mcstasscript import ScanPoint, ScanResult
from mcstasscript.data.collection import McStasDataCollection
from mcstasscript.data.data import McStasDataBinned, McStasMetaData
from mcstasscript.data.scan import (
    ScanDataError, ScanMonitor, ScanMonitorSummary, ScanPointData, ScanSource,
)


MONITOR = ScanMonitor("detector", "detector.dat")
TOTALS = ("total_I", "total_E", "total_N")


def make_dataset(monitor=MONITOR):
    metadata = McStasMetaData()
    metadata.component_name = monitor.component
    metadata.filename = monitor.filename
    metadata.dimension = 1
    metadata.total_I, metadata.total_E, metadata.total_N = 2.0, 0.0, 3
    return McStasDataBinned(metadata, np.array([2.0]), np.array([0.0]),
                            np.array([3]), xaxis=np.array([0.0]))


def make_point(point_id=0, parameters=None, summaries=None, **kwargs):
    if summaries is None:
        summaries = [ScanMonitorSummary(MONITOR, total_I=2.0)]
    return ScanPoint(point_id, {"theta": 1.0} if parameters is None else parameters,
                     summaries=summaries, **kwargs)


class TestScanRecords(unittest.TestCase):
    def test_relative_source_is_anchored_before_lazy_access(self):
        original = Path.cwd()
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            try:
                os.chdir(first)
                source = ScanSource(Path("points") / ".." / "point_7")
                expected = Path(first).resolve() / "point_7"
                loader = Mock(return_value=ScanPointData([]))
                point = make_point(source=source, loader=loader)
                os.chdir(second)
                self.assertEqual(source.path, expected)
                self.assertEqual(str(source), str(expected))
                self.assertEqual(point.data, [])
                loader.assert_called_once_with(source)
                self.assertEqual(loader.call_args.args[0].path, expected)
            finally:
                os.chdir(original)

    def test_explicit_nexus_entries_are_normalized_not_point_ids(self):
        for entry in ("entry_42", "/entry_42/", "entry_42/instrument"):
            with self.subTest(entry=entry):
                source = ScanSource("absent.h5", "nexus", entry)
                self.assertEqual(source.entry, "/" + entry.strip("/"))
                self.assertEqual(str(source), str(source.path) + "::" + source.entry)
                self.assertEqual(make_point("measurement-A", source=source).point_id,
                                 "measurement-A")

    def test_invalid_source_locators(self):
        cases = [dict(format="other"), dict(entry="entry"),
                 *[dict(format="nexus", entry=entry) for entry in
                    (None, "", "/", "///", 1, "a//b", "a/./b", "a/../b", ".", "..",
                     "entry1\x00other")]]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ScanSource("unused", **kwargs)
        for format, entry in (("ascii", None), ("nexus", "entry1")):
            with self.subTest(format=format), self.assertRaises(ValueError):
                ScanSource("file.h5\x00other", format, entry)

    def test_monitor_identity_validation(self):
        self.assertEqual(MONITOR, ScanMonitor("detector", "detector.dat"))
        self.assertEqual(hash(MONITOR), hash(ScanMonitor("detector", "detector.dat")))
        self.assertNotEqual(MONITOR, ScanMonitor("detector", "other.dat"))
        self.assertNotEqual(MONITOR, ScanMonitor("detector", "detector.dat", "second"))
        self.assertIsNone(ScanMonitor("summary_only").filename)
        for component in (None, "", 7):
            with self.subTest(component=component), self.assertRaises(ValueError):
                ScanMonitor(component)
        for kwargs in ({"filename": 7}, {"discriminator": 7}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                ScanMonitor("detector", **kwargs)

    def test_summary_copies_availability_provenance_and_metadata_separately(self):
        availability = {"total_I": True, "total_E": False}
        nested = {"origin": ["native summary"]}
        summary = ScanMonitorSummary(MONITOR, total_I=0, availability=availability,
                                     provenance=nested, metadata=nested)
        availability["total_I"] = False
        nested["origin"].append("changed")
        self.assertEqual(summary.availability,
                         {"total_I": True, "total_E": False, "total_N": None})
        self.assertEqual(summary.provenance, {"origin": ["native summary"]})
        self.assertEqual(summary.metadata, {"origin": ["native summary"]})
        summary.metadata["origin"].append("display metadata")
        self.assertEqual(summary.provenance, {"origin": ["native summary"]})
        summary.metadata["availability"] = {"total_I": False}
        self.assertIs(summary.availability["total_I"], True)

    def test_summary_rejects_invalid_totals_and_conflicting_availability(self):
        for quantity in TOTALS:
            for value in (True, "0", [0], 1j):
                with self.subTest(quantity=quantity, value=value):
                    with self.assertRaises(TypeError):
                        ScanMonitorSummary(MONITOR, **{quantity: value})
            for value, state in ((0, False), (0, None), (None, True)):
                with self.subTest(quantity=quantity, value=value, state=state):
                    with self.assertRaises(ValueError):
                        ScanMonitorSummary(MONITOR, **{quantity: value},
                                           availability={quantity: state})
        with self.assertRaises(TypeError):
            ScanMonitorSummary("detector")
        with self.assertRaises(ValueError):
            ScanMonitorSummary(MONITOR, availability={"Error": False})
        with self.assertRaises(TypeError):
            ScanMonitorSummary(MONITOR, availability={"total_I": 1})

    def test_validated_availability_is_read_only_and_copyable(self):
        summary = ScanMonitorSummary(MONITOR, total_I=0, availability={"total_E": False},
                                     metadata={"notes": ["original"]})
        points = [make_point(i, summaries=[summary]) for i in range(2)]
        with self.assertRaises(TypeError):
            summary.availability["total_I"] = False
        with self.assertRaises(AttributeError):
            summary.availability.clear()
        np.testing.assert_array_equal(ScanResult(points).dataset("detector").valid, [True, True])
        for restored in (deepcopy(summary), pickle.loads(pickle.dumps(summary))):
            self.assertEqual(restored, summary)
            self.assertIsNot(restored, summary)
            with self.assertRaises(TypeError):
                restored.availability["total_I"] = False
            restored.metadata["notes"].append("copy only")
            self.assertEqual(summary.metadata, {"notes": ["original"]})

    def test_point_data_converts_lists_and_detaches_sidecars(self):
        dataset = make_dataset()
        diagnostics = {MONITOR: ["missing Error"]}
        availability = {MONITOR: {"Intensity": True, "Error": False}}
        report = ScanPointData([dataset], diagnostics, availability)
        self.assertIsInstance(report.data, McStasDataCollection)
        self.assertIs(report.data[0], dataset)
        diagnostics[MONITOR].append("changed")
        availability[MONITOR]["Error"] = True
        self.assertEqual(report.diagnostics[MONITOR], ["missing Error"])
        self.assertIs(report.availability[MONITOR]["Error"], False)
        collection = McStasDataCollection([dataset])
        self.assertIs(ScanPointData(collection).data, collection)
        for kwargs in ({"data": ()}, {"data": [], "diagnostics": {"detector": "bad"}},
                       {"data": [], "availability": {"detector": {}}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                ScanPointData(**kwargs)


class TestScanResult(unittest.TestCase):
    def test_repeated_coordinates_have_distinct_stable_read_only_ids(self):
        first, second = make_point(7), make_point("repeat")
        scan = ScanResult([first, second])
        self.assertEqual(scan.points, (first, second))
        self.assertEqual(first.parameters, second.parameters)
        np.testing.assert_array_equal(scan.dataset("detector").point_ids,
                                      np.array([7, "repeat"], dtype=object))
        with self.assertRaises(AttributeError):
            first.point_id = 99
        first.parameters["theta"] = 9
        first.clear_cache()
        self.assertEqual(first.point_id, 7)
        with self.assertRaises(ValueError):
            ScanResult([first, make_point(7, {"theta": 100})])

    def test_point_and_coordinate_order_never_infers_grid_or_sorts_ids(self):
        parameters = [{"x": 1, "y": "b"}, {"x": 0, "y": "a"},
                      {"x": 1, "y": "a"}, {"x": 0, "y": "b"},
                      {"x": 1, "y": "b"}, {"x": 4, "mode": "extra"}]
        points = [make_point(point_id, coords) for point_id, coords in
                  zip((9, "a", 2, 17, "again", 8), parameters)]
        scan = ScanResult(iter(points))
        self.assertEqual(scan.points, tuple(points))
        self.assertEqual(scan.parameter_names, ("x", "y", "mode"))
        self.assertIsNone(scan.plan)
        self.assertEqual(scan.execution, {})
        np.testing.assert_array_equal(scan.coordinates["x"], [1, 0, 1, 0, 1, 4])
        np.testing.assert_array_equal(scan.coordinates["y"], ["b", "a", "a", "b", "b", None])
        np.testing.assert_array_equal(scan.coordinates["mode"], [None] * 5 + ["extra"])
        self.assertEqual(scan.coordinates["x"].dtype, np.dtype(object))
        scan.coordinates["x"][0] = 999
        self.assertEqual(points[0].parameters["x"], 1)
        self.assertEqual(ScanResult(points, parameter_names=["y", "x"]).parameter_names,
                         ("y", "x"))

    def test_empty_scan_and_explicit_plan(self):
        scan = ScanResult([], parameter_names=["theta"])
        self.assertEqual(scan.points, ())
        self.assertEqual(scan.monitors, ())
        self.assertEqual(scan.coordinates["theta"].size, 0)
        self.assertIsNone(scan.plan)
        plan = {"axes": ["theta"], "kind": "explicit"}
        self.assertEqual(ScanResult([], plan=plan).plan, plan)
        with self.assertRaises(NameError):
            scan.dataset("detector")

    def test_parameters_metadata_and_execution_inputs_are_detached(self):
        parameters = {"theta": 1, "nested": [2]}
        metadata = {"source": ["original"]}
        execution = {"status": "failed", "details": ["exit 1"]}
        point = make_point(parameters=parameters, metadata=metadata)
        scan = ScanResult([point], metadata=metadata, execution=execution)
        parameters["nested"].append(3)
        metadata["source"].append("mutated")
        execution["details"].clear()
        self.assertEqual(point.parameters["nested"], [2])
        self.assertEqual(point.metadata, {"source": ["original"]})
        self.assertEqual(scan.metadata, {"source": ["original"]})
        self.assertEqual(scan.execution["details"], ["exit 1"])

    def test_invalid_points_and_parameter_names(self):
        for point_id, error in ((True, TypeError), (1.5, TypeError), (None, TypeError),
                                (-1, ValueError), ("", ValueError)):
            with self.subTest(point_id=point_id), self.assertRaises(error):
                make_point(point_id)
        for kwargs in ({"source": "path"}, {"loader": 1}, {"summaries": [MONITOR]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                make_point(**kwargs)
        for kwargs in ({"status": "completed"}, {"data_status": "failed"},
                       {"summaries": [ScanMonitorSummary(MONITOR)] * 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_point(**kwargs)
        with self.assertRaises(TypeError):
            ScanResult([object()])
        for names, error in (("theta", TypeError), ([1], TypeError),
                             (["theta", "theta"], ValueError)):
            with self.subTest(names=names), self.assertRaises(error):
                ScanResult([], parameter_names=names)

    def test_monitor_order_and_scan_wide_precedence_prevent_pointwise_fallback(self):
        by_component = ScanMonitor("target", "other.dat")
        by_filename = ScanMonitor("other", "target")
        points = [make_point(0, summaries=[ScanMonitorSummary(by_component, total_I=3)]),
                  make_point(1, summaries=[ScanMonitorSummary(by_filename, total_I=99)]),
                  make_point(2, summaries=[ScanMonitorSummary(by_component, total_I=5)])]
        scan = ScanResult(points)
        self.assertEqual(scan.monitors, (by_component, by_filename))
        view = scan.dataset("target")
        self.assertEqual(view.monitor, by_component)
        np.testing.assert_allclose(view.total_I, [3, np.nan, 5], equal_nan=True)
        np.testing.assert_array_equal(view.availability["total_I"], [True, None, True])
        self.assertEqual(scan.dataset(filename="target").monitor, by_filename)
        self.assertEqual(scan.dataset("other.dat").monitor, by_component)
        self.assertEqual(scan.dataset(component="target", filename="other.dat").monitor,
                         by_component)
        for kwargs in ({"component": "missing"}, {"filename": "missing"},
                       {"component": "target", "filename": "target"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NameError):
                scan.dataset(**kwargs)

    def test_ambiguous_outputs_require_exact_identity_or_filename(self):
        monitors = [ScanMonitor("detector", "a.dat"), ScanMonitor("detector", "b.dat")]
        scan = ScanResult([make_point(summaries=[ScanMonitorSummary(m) for m in monitors])])
        for kwargs in ({"name": "detector"}, {"component": "detector"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scan.dataset(**kwargs)
        self.assertEqual(scan.dataset(component="detector", filename="b.dat").monitor,
                         monitors[1])
        self.assertEqual(scan.dataset(monitors[0]).monitor, monitors[0])

    def test_duplicate_filenames_and_discriminators_are_not_collapsed(self):
        monitors = [ScanMonitor("detector", "same.dat", "first"),
                    ScanMonitor("detector", "same.dat", "second")]
        scan = ScanResult([make_point(summaries=[
            ScanMonitorSummary(monitor, total_I=value)
            for monitor, value in zip(monitors, (3, 8))])])
        self.assertEqual(scan.monitors, tuple(monitors))
        for kwargs in ({"name": "detector"}, {"filename": "same.dat"},
                       {"component": "detector", "filename": "same.dat"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scan.dataset(**kwargs)
        for monitor, value in zip(monitors, (3, 8)):
            with self.subTest(monitor=monitor):
                view = scan.dataset(ScanMonitor(monitor.component, monitor.filename,
                                                monitor.discriminator))
                self.assertEqual(view.monitor, monitor)
                np.testing.assert_array_equal(view.total_I, [value])
        with self.assertRaises(NameError):
            scan.dataset(ScanMonitor("detector", "same.dat", "absent"))

    def test_invalid_selectors(self):
        scan = ScanResult([make_point()])
        for kwargs in ({}, {"name": 1}, {"component": 1}, {"filename": 1},
                       {"name": "detector", "component": "detector"},
                       {"name": MONITOR, "filename": "detector.dat"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                scan.dataset(**kwargs)

    def test_summary_access_is_io_free_and_preserves_zero_missing_unknown(self):
        summaries = [ScanMonitorSummary(MONITOR, total_I=0, total_E=0, total_N=0),
                     ScanMonitorSummary(MONITOR, availability=dict.fromkeys(TOTALS, False)),
                     ScanMonitorSummary(MONITOR), None,
                     ScanMonitorSummary(MONITOR, total_I=np.inf, total_E=np.nan, total_N=7)]
        loader = Mock(side_effect=AssertionError("Summary access must not load arrays"))
        points = [make_point(i, summaries=[] if summary is None else [summary], loader=loader)
                  for i, summary in enumerate(summaries)]
        scan = ScanResult(points)
        view = scan.dataset("detector")
        np.testing.assert_array_equal(view.point_ids, range(5))
        np.testing.assert_array_equal(view.coordinates["theta"], [1.0] * 5)
        for quantity, last, last_valid in (("total_I", np.inf, False),
                                          ("total_E", np.nan, False), ("total_N", 7, True)):
            with self.subTest(quantity=quantity):
                np.testing.assert_allclose(getattr(view, quantity),
                                           [0, np.nan, np.nan, np.nan, last], equal_nan=True)
                np.testing.assert_array_equal(view.availability[quantity],
                                              [True, False, None, None, True])
                np.testing.assert_array_equal(view.validity(quantity),
                                              [True, False, False, False, last_valid])
        np.testing.assert_array_equal(view.valid, [True, False, False, False, False])
        with self.assertRaises(ValueError):
            view.validity("Intensity")
        view.total_I[0] = 100
        view.availability["total_I"][0] = False
        self.assertEqual(view.total_I[0], 0)
        self.assertIs(view.availability["total_I"][0], True)
        for point in points:
            self.assertFalse(point.is_loaded)
            self.assertEqual(point.array_availability, {})
            self.assertEqual(point.data_status, "unknown")
        loader.assert_not_called()

    def test_mutable_dataset_names_do_not_change_summary_identity(self):
        dataset = make_dataset()
        point = make_point(loader=Mock(return_value=ScanPointData([dataset])))
        scan = ScanResult([point])
        view = scan.dataset("detector")
        point.data[0].name = "renamed for display"
        point.data[0].metadata.component_name = "also renamed"
        point.data[0].metadata.filename = "renamed.dat"
        point.data[0].metadata.total_I = 999
        self.assertEqual(scan.monitors, (MONITOR,))
        self.assertEqual(view.monitor, MONITOR)
        self.assertEqual(scan.dataset("detector.dat").monitor, MONITOR)
        np.testing.assert_array_equal(view.total_I, [2.0])
        with self.assertRaises(NameError):
            scan.dataset("renamed for display")


class TestScanPointLoading(unittest.TestCase):
    def test_full_and_empty_results_load_lazily_and_cache_same_collection(self):
        for datasets in ([make_dataset()], []):
            with self.subTest(empty=not datasets):
                report = ScanPointData(datasets)
                loader = Mock(return_value=report)
                source = ScanSource("point")
                point = make_point(source=source, loader=loader)
                self.assertFalse(point.is_loaded)
                loader.assert_not_called()
                self.assertIs(point.data, report.data)
                self.assertIs(point.data, report.data)
                self.assertTrue(point.is_loaded)
                self.assertEqual(point.data_status, "available")
                self.assertEqual(point.status, "unknown")
                loader.assert_called_once_with(source)

    def test_partial_load_warns_once_and_caches_per_monitor_diagnostics(self):
        missing = ScanMonitor("missing", "missing.dat")
        report = ScanPointData([make_dataset()], {missing: "bad monitor header"},
                               {MONITOR: {"Error": False}, missing: {"Intensity": False}})
        loader = Mock(return_value=report)
        source = ScanSource("partial")
        point = make_point("partial-id", source=source, status="succeeded", loader=loader)
        with self.assertWarns(UserWarning) as caught:
            data = point.data
        self.assertIn("partial-id", str(caught.warning))
        self.assertIn(str(source), str(caught.warning))
        self.assertIn("bad monitor header", str(caught.warning))
        self.assertEqual(point.data_status, "partial")
        self.assertEqual(point.status, "succeeded")
        self.assertEqual(point.diagnostics, {missing: "bad monitor header"})
        self.assertIs(point.array_availability[MONITOR]["Error"], False)
        report.availability[MONITOR]["Error"] = True
        self.assertIs(point.array_availability[MONITOR]["Error"], False)
        with warnings.catch_warnings(record=True) as repeated:
            warnings.simplefilter("always")
            self.assertIs(point.data, data)
        self.assertEqual(repeated, [])
        loader.assert_called_once_with(source)

    def test_totals_and_legacy_zero_arrays_do_not_infer_array_availability(self):
        dataset = make_dataset()
        summary = ScanMonitorSummary(MONITOR, total_I=2, total_E=0, total_N=3)
        point = make_point(summaries=[summary], data=ScanPointData(
            [dataset], availability={MONITOR: {"Intensity": True, "Error": False,
                                              "Ncount": None}}))
        np.testing.assert_array_equal(point.data[0].Error, [0])
        self.assertTrue(summary.availability["total_E"])
        self.assertEqual(point.array_availability[MONITOR],
                         {"Intensity": True, "Error": False, "Ncount": None})
        self.assertEqual(make_point(data=[make_dataset()]).array_availability, {})
        self.assertNotIn("availability", dataset.metadata.info)
        self.assertEqual(dataset.metadata.total_E, 0)

    def test_entirely_unreadable_report_is_a_cached_failure_not_empty_success(self):
        report = ScanPointData([], {MONITOR: "cannot parse detector"},
                               {MONITOR: {"Intensity": False}})
        loader = Mock(return_value=report)
        source = ScanSource("unreadable")
        point = make_point(source=source, loader=loader)
        messages = []
        for _ in range(2):
            with self.assertRaises(ScanDataError) as caught:
                _ = point.data
            messages.append(str(caught.exception))
            self.assertEqual(caught.exception.data_status, "unreadable")
        self.assertEqual(messages[0], messages[1])
        self.assertIn(str(source), messages[0])
        self.assertIn("cannot parse detector", messages[0])
        self.assertFalse(point.is_loaded)
        self.assertEqual(point.data_status, "unreadable")
        self.assertEqual(point.diagnostics, report.diagnostics)
        self.assertEqual(point.array_availability, report.availability)
        np.testing.assert_array_equal(ScanResult([point]).dataset("detector").total_I, [2])
        loader.assert_called_once_with(source)

    def test_eager_unreadable_report_preserves_point_and_summaries(self):
        report = ScanPointData([], {MONITOR: "cannot parse detector"})
        source = ScanSource("unreadable")
        loader = Mock(return_value=ScanPointData([make_dataset()]))
        point = make_point("eager-failure", source=source, status="failed", data=report,
                           loader=loader)
        self.assertEqual(point.data_status, "unreadable")
        self.assertEqual(point.diagnostics, report.diagnostics)
        np.testing.assert_array_equal(ScanResult([point]).dataset("detector").total_I, [2])
        for _ in range(2):
            with self.assertRaises(ScanDataError) as caught:
                _ = point.data
            self.assertIn("eager-failure", str(caught.exception))
            self.assertIn(str(source), str(caught.exception))
            self.assertEqual(point.status, "failed")
        loader.assert_not_called()
        point.clear_cache()
        self.assertIsInstance(point.data, McStasDataCollection)
        self.assertEqual(point.status, "failed")
        loader.assert_called_once_with(source)

    def test_expected_errors_are_source_wrapped_and_cached_without_retry(self):
        errors = [(FileNotFoundError("gone"), "missing"), (OSError("bad file"), "unreadable"),
                  *[(ScanDataError("adapter failure", state), state)
                    for state in ("unknown", "missing", "unreadable")]]
        for error, state in errors:
            with self.subTest(error=error, state=state):
                source = ScanSource("failed.h5", "nexus", "/entry_3")
                loader = Mock(side_effect=error)
                point = make_point("failed-id", source=source, loader=loader)
                messages = []
                for attempt in range(2):
                    with self.assertRaises(ScanDataError) as caught:
                        _ = point.data
                    if attempt == 0:
                        self.assertIs(caught.exception.__cause__, error)
                    self.assertEqual(caught.exception.data_status, state)
                    messages.append(str(caught.exception))
                self.assertEqual(messages[0], messages[1])
                self.assertIn("failed-id", messages[0])
                self.assertIn(str(source), messages[0])
                self.assertIn(str(error), messages[0])
                self.assertEqual(point.data_status, state)
                self.assertFalse(point.is_loaded)
                self.assertEqual(point.status, "unknown")
                loader.assert_called_once_with(source)

    def test_absent_loader_has_no_implicit_retry(self):
        for source, initial, expected in ((None, "unknown", "missing"),
                                          (ScanSource("absent"), "unknown", "unknown"),
                                          (ScanSource("absent"), "missing", "missing")):
            with self.subTest(source=source, initial=initial):
                point = make_point(source=source, data_status=initial)
                for _ in range(2):
                    with self.assertRaises(ScanDataError) as caught:
                        _ = point.data
                    self.assertEqual(caught.exception.data_status, expected)
                self.assertFalse(point.is_loaded)

    def test_clear_cache_explicitly_reloads_success_partial_empty_and_failure(self):
        for initial in (ScanPointData([make_dataset()]), ScanPointData([]),
                        ScanPointData([make_dataset()], {MONITOR: "incomplete"}),
                        ScanDataError("temporarily unreadable")):
            with self.subTest(initial=type(initial).__name__):
                replacement = ScanPointData([make_dataset()])
                loader = Mock(side_effect=[initial, replacement])
                point = make_point(status="failed", loader=loader)
                summaries = point.summaries.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if isinstance(initial, ScanDataError):
                        with self.assertRaises(ScanDataError):
                            _ = point.data
                    else:
                        _ = point.data
                point.clear_cache()
                self.assertFalse(point.is_loaded)
                self.assertEqual(point.data_status, "unknown")
                self.assertEqual(point.diagnostics, {})
                self.assertEqual(point.array_availability, {})
                self.assertEqual(point.status, "failed")
                self.assertEqual(point.summaries, summaries)
                self.assertEqual(loader.call_count, 1)
                self.assertIs(point.data, replacement.data)
                self.assertIs(point.data, replacement.data)
                self.assertEqual(loader.call_count, 2)

    def test_execution_status_is_independent_of_aggregate_and_readability(self):
        for status in ("unknown", "pending", "running", "succeeded", "failed",
                       "interrupted", "skipped"):
            for readable in (False, True):
                with self.subTest(status=status, readable=readable):
                    loader = (Mock(return_value=ScanPointData([make_dataset()])) if readable
                              else Mock(side_effect=ScanDataError("unreadable")))
                    point = make_point(status=status, loader=loader)
                    execution = {"status": "succeeded" if status == "failed" else "failed",
                                 "returncode": 0 if status == "failed" else 1}
                    scan = ScanResult([point], execution=execution)
                    if readable:
                        _ = point.data
                    else:
                        with self.assertRaises(ScanDataError):
                            _ = point.data
                    self.assertEqual(point.status, status)
                    self.assertEqual(scan.execution, execution)
                    point.clear_cache()
                    self.assertEqual(point.status, status)

    def test_programming_errors_propagate_without_becoming_cached_data_errors(self):
        for error in (ValueError("bug"), TypeError("bug"), RuntimeError("bug"), KeyError("bug")):
            with self.subTest(error=type(error).__name__):
                loader = Mock(side_effect=error)
                point = make_point(loader=loader)
                for _ in range(2):
                    with self.assertRaises(type(error)) as caught:
                        _ = point.data
                    self.assertIs(caught.exception, error)
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(point.data_status, "unknown")
                self.assertFalse(point.is_loaded)

    def test_invalid_callback_result_is_a_programming_error(self):
        loader = Mock(return_value=[])
        point = make_point(loader=loader)
        with self.assertRaises(TypeError):
            _ = point.data
        self.assertFalse(point.is_loaded)
        self.assertEqual(point.data_status, "unknown")


class TestScanHDF5Callbacks(unittest.TestCase):
    def test_explicit_entry_load_returns_detached_arrays_and_closes_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("entry_1/intensity", data=[3.0])
                handle.create_dataset("entry_9/intensity", data=[9.0])
            handles = []

            def read_point(source):
                with h5py.File(source.path, "r") as handle:
                    handles.append(handle)
                    dataset = make_dataset()
                    dataset.Intensity = np.array(handle[source.entry + "/intensity"])
                return ScanPointData([dataset])

            loader = Mock(side_effect=read_point)
            source = ScanSource(path, "nexus", "entry_9")
            point = make_point("not-an-entry-number", source=source, loader=loader)
            self.assertEqual(handles, [])
            dataset = point.data[0]
            self.assertFalse(handles[0].id.valid)
            self.assertIsInstance(dataset.Intensity, np.ndarray)
            np.testing.assert_array_equal(dataset.Intensity, [9])
            self.assertIs(point.data[0], dataset)
            loader.assert_called_once_with(source)
            path.unlink()
            np.testing.assert_array_equal(dataset.Intensity, [9])

    def test_malformed_hdf5_adapter_error_wraps_source_and_closes_handle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("entry_2/intensity", data=[b"not numeric"])
            handles = []

            def read_point(source):
                try:
                    with h5py.File(source.path, "r") as handle:
                        handles.append(handle)
                        np.asarray(handle[source.entry + "/intensity"][()], dtype=float)
                except ValueError as error:
                    raise ScanDataError("Malformed intensity array") from error
                raise AssertionError("Malformed test data unexpectedly parsed")

            loader = Mock(side_effect=read_point)
            source = ScanSource(path, "nexus", "/entry_2")
            point = make_point("bad-point", source=source, loader=loader)
            for _ in range(2):
                with self.assertRaises(ScanDataError) as caught:
                    _ = point.data
                self.assertIn(str(source), str(caught.exception))
                self.assertIn("bad-point", str(caught.exception))
                self.assertIn("Malformed intensity array", str(caught.exception))
            self.assertEqual(point.data_status, "unreadable")
            self.assertFalse(point.is_loaded)
            self.assertEqual(len(handles), 1)
            self.assertFalse(handles[0].id.valid)
            loader.assert_called_once_with(source)


if __name__ == "__main__":
    unittest.main()
