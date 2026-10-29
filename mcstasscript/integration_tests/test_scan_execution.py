"""Real McStas 3.10 native scan execution tests."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from mcstasscript import McStas_instr, ScanPlan, ScanExecutionError, load_scan
from mcstasscript.data.scan import ScanDataError
from mcstasscript.helper.scan_command import native_index


def _runner_version():
    result = subprocess.run(
        ["mcrun", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    for token in (result.stdout or "").split():
        parts = token.split(".")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)
    return None


def _make_instrument(workdir):
    instrument = McStas_instr("phase7_tiny", input_path=str(workdir))
    instrument.add_parameter("theta", value=0.0)
    instrument.add_parameter("wavelength", value=4.0)
    instrument.add_parameter("string", "category", value='"vanadium"')
    instrument.add_parameter("int", "fail", value=0)
    instrument.append_declare("double source_flux;")
    instrument.append_initialize(
        'if (fail) { fprintf(stderr, "PHASE7_INTENTIONAL_FAILURE\\n"); exit(23); }'
    )
    instrument.append_initialize(
        'source_flux = strcmp(category, "nickel") == 0 ? 2e10 : 1e10;'
    )

    origin = instrument.add_component("origin", "Progress_bar")
    origin.set_AT([0, 0, 0], RELATIVE="ABSOLUTE")
    source = instrument.add_component("source", "Source_simple")
    source.radius = 0.001
    source.dist = 1
    source.focus_xw = 0.01
    source.focus_yh = 0.01
    source.lambda0 = "wavelength"
    source.dlambda = 0.1
    source.flux = "source_flux"
    source.set_AT([0, 0, 0], RELATIVE="origin")
    monitor = instrument.add_component("spectrum", "L_monitor")
    monitor.xwidth = 0.02
    monitor.yheight = 0.02
    monitor.nL = 4
    monitor.Lmin = 1
    monitor.Lmax = 7
    monitor.filename = '"spectrum.dat"'
    monitor.set_AT([0, 0, 1], RELATIVE="origin")
    return instrument


class TestRealScanExecution(unittest.TestCase):
    """Exercise the native scan owner against actual mcrun output."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("mcrun") is None:
            raise unittest.SkipTest("mcrun is not available")
        if _runner_version() != (3, 10, 0):
            raise unittest.SkipTest("Phase 7 requires McStas 3.10.0")

    def _run(self, workdir, output, plan, *, nexus=False, on_error="raise"):
        instrument = _make_instrument(workdir)
        instrument.settings(
            output_path=str(output),
            ncount=64,
            seed=314159,
            NeXus=nexus,
        )
        snapshot = {
            "parameters": instrument.parameters,
            "parameter_items": {
                name: (parameter, copy.deepcopy(parameter.value), parameter.type)
                for name, parameter in instrument.parameters.parameters.items()
            },
            "run_settings": instrument._run_settings,
            "run_settings_value": copy.deepcopy(instrument._run_settings),
            "output": instrument.output,
            "output_keys": instrument.output_keys,
            "output_path": instrument.output_path,
        }
        execution_error = None
        try:
            result = instrument.scan(plan, on_error=on_error)
        except ScanExecutionError as error:
            result = error.result
            self.assertTrue((output / "scan_manifest.json").is_file())
            execution_error = error
        finally:
            self._assert_unchanged(instrument, snapshot)
        if execution_error is not None and on_error == "raise":
            raise execution_error
        return result

    def _assert_unchanged(self, instrument, snapshot):
        self.assertIs(instrument.parameters, snapshot["parameters"])
        self.assertIs(instrument._run_settings, snapshot["run_settings"])
        self.assertIs(instrument.output, snapshot["output"])
        self.assertIs(instrument.output_keys, snapshot["output_keys"])
        self.assertEqual(instrument.output_path, snapshot["output_path"])
        self.assertEqual(instrument._run_settings, snapshot["run_settings_value"])
        for name, (parameter, value, parameter_type) in snapshot["parameter_items"].items():
            current = instrument.parameters.parameters[name]
            self.assertIs(current, parameter)
            self.assertEqual(current.value, value)
            self.assertEqual(current.type, parameter_type)

    def _assert_records(self, result, rows, *, require_data=True):
        self.assertEqual(result.plan.point_count, len(rows))
        self.assertEqual([point.point_id for point in result.points], list(range(len(rows))))
        for point, expected in zip(result.points, rows):
            self.assertEqual(dict(point.parameters), expected)
            self.assertEqual(point.status, "unknown")
            if require_data:
                expected_seed = 314159 + 1024 * native_index(result.plan, point.point_id)
                # Native grid indices use the first axis as the fastest-changing axis.
                self.assertEqual(point.metadata["seed"]["intended"], expected_seed)
                self.assertEqual(point.metadata["seed"]["actual"],
                                 expected_seed)
                data = point.data
                self.assertEqual(len(data), 1)
                self.assertTrue(np.all(np.isfinite(data[0].Intensity)))
                self.assertTrue(np.any(data[0].Intensity != 0))
                self.assertEqual(int(data[0].Ncount.sum()), 64)
                summary = next(iter(point.summaries.values()))
                self.assertIsNotNone(summary.total_I)
                self.assertGreater(summary.total_I, 0)

    def _assert_manifest(self, output, result):
        document = json.loads((output / "scan_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(document["execution"], result.execution)
        self.assertEqual(len(document["points"]), result.execution["recorded"])
        reloaded = load_scan(output)
        self.assertEqual(reloaded.plan, result.plan)
        self.assertEqual(reloaded.execution, result.execution)
        self.assertEqual(
            [(point.point_id, dict(point.parameters), point.metadata["seed"])
             for point in reloaded.points],
            [(point.point_id, dict(point.parameters), point.metadata["seed"])
             for point in result.points],
        )
        return reloaded

    def _assert_subprocess_reload(self, output, result):
        code = (
            "import json, sys; from mcstasscript import load_scan; "
            "scan=load_scan(sys.argv[1]); "
            "print(json.dumps({'ids':[p.point_id for p in scan.points], "
            "'parameters':[dict(p.parameters) for p in scan.points], "
            "'seeds':[p.metadata['seed'] for p in scan.points], "
            "'execution':scan.execution}, sort_keys=True))"
        )
        environment = dict(os.environ)
        repository = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = repository + ":" + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, "-c", code, str(output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env=environment,
        )
        detached = json.loads(completed.stdout)
        self.assertEqual(detached["ids"], [point.point_id for point in result.points])
        self.assertEqual(detached["parameters"], [dict(point.parameters) for point in result.points])
        self.assertEqual(detached["seeds"], [point.metadata["seed"] for point in result.points])
        self.assertEqual(detached["execution"], result.execution)

    def _assert_native_once(self, output):
        log = (output / "execution.log").read_text(encoding="utf-8")
        self.assertEqual(log.count("INFO: Using directory:"), 1)
        self.assertEqual(log.count("INFO: Recompiling:"), 1)
        self.assertNotIn("recovery", log.lower())
        self.assertFalse((output / "nexus_recovery").exists())

    def _assert_nexus_file(self, output, *, valid):
        path = output / "native" / "mccode.h5"
        self.assertTrue(path.is_file())
        if valid:
            with h5py.File(path, "r") as handle:
                self.assertTrue(handle)
        else:
            with self.assertRaises(OSError):
                h5py.File(path, "r")

    def test_grid_2x3_ascii_and_nexus(self):
        plan = ScanPlan.grid(theta=[-1, 1], wavelength=[2, 4, 6])
        expected = [
            {"theta": theta, "wavelength": wavelength}
            for theta in (-1, 1)
            for wavelength in (2, 4, 6)
        ]
        with TemporaryDirectory(prefix="mcstasscript-phase7-grid-") as directory:
            root = Path(directory)
            for format_name, nexus in (("ascii", False), ("nexus", True)):
                with self.subTest(format=format_name):
                    output = root / format_name
                    result = self._run(root, output, plan, nexus=nexus)
                    self._assert_records(result, expected)
                    self._assert_manifest(output, result)
                    if not nexus:
                        self._assert_subprocess_reload(output, result)
                        moved = root / "moved"
                        shutil.copytree(output, moved)
                        moved_result = load_scan(moved)
                        self.assertTrue(all(str(point.source.path).startswith(str(moved))
                                            for point in moved_result.points))
                        for point in moved_result.points:
                            self.assertTrue(np.any(point.data[0].Intensity != 0))

    def test_paired_points_ascii_and_nexus(self):
        paired = ScanPlan.paired(theta=[-1, 0, 1], wavelength=[2, 4, 6])
        with TemporaryDirectory(prefix="mcstasscript-phase7-paired-") as directory:
            root = Path(directory)
            paired_rows = [
                {"theta": theta, "wavelength": wavelength}
                for theta, wavelength in zip((-1, 0, 1), (2, 4, 6))
            ]
            for format_name, nexus in (("ascii", False), ("nexus", True)):
                with self.subTest(format=format_name):
                    paired_result = self._run(root, root / ("paired_" + format_name),
                                              paired, nexus=nexus)
                    self._assert_records(paired_result, paired_rows)

    def test_repeated_categorical_points_ascii_and_nexus(self):
        categories = ["vanadium", "with space", "comma,value", 'quote"mark',
                      r"back\slash", "'single'", "vanadium"]
        categorical = ScanPlan.points([{"category": value} for value in categories])
        with TemporaryDirectory(prefix="mcstasscript-phase7-categorical-") as directory:
            root = Path(directory)
            for format_name, nexus in (("ascii", False), ("nexus", True)):
                with self.subTest(format=format_name):
                    result = self._run(root, root / format_name, categorical, nexus=nexus)
                    self._assert_records(result, [{"category": value} for value in categories])
                    self.assertEqual(
                        [point.parameters["category"] for point in result.points], categories
                    )
                    self.assertEqual(result.execution["counts"]["unknown"], len(categories))

    def test_singleton_empty_string_rejected_before_output(self):
        plan = ScanPlan.points([{"category": ""}])
        with TemporaryDirectory(prefix="mcstasscript-phase7-empty-string-") as directory:
            root = Path(directory)
            output = root / "scan"
            instrument = _make_instrument(root)
            instrument.settings(
                output_path=str(output),
                ncount=64,
                seed=314159,
                NeXus=False,
            )
            with self.assertRaisesRegex(ValueError, "empty"):
                instrument.scan(plan)
            self.assertFalse(output.exists())

    def test_singleton_one_point(self):
        plan = ScanPlan.points([{"theta": 0, "wavelength": 4}])
        with TemporaryDirectory(prefix="mcstasscript-phase7-singleton-") as directory:
            root = Path(directory)
            for format_name, nexus in (("ascii", False), ("nexus", True)):
                with self.subTest(format=format_name):
                    output = root / format_name
                    result = self._run(root, output, plan, nexus=nexus)
                    self._assert_records(result, [{"theta": 0, "wavelength": 4}])
                    self._assert_manifest(output, result)

    def test_partial_first_and_middle_failures_continue(self):
        cases = (("first", 0), ("middle", 1))
        with TemporaryDirectory(prefix="mcstasscript-phase7-partial-") as directory:
            root = Path(directory)
            for format_name, nexus in (("ascii", False), ("nexus", True)):
                for label, failed_id in cases:
                    with self.subTest(format=format_name, position=label):
                        plan = ScanPlan.points([
                            {"theta": -1, "fail": int(failed_id == 0)},
                            {"theta": 0, "fail": int(failed_id == 1)},
                            {"theta": 1, "fail": int(failed_id == 2)},
                        ])
                        output = root / (label + "_" + format_name)
                        result = self._run(root, output, plan, nexus=nexus,
                                           on_error="continue")
                        self._assert_records(result, list(plan_to_rows(plan)), require_data=False)
                        self.assertEqual(result.execution["process"]["status"], "succeeded")
                        self.assertEqual(result.execution["process"]["returncode"], 0)
                        self.assertEqual(result.execution["process"]["outcome"],
                                         "completed_with_output_issues")
                        self.assertGreater(result.execution["recorded"], 0)
                        self.assertLessEqual(result.execution["recorded"],
                                             result.execution["point_count"])
                        self.assertEqual(
                            [point.point_id for point in result.points], [0, 1, 2]
                        )
                        self._assert_native_once(output)
                        if nexus:
                            self._assert_nexus_file(output, valid=failed_id != 0)
                        for point in result.points:
                            expected_seed = 314159 + 1024 * native_index(
                                result.plan, point.point_id
                            )
                            self.assertEqual(point.metadata["seed"]["intended"], expected_seed)
                            if point.point_id == failed_id:
                                allowed_statuses = (("missing", "unreadable")
                                                     if nexus and failed_id == 0
                                                     else ("missing",))
                                self.assertIn(point.data_status, allowed_statuses)
                                if point.metadata["seed"]["actual"] is not None:
                                    self.assertEqual(point.metadata["seed"]["actual"],
                                                     expected_seed)
                                if point.source is None and not point.summaries:
                                    self.assertIsNone(point.metadata["seed"]["actual"])
                                    self.assertIsNone(point.metadata["native_point_id"])
                                    self.assertIsNone(point.metadata["storage_point_id"])
                                    self.assertIsNone(point.metadata["storage_entry"])
                            elif nexus and failed_id == 0:
                                # A corrupt combined HDF5 file leaves summaries but no arrays.
                                self.assertTrue(point.summaries)
                                self.assertTrue(any(summary.total_I > 0
                                                    for summary in point.summaries.values()))
                                with self.assertRaises(ScanDataError):
                                    point.data
                                self.assertIn(point.metadata["seed"]["actual"],
                                              (None, expected_seed))
                            else:
                                self.assertNotEqual(point.data_status, "missing")
                                self.assertEqual(point.metadata["seed"]["actual"], expected_seed)
                                self.assertTrue(np.any(point.data[0].Intensity != 0))
                        self._assert_manifest(output, result)
                        if nexus and failed_id != 0:
                            # Middle failures retain a valid combined file with omitted entries.
                            self._assert_nexus_file(output, valid=True)

    def test_first_failure_repeated_coordinates_are_not_guessed(self):
        plan = ScanPlan.points([
            {"theta": 0, "fail": 1},
            {"theta": 0, "fail": 0},
            {"theta": 0, "fail": 0},
        ])
        with TemporaryDirectory(prefix="mcstasscript-phase7-ambiguous-") as directory:
            root = Path(directory)
            output = root / "ambiguous"
            result = self._run(root, output, plan, nexus=True, on_error="continue")
            self._assert_native_once(output)
            self._assert_nexus_file(output, valid=False)
            self.assertEqual(result.execution["process"]["outcome"],
                             "completed_with_output_issues")
            self.assertTrue(all(point.status == "unknown" for point in result.points))
            self.assertTrue(all(point.data_status in ("missing", "unreadable")
                                for point in result.points))
            self.assertTrue(all(not point.summaries for point in result.points))

    def test_all_points_failed_raise_with_sparse_records(self):
        plan = ScanPlan.points([
            {"theta": -1, "fail": 1},
            {"theta": 0, "fail": 1},
            {"theta": 1, "fail": 1},
        ])
        with TemporaryDirectory(prefix="mcstasscript-phase7-allfailed-") as directory:
            root = Path(directory)
            for format_name, nexus in (("ascii", False), ("nexus", True)):
                with self.subTest(format=format_name):
                    output = root / format_name
                    with self.assertRaises(ScanExecutionError) as caught:
                        self._run(root, output, plan, nexus=nexus, on_error="raise")
                    result = caught.exception.result
                    self.assertEqual(result.execution["counts"]["unknown"], 3)
                    self.assertTrue(all(point.status == "unknown" for point in result.points))
                    self.assertEqual(result.execution["process"]["status"], "succeeded")
                    self.assertEqual(result.execution["process"]["returncode"], 0)
                    self.assertEqual(result.execution["process"]["outcome"],
                                     "completed_with_output_issues")
                    self.assertTrue((output / "scan_manifest.json").is_file())
                    self._assert_native_once(output)
                    if nexus:
                        # McStas leaves an invalid empty combined file when no point succeeds.
                        self._assert_nexus_file(output, valid=False)
                    else:
                        self.assertFalse(list((output / "native").glob("[0-9]/spectrum.dat")))
                        rows = [line for line in (output / "native" / "mccode.dat")
                                .read_text(encoding="utf-8").splitlines()
                                if line.strip() and not line.lstrip().startswith("#")]
                        self.assertEqual(rows, [])
                    reloaded = load_scan(output)
                    self.assertEqual(reloaded.execution, result.execution)
                    document = json.loads((output / "scan_manifest.json").read_text(
                        encoding="utf-8"))
                    self.assertEqual(len(document["points"]), result.execution["recorded"])
                    for point in result.points:
                        if point.source is None and not point.summaries:
                            self.assertIsNone(point.metadata["seed"]["actual"])

    def test_scan_preserves_populated_backengine_output(self):
        with TemporaryDirectory(prefix="mcstasscript-phase7-backengine-") as directory:
            root = Path(directory)
            instrument = _make_instrument(root)
            single_output = root / "single"
            scan_output = root / "scan"
            instrument.settings(
                output_path=str(single_output),
                increment_folder_name=False,
                ncount=64,
                seed=314159,
            )
            instrument.backengine()

            output_container = instrument.output
            output_keys = instrument.output_keys
            output_data = output_container[output_keys[0]]
            populated_collection = output_data.get_data()["data"]
            self.assertIsNotNone(populated_collection)
            self.assertGreater(len(populated_collection), 0)
            self.assertIs(instrument.output[output_keys[0]], output_data)

            instrument.settings(
                output_path=str(scan_output),
                ncount=64,
                seed=314159,
                NeXus=False,
            )
            parameters = instrument.parameters
            parameter_state = {
                name: (parameter, copy.deepcopy(parameter.value), parameter.type)
                for name, parameter in parameters.parameters.items()
            }
            run_settings = instrument._run_settings
            run_settings_copy = copy.deepcopy(run_settings)
            run_parameters = run_settings["parameters"]
            run_parameters_copy = copy.deepcopy(run_parameters)

            plan = ScanPlan.points([{"theta": 0, "wavelength": 4}])
            try:
                result = instrument.scan(plan)
            finally:
                self.assertIs(instrument.output, output_container)
                self.assertIs(instrument.output_keys, output_keys)
                self.assertIs(instrument.output[output_keys[0]], output_data)
                self.assertIs(output_data.get_data()["data"], populated_collection)
                self.assertIs(instrument.parameters, parameters)
                self.assertIs(instrument._run_settings, run_settings)
                self.assertEqual(instrument._run_settings, run_settings_copy)
                self.assertIs(instrument._run_settings["parameters"], run_parameters)
                self.assertEqual(instrument._run_settings["parameters"], run_parameters_copy)
                for name, (parameter, value, parameter_type) in parameter_state.items():
                    current = instrument.parameters.parameters[name]
                    self.assertIs(current, parameter)
                    self.assertEqual(current.value, value)
                    self.assertEqual(current.type, parameter_type)

            self._assert_records(result, [{"theta": 0, "wavelength": 4}])
            self.assertTrue(single_output.exists())
            self.assertTrue(scan_output.exists())


def plan_to_rows(plan):
    for point in plan:
        yield dict(point.parameters)


if __name__ == "__main__":
    unittest.main()
