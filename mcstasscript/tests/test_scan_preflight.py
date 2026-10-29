import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mcstasscript.data.scan_plan import ScanPlan
from mcstasscript.data.scan_manifest import ScanManifest
from mcstasscript.helper.mcstas_objects import provide_parameter
from mcstasscript.helper.scan_preflight import scan_preflight
from mcstasscript.interface.instr import McStas_instr


def _runner_fixture(root):
    runner_dir = Path(root) / "runner-bin"
    runner_dir.mkdir(exist_ok=True)
    runner = runner_dir / "mcrun"
    runner.write_text("")
    runner.chmod(0o755)
    return runner_dir


def fake_instrument(parameters, output_path, version=(3, 10, 0), package="McStas"):
    runner_dir = _runner_fixture(Path(output_path).parent)
    return SimpleNamespace(
        parameters=SimpleNamespace(parameters={p.name: p for p in parameters}),
        output_path=str(output_path),
        _run_settings={"run_path": ".", "executable": "mcrun",
                       "executable_path": str(runner_dir)},
        package_name=package,
        mccode_major_version=version[0] if version else "Unknown",
        mccode_minor_version=version[1] if version else None,
        mccode_patch_version=str(version[2]) if version else None,
        run_from_ref=None,
        run_to_ref=None,
        run_from_component_parameters=None,
        run_to_component_parameters=None,
    )


class TestScanPlan(unittest.TestCase):
    def test_lazy_grid_order_and_detached_dict(self):
        plan = ScanPlan.grid({"x": (1, 2), "y": (10, 20, 30)})
        self.assertEqual(plan.point_count, 6)
        self.assertEqual(plan[1].parameters, {"x": 1, "y": 20})
        self.assertEqual(plan[4].grid_indices, {"x": 1, "y": 1})
        data = plan.to_dict()
        data["axes"]["x"][0] = 99
        self.assertEqual(plan[0].parameters["x"], 1)
        self.assertEqual(ScanPlan.from_dict(plan.to_dict()).to_dict(), plan.to_dict())


class TestScanPreflight(unittest.TestCase):
    def test_relative_runner_path_is_anchored_for_command_construction(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new")
            runner_dir = instrument._run_settings["executable_path"]
            instrument._run_settings["executable_path"] = os.path.relpath(runner_dir)
            report = scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]))
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.tool["path_evidence"]["resolved_executable"],
                             str(Path(runner_dir) / "mcrun"))

    def test_plan_input_is_strict(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new")
            with self.assertRaises(TypeError):
                scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]).to_dict())

    def test_cached_runner_must_resolve_without_probing(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new")
            instrument._run_settings["executable"] = None
            report = scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]))
            self.assertFalse(report.valid)
            self.assertTrue(any("executable must be" in error for error in report.errors))

            instrument._run_settings["executable"] = "mcrun"
            instrument._run_settings["executable_path"] = str(Path(directory) / "missing")
            report = scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]))
            self.assertFalse(report.valid)
            self.assertTrue(any("executable_path must be" in error
                                for error in report.errors))

            instrument._run_settings["executable_path"] = str(_runner_fixture(directory))
            instrument._run_settings["package_path"] = str(Path(directory) / "missing-package")
            report = scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]))
            self.assertFalse(report.valid)
            self.assertTrue(any("package_path must be" in error
                                for error in report.errors))

    def test_array_settings_and_overrides_are_diagnostics_not_truth_tests(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new")
            report = scan_preflight(
                instrument, ScanPlan.points([{"x": 1.0}]),
                settings={"mpi": np.array([1, 2]),
                          "executable": np.array(["mcrun"])},
            )
            self.assertFalse(report.valid)
            self.assertTrue(any("mpi must be" in error for error in report.errors))
            with self.assertRaises(TypeError):
                report.to_dict()

    def test_report_is_valid_and_json_safe(self):
        theta = provide_parameter("double", "theta", value=2.0)
        fixed = provide_parameter("int", "fixed", value=3)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([theta, fixed], Path(directory) / "new")
            plan = ScanPlan.grid({"theta": (1.0, 2.0)})
            report = scan_preflight(instrument, plan, settings={"seed": "42"})
            self.assertTrue(report.valid)
            self.assertEqual(report.effective_settings["seed"], "42")
            self.assertIn("package_path", report.effective_settings)
            self.assertEqual(report.tool["capability_evidence"],
                             "cached instrument version; verified fixtures")
            self.assertEqual(json.loads(json.dumps(report.to_dict()))["valid"], True)
            with self.assertRaises(TypeError):
                report.effective_settings["ncount"] = 4
            with self.assertRaises(TypeError):
                report.tool["capabilities"]["new"] = True
            report.raise_for_errors()

    def test_bad_values_and_existing_output_are_diagnostics_without_mutation(self):
        parameter = provide_parameter("int", "count", value=2)
        original = parameter.value
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            report = scan_preflight(
                fake_instrument([parameter], output),
                ScanPlan.points([{"count": 1.5}]),
                settings={"ncount": 1.5, "custom_flags": "--unsafe"},
            )
            self.assertFalse(report.valid)
            self.assertEqual(parameter.value, original)
            self.assertGreaterEqual(len(report.errors), 3)
            with self.assertRaises(ValueError):
                report.raise_for_errors()

    def test_unknown_capability_and_partial_run_are_errors(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new", version=None)
            instrument.run_to_ref = "monitor"
            report = scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]))
            self.assertFalse(report.valid)
            self.assertTrue(any("capabilities" in error for error in report.errors))
            self.assertTrue(any("run_from/run_to" in error for error in report.errors))

    def test_constraints_are_checked_on_copies_and_operations_are_forbidden(self):
        parameter = provide_parameter("int", "x", value=1, options=[1, 2])
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new")
            for method_name in ("write_full_instrument", "backengine", "set_parameters", "settings"):
                setattr(instrument, method_name,
                        lambda: self.fail("preflight must not call " + method_name))
            with patch.object(subprocess, "check_output", side_effect=AssertionError("no probing")):
                report = scan_preflight(
                    instrument,
                    ScanPlan.points([{"x": 3.5}, {"x": 4}]),
                    settings={"seed": 19},
                )
            self.assertFalse(report.valid)
            self.assertGreaterEqual(len(report.errors), 2)
            self.assertEqual(parameter.value, 1)
            self.assertEqual(parameter.type, "int")

    def test_unset_parameters_unknown_names_and_intervals(self):
        scanned = provide_parameter("double", "x")
        fixed = provide_parameter("double", "fixed")
        scanned.add_interval(0.0, 2.0, True)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([scanned, fixed], Path(directory) / "new")
            report = scan_preflight(instrument, ScanPlan.paired(x=[0.0, 2.0]))
            self.assertFalse(report.valid)
            self.assertTrue(any("Fixed parameter 'fixed' is unset" in error
                                for error in report.errors))
            fixed.value = 1.0
            report = scan_preflight(instrument, ScanPlan.paired(x=[0.0, 2.0]))
            self.assertTrue(report.valid, report.errors)
            self.assertIsNone(scanned.value)
            report = scan_preflight(instrument, ScanPlan.points([{"x": 3.0}]))
            self.assertFalse(report.valid)
            self.assertIsNone(scanned.value)
            report = scan_preflight(instrument, ScanPlan.points([{"unknown": 1.0}]))
            self.assertTrue(any("Unknown scan parameter" in error for error in report.errors))

    def test_output_override_and_effective_seed_are_detached(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "unused")
            output = Path(directory) / "override"
            report = scan_preflight(
                instrument, ScanPlan.points([{"x": 2.0}]),
                settings={"output_path": output, "seed": 23, "ncount": 10},
            )
            self.assertTrue(report.valid)
            self.assertEqual(report.output_path, str(output))
            self.assertEqual(report.effective_settings["seed"], 23)
            self.assertEqual(instrument.output_path, str(Path(directory) / "unused"))

    def test_runner_and_run_path_overrides_are_conservative(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            instrument = fake_instrument([parameter], Path(directory) / "new")
            instrument._run_settings["executable_path"] = directory
            report = scan_preflight(
                instrument, ScanPlan.points([{"x": 1.0}]),
                settings={"executable": "other", "executable_path": "/other"},
            )
            self.assertFalse(report.valid)
            self.assertTrue(any("cached instrument runner" in error
                                for error in report.errors))

            instrument._run_settings["run_path"] = str(Path(directory) / "missing")
            report = scan_preflight(instrument, ScanPlan.points([{"x": 1.0}]))
            self.assertFalse(report.valid)
            self.assertTrue(any("run_path must be" in error for error in report.errors))

    def test_seed_policy_is_reported_for_37(self):
        parameter = provide_parameter("double", "x", value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            report = scan_preflight(
                fake_instrument([parameter], Path(directory) / "new", version=(3, 7, 14)),
                ScanPlan.points([{"x": 1.0}]),
            )
            self.assertTrue(report.valid)
            self.assertEqual(report.tool["capabilities"]["seed_policy"],
                             "sequential_reuse")
            self.assertTrue(report.warnings)

    def test_large_grid_is_preflighted_by_axis_without_expansion(self):
        x = provide_parameter("double", "x", value=0.0)
        y = provide_parameter("double", "y", value=0.0)
        with tempfile.TemporaryDirectory() as directory:
            plan = ScanPlan.grid(x=range(1000), y=range(1000))
            report = scan_preflight(fake_instrument([x, y], Path(directory) / "new"), plan)
            self.assertTrue(report.valid)
            self.assertEqual(report.plan.point_count, 1000 ** 2)

    def test_mccode_method_is_thin_and_uses_cached_version(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "mcstasscript.interface.instr.check_mcstas_version",
                return_value=(3, 10, "0")):
            runner_dir = _runner_fixture(directory)
            def calibration(instrument):
                instrument.line_limit = 180
                instrument._run_settings["executable_path"] = str(runner_dir)
                instrument._run_settings["package_path"] = "/"

            with patch.object(McStas_instr, "_read_calibration", calibration):
                instrument = McStas_instr(
                    "preflight_test", package_path="/",
                    executable_path=str(runner_dir),
                    output_path=str(Path(directory) / "new"))
            instrument.add_parameter("double", "x", value=1.0)
            with patch.object(subprocess, "check_output",
                              side_effect=AssertionError("no version probing")), \
                    patch.object(subprocess, "run",
                                 side_effect=AssertionError("no execution")):
                report = instrument.scan_preflight(ScanPlan.points([{"x": 1.0}]))
            self.assertTrue(report.valid)

            manifest = ScanManifest.from_preflight(report)
            restored = ScanManifest.from_dict(
                json.loads(json.dumps(manifest.to_dict())))
            self.assertEqual(restored.plan, report.plan)
            self.assertEqual(restored.output_root, Path(report.output_path))


if __name__ == "__main__":
    unittest.main()
