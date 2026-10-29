import json
from copy import deepcopy
from contextlib import ExitStack
import inspect
from pathlib import Path
import signal
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mcstasscript.data.scan import ScanMonitor, ScanMonitorSummary, ScanPoint, ScanResult, ScanSource
from mcstasscript.data.scan_manifest import ScanManifest
from mcstasscript.data.scan_plan import ScanPlan
from mcstasscript.helper.scan_execution import (
    ScanExecutionError, _interrupt, _point_has_issue, _set_copy_values,
    _stop_process, execute_scan,
)
from mcstasscript.helper.scan_preflight import ScanPreflight
from mcstasscript.interface.instr import McCode_instr


class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.type = None


def make_report(root, plan, *, valid=True, errors=(), effective_overrides=None):
    settings = {
        "ncount": 10, "seed": 7, "mpi": None, "NeXus": False,
        "openacc": False, "gravity": False, "force_compile": True,
        "increment_folder_name": True, "custom_flags": "", "suppress_output": True,
        "checks": True, "save_comp_pars": False, "executable": "mcrun",
        "executable_path": str(root), "package_path": str(root),
        "run_path": str(root), "output_path": str(root / "scan"),
    }
    if effective_overrides:
        settings.update(effective_overrides)
    return ScanPreflight(
        plan=plan, fixed_parameters={}, parameter_snapshot={"theta": 0.0},
        effective_settings=settings, tool={
            "package": "McStas", "version": "3.10.0", "executable": "mcrun",
            "capabilities": {"seed_policy": "point_offset_1024"},
        }, output_path=str(root / "scan"), errors=errors, warnings=())


class FakeInstrument:
    def __init__(self, report, *, run_settings=None):
        self.name = "scan_instrument"
        self.report = report
        self.parameters = SimpleNamespace(parameters={"theta": Parameter("theta", 0.0)})
        self._run_settings = {"checks": True, "save_comp_pars": False}
        if run_settings:
            self._run_settings.update(run_settings)
        self.output_marker = object()
        self.writes = 0
        self.preflight_calls = []

    def scan_preflight(self, plan):
        self.preflight_calls.append(plan)
        return self.report

    def __getstate__(self):
        state = dict(self.__dict__)
        state["report"] = None
        return state

    def write_full_instrument(self):
        self.writes += 1
        path = Path(self.input_path) / (self.name + ".instr")
        path.write_text("DEFINE INSTRUMENT scan_instrument(double theta=0)\nEND\n")


def good_finalizer(manifest, root):
    monitor = ScanMonitor("detector", "detector.dat")
    point = ScanPoint(
        0, {"theta": 0.0}, ScanSource(Path(root) / "native" / "0"),
        summaries=[ScanMonitorSummary(monitor, total_I=1.0)],
    )
    manifest.record_point(0, source=point.source, data_status="unknown")
    return manifest, ScanResult([point], parameter_names=("theta",), plan=manifest.plan)


class TestScanExecution(unittest.TestCase):
    def test_settings_keyword_is_rejected_before_writing(self):
        self.assertNotIn("settings", inspect.signature(McCode_instr.scan).parameters)
        self.assertNotIn("settings", inspect.signature(execute_scan).parameters)

        for caller in ("public", "helper"):
            with self.subTest(caller=caller), TemporaryDirectory() as directory:
                root = Path(directory)
                plan = ScanPlan.points([{"theta": 0.0}])
                instrument = FakeInstrument(make_report(root, plan))
                if caller == "public":
                    call = lambda: McCode_instr.scan(
                        instrument, plan, settings={"ncount": 1})
                else:
                    call = lambda: execute_scan(
                        instrument, plan, settings={"ncount": 1})

                with self.assertRaises(TypeError):
                    call()
                self.assertEqual(instrument.writes, 0)
                self.assertFalse((root / "scan").exists())

    def test_inherited_settings_are_used_without_preflight_overrides(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            inherited = {"ncount": 123, "seed": 19, "NeXus": True}
            report = make_report(root, plan, effective_overrides=inherited)
            instrument = FakeInstrument(report, run_settings=inherited)
            process = Mock()
            process.wait.return_value = 0
            captured = {}

            def command(command_report, instrument_file, native_path, *, seed):
                captured["settings"] = dict(command_report.effective_settings)
                captured["seed"] = seed
                return ["mcrun", "instrument.instr"]

            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       side_effect=command), \
                     patch("mcstasscript.data.scan_persistence.finalize_manifest",
                           side_effect=good_finalizer), \
                     patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                           return_value=process):
                execute_scan(instrument, plan)

            self.assertEqual(instrument.preflight_calls, [plan])
            self.assertEqual(
                {key: captured["settings"][key] for key in inherited}, inherited)
            self.assertEqual(captured["seed"], inherited["seed"])

    def test_force_compile_setting_is_preserved_for_all_outcomes(self):
        outcomes = ("success", "write_failure", "popen_failure",
                    "native_nonzero", "keyboard_interrupt")
        for configured in (True, False):
            for outcome in outcomes:
                with self.subTest(force_compile=configured, outcome=outcome), \
                        TemporaryDirectory() as directory:
                    root = Path(directory)
                    plan = ScanPlan.points([{"theta": 0.0}])
                    instrument = FakeInstrument(
                        make_report(root, plan,
                                    effective_overrides={"force_compile": configured}),
                        run_settings={"force_compile": configured})
                    original_parameters = instrument.parameters
                    original_parameter = instrument.parameters.parameters["theta"]
                    original_parameter_value = original_parameter.value
                    original_settings = instrument._run_settings
                    original_settings_snapshot = deepcopy(original_settings)
                    original_output_marker = instrument.output_marker
                    process = Mock()
                    process.wait.return_value = (
                        3 if outcome == "native_nonzero" else 0)

                    if outcome == "write_failure":
                        instrument.write_full_instrument = Mock(
                            side_effect=OSError("write failed"))

                    with ExitStack() as stack:
                        stack.enter_context(patch(
                            "mcstasscript.helper.scan_command.build_scan_command",
                            return_value=["mcrun", "instrument.instr"]))
                        if outcome in ("success", "native_nonzero"):
                            stack.enter_context(patch(
                                "mcstasscript.data.scan_persistence.finalize_manifest",
                                side_effect=good_finalizer))
                        if outcome == "popen_failure":
                            stack.enter_context(patch(
                                "mcstasscript.helper.scan_execution.subprocess.Popen",
                                side_effect=OSError("not executable")))
                        elif outcome in ("success", "native_nonzero",
                                         "keyboard_interrupt"):
                            stack.enter_context(patch(
                                "mcstasscript.helper.scan_execution.subprocess.Popen",
                                return_value=process))
                        if outcome == "keyboard_interrupt":
                            process.wait.side_effect = KeyboardInterrupt()
                            stack.enter_context(patch(
                                "mcstasscript.helper.scan_execution.os.killpg"))

                        if outcome == "keyboard_interrupt":
                            with self.assertRaises(KeyboardInterrupt):
                                execute_scan(instrument, plan)
                        elif outcome in ("write_failure", "popen_failure",
                                          "native_nonzero"):
                            execute_scan(instrument, plan, on_error="continue")
                        else:
                            execute_scan(instrument, plan)

                    self.assertIs(instrument.parameters, original_parameters)
                    self.assertIs(instrument.parameters.parameters["theta"],
                                  original_parameter)
                    self.assertEqual(instrument.parameters.parameters["theta"].value,
                                     original_parameter_value)
                    self.assertIs(instrument._run_settings, original_settings)
                    self.assertEqual(instrument._run_settings,
                                     original_settings_snapshot)
                    self.assertIs(instrument.output_marker, original_output_marker)
                    self.assertEqual(instrument._run_settings["force_compile"],
                                     configured)
                    self.assertEqual(
                        instrument.report.effective_settings["force_compile"], configured)
                    document = json.loads((root / "scan" / "scan_manifest.json").read_text())
                    self.assertTrue(document["effective_settings"]["force_compile"])

    def test_success_isolated_and_nonmutating(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            report = make_report(root, plan)
            instrument = FakeInstrument(report)
            original_value = instrument.parameters.parameters["theta"].value
            original_marker = instrument.output_marker
            output = root / "scan"
            process = Mock()
            process.wait.return_value = 0
            native_existed_at_launch = []

            def launch(*args, **kwargs):
                native_existed_at_launch.append((output / "native").exists())
                return process

            instrument._run_settings["parameters"] = {"theta": 0.0}
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.data.scan_persistence.finalize_manifest",
                          side_effect=good_finalizer), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          side_effect=launch) as popen:
                result = execute_scan(instrument, plan)

            self.assertEqual(instrument.writes, 0)
            self.assertEqual(instrument.parameters.parameters["theta"].value, original_value)
            self.assertIs(instrument.output_marker, original_marker)
            self.assertTrue((output / "source" / "scan_instrument.instr").is_file())
            self.assertTrue((output / "execution.log").is_file())
            self.assertEqual(result.plan, plan)
            self.assertEqual(result.points[0].point_id, 0)
            self.assertEqual(popen.call_args.kwargs["shell"], False)
            self.assertEqual(popen.call_args.kwargs["cwd"], str(root))
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(native_existed_at_launch, [False])
            document = json.loads((output / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"], "succeeded")

    def test_invalid_nexus_is_not_reexecuted_or_replaced(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            base = make_report(root, plan)
            settings = dict(base.effective_settings)
            settings["NeXus"] = True
            report = ScanPreflight(
                plan=plan, fixed_parameters={}, parameter_snapshot={"theta": 0.0},
                effective_settings=settings,
                tool=json.loads(json.dumps(base.tool)),
                output_path=base.output_path)
            instrument = FakeInstrument(report)
            process = Mock()
            process.wait.return_value = 0
            raw_h5 = b"invalid native h5"

            def finalizer(manifest, native_root):
                native = Path(native_root) / "native"
                native.mkdir()
                (native / "mccode.h5").write_bytes(raw_h5)
                return good_finalizer(manifest, native_root)

            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.data.scan_persistence.finalize_manifest",
                          side_effect=finalizer), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process) as popen:
                execute_scan(instrument, plan, on_error="continue")

            self.assertEqual(popen.call_count, 1)
            self.assertEqual((root / "scan" / "native" / "mccode.h5").read_bytes(), raw_h5)

    def test_unsafe_instrument_name_is_rejected_before_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            instrument.name = "scan-instrument"
            with self.assertRaises(ValueError):
                execute_scan(instrument, plan)
            self.assertFalse((root / "scan").exists())

    def test_large_plan_rejects_signed64_seed_overflow_before_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.grid({name: range(1000) for name in "abcdef"})
            report = make_report(root, plan)
            instrument = FakeInstrument(report)
            with self.assertRaisesRegex(ValueError, "too large"):
                execute_scan(instrument, plan)
            self.assertFalse((root / "scan").exists())

    def test_fixed_c_string_is_canonicalized_but_snapshot_stays_raw(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            base = make_report(root, plan)
            report = ScanPreflight(
                plan=plan, fixed_parameters={"category": '"vanadium"'},
                parameter_snapshot={"theta": 0.0, "category": '"vanadium"'},
                effective_settings=dict(base.effective_settings),
                tool={"package": "McStas", "version": "3.10.0",
                      "capabilities": {"seed_policy": "point_offset_1024"}},
                output_path=base.output_path)
            instrument = FakeInstrument(report)
            category = Parameter("category", '"vanadium"')
            category.type = "string"
            instrument.parameters.parameters["category"] = category
            captured = {}
            process = Mock()
            process.wait.return_value = 0

            def command(command_report, instrument_file, native_path, *, seed):
                captured["fixed"] = dict(command_report.fixed_parameters)
                return ["mcrun", "instrument.instr"]

            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       side_effect=command), \
                    patch("mcstasscript.data.scan_persistence.finalize_manifest",
                          side_effect=good_finalizer), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process):
                execute_scan(instrument, plan)
            self.assertEqual(captured["fixed"]["category"], "vanadium")
            self.assertEqual(instrument.parameters.parameters["category"].value,
                             '"vanadium"')
            document = json.loads((root / "scan" / "scan_manifest.json").read_text())
            self.assertEqual(document["instrument"]["fixed_parameters"]["category"],
                             "vanadium")
            self.assertEqual(document["instrument"]["parameter_snapshot"]["category"],
                             '"vanadium"')

    def test_scanned_string_values_keep_edge_quotes_and_copy_has_no_string_default(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"category": '"literal quotes"'}])
            report = make_report(root, plan)
            instrument = FakeInstrument(report)
            category = Parameter("category", None)
            category.type = "string"
            instrument.parameters.parameters = {"category": category}
            working = deepcopy(instrument)
            _set_copy_values(working, plan)
            self.assertEqual(working.parameters.parameters["category"].value,
                             '"__mcstasscript_scan__"')
            self.assertIsNone(instrument.parameters.parameters["category"].value)

    def test_manifest_normalizes_numpy_and_path_settings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            base = make_report(root, plan)
            settings = dict(base.effective_settings)
            settings["mpi"] = np.int64(2)
            settings["package_path"] = root
            report = ScanPreflight(
                plan=plan, fixed_parameters={}, parameter_snapshot={"theta": 0.0},
                effective_settings=settings, tool={
                    "package": "McStas", "version": "3.10.0",
                    "capabilities": {"seed_policy": "point_offset_1024"}},
                output_path=base.output_path)
            instrument = FakeInstrument(report)
            process = Mock()
            process.wait.return_value = 0
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.data.scan_persistence.finalize_manifest",
                          side_effect=good_finalizer), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process):
                execute_scan(instrument, plan)
            document = json.loads((root / "scan" / "scan_manifest.json").read_text())
            self.assertEqual(document["effective_settings"]["mpi"], 2)
            self.assertEqual(document["effective_settings"]["package_path"], str(root))

    def test_association_issues_are_execution_failures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            process = Mock()
            process.wait.return_value = 0

            def conflicted_finalizer(manifest, native_root):
                manifest, result = good_finalizer(manifest, native_root)
                result.metadata["association_issues"] = ["ambiguous native match"]
                return manifest, result

            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.data.scan_persistence.finalize_manifest",
                          side_effect=conflicted_finalizer), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process):
                result = execute_scan(instrument, plan, on_error="continue")
            self.assertIn("ambiguous native match",
                          result.execution["process"]["diagnostics"])

    def test_native_advisory_diagnostics_are_not_execution_failures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            process = Mock()
            process.wait.return_value = 0

            def advisory_finalizer(manifest, native_root):
                manifest, result = good_finalizer(manifest, native_root)
                result.metadata["diagnostics"] = [
                    "Entry order identifies storage only; original point IDs are unknown.",
                    "Declared Numpoints differs from summary rows.",
                ]
                result.metadata["native_data_status"] = "available"
                return manifest, result

            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.data.scan_persistence.finalize_manifest",
                          side_effect=advisory_finalizer), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process):
                result = execute_scan(instrument, plan)

            self.assertEqual(result.execution["process"]["status"], "succeeded")
            self.assertEqual(result.execution["process"]["diagnostics"], [])

    def test_nan_summary_and_known_missing_source_are_issues(self):
        monitor = ScanMonitor("detector")
        nan_point = ScanPoint(
            0, {"theta": 0.0},
            summaries=[ScanMonitorSummary(monitor, total_I=float("nan"))],
        )
        missing_point = ScanPoint(1, {"theta": 0.0}, ScanSource("/missing"),
                                  summaries=[ScanMonitorSummary(monitor, total_I=1.0)],
                                  data_status="missing")
        self.assertTrue(_point_has_issue(nan_point))
        self.assertTrue(_point_has_issue(missing_point))

    def test_preflight_and_old_runtime_fail_before_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            invalid = make_report(root, plan, valid=False, errors=("bad",))
            instrument = FakeInstrument(invalid)
            with self.assertRaises(ValueError), patch.object(instrument, "__deepcopy__",
                                                              side_effect=AssertionError,
                                                              create=True):
                execute_scan(instrument, plan)
            self.assertFalse((root / "scan").exists())

            old = make_report(root, plan)
            # The report is immutable, so construct the old version explicitly.
            old = ScanPreflight(plan=plan, fixed_parameters={},
                                parameter_snapshot={"theta": 0.0},
                                effective_settings=dict(old.effective_settings),
                                tool={"package": "McStas", "version": "3.7.14"},
                                output_path=str(root / "old"))
            instrument.report = old
            with self.assertRaises(ValueError):
                execute_scan(instrument, plan)
            self.assertFalse((root / "old").exists())

    def test_launch_failure_continue_leaves_reloadable_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          side_effect=OSError("not executable")):
                result = execute_scan(instrument, plan, on_error="continue")
            document = json.loads((root / "scan" / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"], "failed")
            self.assertEqual([point.point_id for point in result.points], [0])
            self.assertEqual(result.points[0].data_status, "missing")

    def test_interrupt_is_reraised_with_partial_result(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            process = Mock()
            process.wait.side_effect = KeyboardInterrupt()
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process), \
                    patch("mcstasscript.helper.scan_execution.os.killpg"):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    execute_scan(instrument, plan)
            self.assertIsInstance(caught.exception.result, ScanResult)
            self.assertEqual([point.point_id for point in caught.exception.result.points], [0])
            document = json.loads((root / "scan" / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"], "interrupted")

    def test_wait_failure_stops_process_and_returns_full_partial_result(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            process = Mock()
            process.wait.side_effect = OSError("wait failed")
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                    patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                          return_value=process), \
                    patch("mcstasscript.helper.scan_execution.os.killpg") as killpg:
                result = execute_scan(instrument, plan, on_error="continue")
            self.assertEqual(killpg.call_count, 2)
            self.assertEqual([point.point_id for point in result.points], [0])

    def test_interrupt_during_instrument_write_is_checkpointed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            instrument.write_full_instrument = Mock(side_effect=KeyboardInterrupt())
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    execute_scan(instrument, plan)
            self.assertEqual([point.point_id for point in caught.exception.result.points], [0])
            document = json.loads((root / "scan" / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"], "interrupted")

    def test_stop_process_allows_sigterm_interrupt_to_escape(self):
        process = Mock()
        process.pid = 12345
        with patch("mcstasscript.helper.scan_execution.os.killpg",
                   side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                _stop_process(process)
        process.terminate.assert_not_called()

    def test_interrupt_cleanup_failures_keep_result_and_checkpoint(self):
        for close_error in (KeyboardInterrupt(), OSError("close failed")):
            with self.subTest(close_error=type(close_error).__name__), TemporaryDirectory() as directory:
                root = Path(directory)
                plan = ScanPlan.points([{"theta": 0.0}])
                output = root / "scan"
                output.mkdir()
                manifest = ScanManifest.from_preflight(make_report(root, plan))
                log_handle = Mock(closed=False)
                log_handle.close.side_effect = close_error
                original_handler = object()
                with patch("mcstasscript.helper.scan_execution._stop_process",
                           side_effect=KeyboardInterrupt()), \
                    patch("mcstasscript.helper.scan_execution.signal.signal",
                           side_effect=[original_handler, None]) as signal_call:
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        _interrupt(manifest, output, KeyboardInterrupt(),
                                   log_handle=log_handle)
                self.assertIsInstance(caught.exception.result, ScanResult)
                document = json.loads((output / "scan_manifest.json").read_text())
                self.assertEqual(document["execution"]["process"]["status"],
                                 "interrupted")
                self.assertEqual(signal_call.call_args_list[0].args,
                                 (signal.SIGINT, signal.SIG_IGN))
                self.assertEqual(signal_call.call_args_list[1].args,
                                 (signal.SIGINT, original_handler))

    def test_interrupt_restoration_failure_does_not_replace_original_interrupt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            output = root / "scan"
            output.mkdir()
            manifest = ScanManifest.from_preflight(make_report(root, plan))
            with patch("mcstasscript.helper.scan_execution.signal.signal",
                       side_effect=[signal.SIG_DFL, OSError("restore failed")]):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    _interrupt(manifest, output, KeyboardInterrupt())
            self.assertIsInstance(caught.exception.result, ScanResult)
            document = json.loads((output / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"],
                             "interrupted")

    def test_sigterm_interrupt_is_checkpointed_before_reraising(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            instrument = FakeInstrument(make_report(root, plan))
            process = Mock()
            process.wait.side_effect = KeyboardInterrupt()
            with patch("mcstasscript.helper.scan_command.build_scan_command",
                       return_value=["mcrun", "instrument.instr"]), \
                 patch("mcstasscript.helper.scan_execution.subprocess.Popen",
                       return_value=process), \
                 patch("mcstasscript.helper.scan_execution.os.killpg",
                       side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    execute_scan(instrument, plan)
            self.assertIsInstance(caught.exception.result, ScanResult)
            document = json.loads((root / "scan" / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"],
                             "interrupted")

    def test_worker_interrupt_skips_signal_handler_installation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0.0}])
            output = root / "scan"
            output.mkdir()
            manifest = ScanManifest.from_preflight(make_report(root, plan))
            caught = []

            def interrupt_worker():
                try:
                    _interrupt(manifest, output, KeyboardInterrupt())
                except KeyboardInterrupt as error:
                    caught.append(error)

            with patch("mcstasscript.helper.scan_execution.signal.signal",
                       side_effect=ValueError("main thread only")) as signal_call:
                worker = threading.Thread(target=interrupt_worker)
                worker.start()
                worker.join()
            self.assertEqual(len(caught), 1)
            self.assertFalse(signal_call.called)
            document = json.loads((output / "scan_manifest.json").read_text())
            self.assertEqual(document["execution"]["process"]["status"],
                             "interrupted")


if __name__ == "__main__":
    unittest.main()
