import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mcstasscript.data.scan_plan import ScanPlan
from mcstasscript.helper.scan_command import (
    build_scan_command,
    native_index,
    plan_point_id,
)


def report(plan, *, package="McStas", version="3.10.0", settings=None,
           fixed=None, runner="/opt/mcstas/bin/mcrun"):
    effective = {
        "ncount": 64, "NeXus": False, "mpi": None,
        "openacc": False, "gravity": False, "custom_flags": "",
    }
    if settings:
        effective.update(settings)
    return SimpleNamespace(
        plan=plan,
        fixed_parameters={} if fixed is None else fixed,
        effective_settings=effective,
        tool={"package": package, "version": version,
              "path_evidence": {"resolved_executable": runner}},
        errors=(),
    )


class TestNativeIndex(unittest.TestCase):
    def test_grid_mapping_is_inverse_and_keeps_singletons(self):
        plan = ScanPlan.grid({"first": (1, 2), "single": (7,), "last": (3, 4, 5)})
        expected = [0, 2, 4, 1, 3, 5]
        self.assertEqual([native_index(plan, point) for point in range(6)], expected)
        for native in range(plan.point_count):
            self.assertEqual(plan_point_id(plan, native_index(plan, native)), native)
            self.assertEqual(native_index(plan, plan_point_id(plan, native)), native)

    def test_non_grid_mapping_is_identity(self):
        for plan in (ScanPlan.paired(x=[1, 2]),
                     ScanPlan.points([{"x": 1}, {"x": 2}])):
            self.assertEqual([native_index(plan, i) for i in range(2)], [0, 1])
            self.assertEqual([plan_point_id(plan, i) for i in range(2)], [0, 1])


class TestBuildScanCommand(unittest.TestCase):
    def test_grid_command_reorders_lists_and_serializes_flags(self):
        plan = ScanPlan.grid({"x": (np.int64(-1), np.int64(1)),
                              "y": (2, 4, 6)})
        with tempfile.TemporaryDirectory() as directory:
            command = build_scan_command(
                report(plan, fixed={"fixed": "literal, value \"q\" \\path"},
                       settings={"NeXus": True, "mpi": "auto",
                                 "openacc": True, "gravity": True}),
                "instrument.instr", Path(directory) / "native", seed=np.int64(19))
        self.assertEqual(command[:12], [
            "/opt/mcstas/bin/mcrun", "--force-compile", "--format=NeXus",
            "--seed=19", "--ncount=64", "--mpi=auto", "--openacc", "-g",
            "--dir={}".format(Path(directory) / "native"),
            str(Path.cwd() / "instrument.instr"), "-L", "-M"])
        self.assertEqual(command[12:], [
            r'fixed=literal\, value "q" \path', "x=-1,1", "y=2,4,6"])

    def test_paired_and_points_use_supplied_rows_only(self):
        plans = [
            ScanPlan.paired(x=[1, 2], y=[10, 20]),
            ScanPlan.points([{"x": 1, "y": 10}, {"x": 2, "y": 20}]),
        ]
        for plan in plans:
            command = build_scan_command(report(plan), "/tmp/instr.instr", "/tmp/native",
                                         seed=4)
            self.assertEqual(command[-2:], ["x=1,2", "y=10,20"])
            self.assertEqual(command[-3], "-L")

    def test_single_point_has_no_list_flags(self):
        plan = ScanPlan.points([{"x": np.float64(-2.5)}])
        command = build_scan_command(report(plan, fixed={"n": np.int64(3)}),
                                     "/tmp/instr.instr", "/tmp/native", seed="7")
        self.assertNotIn("-L", command)
        self.assertEqual(command[-2:], ["n=3", "x=-2.5"])

    def test_string_encoding_and_unsafe_values(self):
        plan = ScanPlan.points([{"label": r'a,b "q" \\path'},
                                {"label": r'other \\value'}])
        command = build_scan_command(report(plan), "/tmp/instr.instr", "/tmp/native", seed=1)
        self.assertEqual(command[-1], r'label=a\,b "q" \\path,other \\value')

        for value in ("", "has:colon", "line\nfeed", "nul\x00value"):
            bad = ScanPlan.points([{"label": value}, {"label": "ok"}])
            with self.assertRaises(ValueError):
                build_scan_command(report(bad), "/tmp/instr.instr", "/tmp/native", seed=1)

        single = ScanPlan.points([{"label": ""}])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_scan_command(report(single), "/tmp/instr.instr", "/tmp/native", seed=1)

        for value in tuple(chr(index) for index in range(32)) + ("\x7f", "\x85", "\u2028", "\u2029"):
            bad = ScanPlan.points([{"label": "prefix{}suffix".format(value)},
                                   {"label": "ok"}])
            with self.assertRaises(ValueError):
                build_scan_command(report(bad), "/tmp/instr.instr", "/tmp/native", seed=1)

        plan = ScanPlan.points([{"label": "ok"}])
        for instrument_file, native_path in (("instrument\t.instr", "/tmp/native"),
                                             ("/tmp/instr.instr", "native\u2028path")):
            with self.assertRaises(ValueError):
                build_scan_command(report(plan), instrument_file, native_path, seed=1)

    def test_list_backslash_and_metadata_edge_whitespace_are_rejected(self):
        for value in ("ends\\", " leading", "trailing "):
            plan = ScanPlan.points([{"label": value}, {"label": "ok"}])
            with self.assertRaisesRegex(ValueError, "metadata|end"):
                build_scan_command(report(plan), "/tmp/instr.instr", "/tmp/native", seed=1)

        single = ScanPlan.points([{"label": " leading"}])
        with self.assertRaisesRegex(ValueError, "metadata"):
            build_scan_command(report(single), "/tmp/instr.instr", "/tmp/native", seed=1)

    def test_empty_fixed_string_is_rejected_in_list_mode(self):
        plan = ScanPlan.points([{"label": "ok"}, {"label": "also ok"}])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_scan_command(report(plan, fixed={"category": ""}),
                               "/tmp/instr.instr", "/tmp/native", seed=1)

        single = ScanPlan.points([{"label": "ok"}])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_scan_command(report(single, fixed={"category": ""}),
                               "/tmp/instr.instr", "/tmp/native", seed=1)

    def test_seed_budget_covers_every_native_point(self):
        maximum = 2 ** 63 - 1
        two_points = ScanPlan.points([{"x": 1}, {"x": 2}])
        with self.assertRaisesRegex(ValueError, "seed.*budget"):
            build_scan_command(report(two_points), "/tmp/instr.instr", "/tmp/native",
                               seed=maximum - 1023)
        command = build_scan_command(
            report(two_points), "/tmp/instr.instr", "/tmp/native",
            seed=maximum - 1024)
        self.assertEqual(command[3], "--seed={}".format(maximum - 1024))

        many_points = ScanPlan.grid(x=range(3), y=range(2))
        with self.assertRaisesRegex(ValueError, "seed.*budget"):
            build_scan_command(report(many_points), "/tmp/instr.instr", "/tmp/native",
                               seed=maximum - 1023)

    def test_validation_does_not_probe_or_accept_unverified_tools(self):
        plan = ScanPlan.points([{"x": 1}])
        for package, version in (("McStas", "3.7.14"), ("McXtrace", "3.10.0"),
                                 ("McStas", "unknown")):
            with self.assertRaises(ValueError):
                build_scan_command(report(plan, package=package, version=version),
                                   "/tmp/instr.instr", "/tmp/native", seed=1)
        with self.assertRaises(ValueError):
            build_scan_command(report(plan, runner="mcrun"), "/tmp/instr.instr",
                               "/tmp/native", seed=1)

    def test_parameter_names_and_seed_are_validated(self):
        with self.assertRaises(ValueError):
            build_scan_command(report(ScanPlan.points([{"bad-name": 1}])),
                               "/tmp/instr.instr", "/tmp/native", seed=1)
        with self.assertRaises(ValueError):
            build_scan_command(report(ScanPlan.points([{"x": 1}])),
                               "/tmp/instr.instr", "/tmp/native", seed=0)


if __name__ == "__main__":
    unittest.main()
