"""Native McStas scan output loading tests."""

from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcstasscript import McStasDataCollection, load_scan
from mcstasscript.data.scan import ScanDataError


FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "scan_fixtures" / "tiny_scan.instr"
MIN_MCRUN_VERSION = (3, 10, 0)
VERSION_PATTERN = re.compile(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?(?!\d)")


def _mcrun_version():
    result = subprocess.run(
        ["mcrun", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    match = VERSION_PATTERN.search(result.stdout or "")
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


class TestNativeScanLoading(unittest.TestCase):
    """Load scans produced by native mcrun, without using an execution API."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("mcrun") is None:
            raise unittest.SkipTest("mcrun is not available")
        version = _mcrun_version()
        if version is None:
            raise unittest.SkipTest("Could not parse mcrun --version")
        if version < MIN_MCRUN_VERSION:
            raise unittest.SkipTest(
                "Native scan tests require McStas >= 3.10.0, found {}.{}.{}".format(*version)
            )

    def _run_scan(self, workdir, output_name, format_name, scan_options):
        command = [
            "mcrun",
            "--force-compile",
            "--format={}".format(format_name),
            "--seed=314159",
            "--ncount=64",
            "--dir={}".format(output_name),
            "tiny_scan.instr",
            *scan_options,
        ]
        return subprocess.run(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def _assert_paired_scan(self, scan, expected_format):
        self.assertEqual(scan.metadata["format"], expected_format)
        self.assertEqual(scan.execution, {})
        self.assertEqual([point.status for point in scan.points], ["unknown"] * 3)
        self.assertEqual([point.metadata["seed"] for point in scan.points],
                         [314159, 315183, 316207])
        np.testing.assert_array_equal(scan.coordinates["theta"], [-1, 0, 1])
        np.testing.assert_array_equal(scan.coordinates["wavelength"], [2, 4, 6])

        monitor = scan.dataset("spectrum")
        np.testing.assert_array_equal(monitor.coordinates["theta"], [-1, 0, 1])
        np.testing.assert_array_equal(monitor.coordinates["wavelength"], [2, 4, 6])
        self.assertEqual(len(monitor.total_I), 3)
        self.assertTrue(np.all(np.isfinite(monitor.total_I)))
        self.assertTrue(np.all(monitor.total_I > 0))
        # Overview plotting must work before any full detector arrays are loaded.
        if not any(point.is_loaded for point in scan.points):
            with patch.object(type(scan.points[0]), "data", property(
                    lambda point: self.fail("Overview loaded detector arrays"))):
                figure, axes = scan.plot_overview("spectrum", x="theta", show=False)
            np.testing.assert_allclose(axes.lines[0].get_ydata(), monitor.total_I)
            plt.close(figure)
        for point, total in zip(scan.points, monitor.total_I):
            self.assertIn(monitor.monitor, point.summaries)
            self.assertAlmostEqual(point.summaries[monitor.monitor].total_I, total)
            data = point.data
            self.assertIsInstance(data, McStasDataCollection)
            self.assertIs(point.data, data)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0].name, "spectrum")
            self.assertTrue(np.all(np.isfinite(data[0].Intensity)))
            self.assertTrue(np.any(data[0].Intensity != 0))
            self.assertEqual(int(data[0].Ncount.sum()), 64)

    def _load_nexus_with_handle_checks(self, path):
        opened = []
        real_file = h5py.File

        def tracked_file(*args, **kwargs):
            handle = real_file(*args, **kwargs)
            opened.append(handle)
            return handle

        with patch("mcstasscript.data.scan_nexus.h5py.File", side_effect=tracked_file):
            scan = load_scan(path)
            cached_data = [point.data for point in scan.points]
            for point, data in zip(scan.points, cached_data):
                self.assertIs(point.data, data)

        self.assertTrue(opened)
        self.assertTrue(all(not handle.id.valid for handle in opened))
        for point, data in zip(scan.points, cached_data):
            self.assertIs(point.data, data)
        return scan

    def test_paired_numeric_ascii_and_nexus_scan_loading(self):
        with TemporaryDirectory(prefix="mcstasscript-scan-") as directory:
            workdir = Path(directory)
            shutil.copyfile(FIXTURE, workdir / "tiny_scan.instr")
            options = ["-L", "theta=-1,0,1", "wavelength=2,4,6"]

            ascii_result = self._run_scan(workdir, "ascii", "McCode", options)
            nexus_result = self._run_scan(workdir, "nexus", "NeXus", options)
            self.assertEqual(ascii_result.returncode, 0, msg=ascii_result.stdout)
            self.assertEqual(nexus_result.returncode, 0, msg=nexus_result.stdout)

            ascii_scan = load_scan(workdir / "ascii")
            ascii_file_scan = load_scan(workdir / "ascii" / "mccode.dat")
            nexus_scan = load_scan(workdir / "nexus")
            nexus_file_scan = self._load_nexus_with_handle_checks(
                workdir / "nexus" / "mccode.h5"
            )
            for scan, expected_format in (
                (ascii_scan, "ascii"),
                (ascii_file_scan, "ascii"),
                (nexus_scan, "nexus"),
                (nexus_file_scan, "nexus"),
            ):
                with self.subTest(format=expected_format, source=scan.metadata["source"]):
                    self._assert_paired_scan(scan, expected_format)

            for ascii_point, nexus_point in zip(ascii_scan.points, nexus_scan.points):
                for attribute in ("Intensity", "Error", "Ncount"):
                    np.testing.assert_allclose(
                        getattr(ascii_point.data[0], attribute),
                        getattr(nexus_point.data[0], attribute),
                        rtol=1e-5,
                        atol=0,
                    )

            self.assertEqual(nexus_scan.points[2].parameters["wavelength"], 6.0)
            self.assertFalse(np.array_equal(
                nexus_scan.points[2].data[0].Intensity,
                nexus_scan.points[0].data[0].Intensity,
            ))

    def _assert_partial_summary(self, scan, output_dir):
        summary_text = (output_dir / "mccode.dat").read_text(encoding="utf-8")
        rows = [line for line in summary_text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        self.assertIn("# Numpoints: 3", summary_text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(scan.metadata["declared_numpoints"], "3")
        self.assertTrue(any("Declared Numpoints differs" in diagnostic
                            for diagnostic in scan.metadata["diagnostics"]))

    def test_partial_numeric_scan_preserves_only_native_records(self):
        with TemporaryDirectory(prefix="mcstasscript-scan-partial-") as directory:
            workdir = Path(directory)
            shutil.copyfile(FIXTURE, workdir / "tiny_scan.instr")
            options = ["-L", "theta=0,1,2", "fail=0,1,0"]

            ascii_result = self._run_scan(workdir, "ascii_partial", "McCode", options)
            nexus_result = self._run_scan(workdir, "nexus_partial", "NeXus", options)
            # Native mcrun may return zero even when one scan point exits early.
            self.assertIn(ascii_result.returncode, (0, 1), msg=ascii_result.stdout)
            self.assertIn(nexus_result.returncode, (0, 1), msg=nexus_result.stdout)

            ascii_scan = load_scan(workdir / "ascii_partial")
            nexus_scan = load_scan(workdir / "nexus_partial")

            self._assert_partial_summary(ascii_scan, workdir / "ascii_partial")
            self._assert_partial_summary(nexus_scan, workdir / "nexus_partial")
            self.assertEqual(len(ascii_scan.points), 3)
            np.testing.assert_array_equal(ascii_scan.coordinates["theta"], [0, 1, 2])
            self.assertEqual([point.parameters["fail"] for point in ascii_scan.points], [0, 1, 0])
            self.assertEqual([point.status for point in ascii_scan.points], ["unknown"] * 3)
            self.assertEqual(len(ascii_scan.dataset("spectrum").total_I), 3)
            self.assertTrue(np.isfinite(ascii_scan.dataset("spectrum").total_I[[0, 2]]).all())
            with self.assertRaises(ScanDataError):
                _ = ascii_scan.points[1].data

            self.assertEqual(len(nexus_scan.points), 2)
            np.testing.assert_array_equal(nexus_scan.coordinates["theta"], [0, 2])
            self.assertEqual([point.parameters["fail"] for point in nexus_scan.points], [0, 0])
            self.assertEqual([point.status for point in nexus_scan.points], ["unknown", "unknown"])
            self.assertEqual([point.point_id for point in nexus_scan.points], [0, 1])
            self.assertNotIn(2, [point.point_id for point in nexus_scan.points])
            self.assertEqual(len(nexus_scan.dataset("spectrum").total_I), 2)
            self.assertTrue(np.isfinite(nexus_scan.dataset("spectrum").total_I).all())
            for point in nexus_scan.points:
                self.assertTrue(np.any(point.data[0].Intensity != 0))


if __name__ == "__main__":
    unittest.main()
