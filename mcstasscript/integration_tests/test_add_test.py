import shutil
import subprocess
import tempfile
import unittest

from mcstasscript.interface.instr import McStas_instr


class TestAddTest(unittest.TestCase):
    """Integration test for generated McStas %Example tests."""

    def test_add_test_with_mctest(self):
        if shutil.which("mctest") is None:
            self.skipTest("mctest is not available")

        instrument_name = "integration_test_add_test"
        expected_intensity = 6.283185307179586e8

        with tempfile.TemporaryDirectory() as instrument_dir, \
                tempfile.TemporaryDirectory() as test_output:
            instrument = McStas_instr(instrument_name,
                                      input_path=instrument_dir)

            source = instrument.add_component("source", "Source_simple")
            source.radius = 0.01
            source.dist = 1
            source.focus_xw = 0.1
            source.focus_yh = 0.1
            source.E0 = 14
            source.dE = 1
            source.flux = 1e10

            monitor = instrument.add_component("monitor", "PSD_monitor")
            monitor.set_AT([0, 0, 1], RELATIVE="source")
            monitor.nx = 10
            monitor.ny = 10
            monitor.xmin = -0.05
            monitor.xmax = 0.05
            monitor.ymin = -0.05
            monitor.ymax = 0.05
            monitor.filename = '"monitor.dat"'

            instrument.add_test("monitor", intensity=expected_intensity)
            instrument.write_full_instrument()

            result = subprocess.run(
                ["mctest", "--local", instrument_dir,
                 "--testdir", test_output,
                 "--instr", instrument_name,
                 "--ncount", "10000", "--skipnontest"],
                cwd=instrument_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("SUCCESS", result.stdout)
        self.assertIn("[val:", result.stdout, msg=result.stdout)
