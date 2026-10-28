"""Single-run collection propagation with a real McStas simulation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mcstasscript import McStasDataCollection, McStas_instr, load_data, name_search


class TestDataCollection(unittest.TestCase):
    def test_run_and_reload_collection(self):
        with TemporaryDirectory() as directory:
            instrument = McStas_instr("collection_smoke", input_path=directory)
            source = instrument.add_component("source", "Source_div")
            source.set_parameters(xwidth=0.03, yheight=0.01, focus_aw=0.01,
                                  focus_ah=0.01, E0=81.81, dE=1.0, flux=1e10)
            monitor = instrument.add_component("PSD", "PSDlin_monitor")
            monitor.set_parameters(xwidth=0.1, yheight=0.03, filename='"PSD.dat"')
            if instrument.mccode_version > 2:
                monitor.nbins = 10
            else:
                monitor.nx = 10
            monitor.set_AT([0, 0, 1], RELATIVE="source")
            output_path = str(Path(directory) / "output")
            instrument.settings(output_path=output_path, ncount=10000,
                                seed=12345, suppress_output=True)
            data = instrument.backengine()
            self.assertIsInstance(data, McStasDataCollection)
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            self.assertIs(data.find("PSD"), data[0])
            self.assertIs(name_search("PSD.dat", data), data[0])
            self.assertIs(instrument.output[instrument.output_keys[0]].get_data()["data"], data)
            self.assertGreater(data[0].Intensity.sum(), 0)
            reloaded = load_data(output_path)
            self.assertIsInstance(reloaded, McStasDataCollection)
            np.testing.assert_array_equal(reloaded[0].Intensity, data[0].Intensity)
            np.testing.assert_array_equal(reloaded[0].Error, data[0].Error)
