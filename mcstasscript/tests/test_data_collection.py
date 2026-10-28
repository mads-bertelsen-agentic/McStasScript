"""The new single-run result container retains ordinary list behavior."""

import copy
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mcstasscript import McStasDataCollection, load_data, name_plot_options, name_search
from mcstasscript.data.data import McStasDataBinned, McStasDataEvent, McStasMetaData
from mcstasscript.data.McStasDataFormat import McStasFormat
from mcstasscript.helper.managed_mcrun import ManagedMcrun, load_results
from mcstasscript.interface import plotter
from mcstasscript.tests.test_plot_interface import set_dummy_McStasDataBinned_1d


class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.first = set_dummy_McStasDataBinned_1d()
        self.first.name = "PSD"
        self.first.metadata.filename = "left.dat"
        self.second = set_dummy_McStasDataBinned_1d()
        self.second.name = "PSD"
        self.second.metadata.filename = "right.dat"
        self.data = McStasDataCollection([self.first, self.second])

    def test_list_construction_and_operations(self):
        self.assertIsInstance(self.data, list)
        self.assertEqual(self.data, [self.first, self.second])
        first, second = self.data
        self.assertIs(first, self.first)
        self.assertIs(second, self.second)
        self.assertIs(self.data[-1], self.second)
        for result in (self.data[:], self.data.copy(), self.data + [],
                       [] + self.data, self.data * 2, 2 * self.data):
            self.assertIs(type(result), list)
            self.assertIs(result[0], self.first)
        self.assertEqual(McStasDataCollection(), [])
        self.assertEqual(McStasDataCollection(iter(self.data)), self.data)

    def test_mutation_is_unrestricted_like_list(self):
        data = McStasDataCollection()
        data.append(3)
        data.extend([1, 2])
        data.insert(1, 4)
        data[0] = 5
        data[1:2] = [6, 7]
        del data[-1]
        data.remove(7)
        data += [8]
        data *= 2
        self.assertIsInstance(data, McStasDataCollection)
        self.assertEqual(data.count(8), 2)
        self.assertEqual(data.index(6), 1)
        self.assertEqual(data.pop(), 8)
        data.sort()
        data.reverse()
        self.assertEqual(data, [8, 6, 6, 5, 5, 1, 1])
        data.clear()
        self.assertFalse(data)

    def test_find_unique_and_ambiguous(self):
        self.assertIs(self.data.find("left.dat"), self.first)
        self.assertIs(self.data.find(filename="right.dat"), self.second)
        self.assertIs(self.data.find(component="PSD", filename="right.dat"), self.second)
        for selector in ({"name": "PSD"}, {"component": "PSD"}):
            with self.subTest(selector=selector), self.assertRaisesRegex(ValueError, "find_all"):
                self.data.find(**selector)
        with self.assertRaises(NameError):
            self.data.find("absent")
        with self.assertRaises(NameError):
            self.data.find(component="other", filename="left.dat")

    def test_find_all_preserves_identity_order_and_type(self):
        matches = self.data.find_all("PSD")
        self.assertIsInstance(matches, McStasDataCollection)
        self.assertIsNot(matches, self.data)
        self.assertEqual(matches, self.data)
        self.assertIs(matches[0], self.first)
        matches.clear()
        self.assertEqual(len(self.data), 2)
        self.assertEqual(self.data.find_all("absent"), [])
        self.assertIsInstance(McStasDataCollection().find_all("PSD"), McStasDataCollection)
        self.assertEqual(self.data.find_all(filename="right.dat"), [self.second])

    def test_name_precedence_and_explicit_filters(self):
        self.second.metadata.filename = "PSD"
        self.second.name = "other"
        self.assertIs(self.data.find("PSD"), self.first)
        self.assertIs(self.data.find(filename="PSD"), self.second)
        self.assertEqual(self.data.find_all(component="left.dat"), [])
        self.first.name = "renamed"
        self.assertIs(self.data.find("PSD"), self.second)
        self.assertIs(self.data.find(component="renamed"), self.first)
        self.data.pop()
        self.assertEqual(self.data.find_all("PSD"), [])
        self.first.metadata.filename = "changed.dat"
        self.assertIs(self.data.find("changed.dat"), self.first)

    def test_invalid_selectors(self):
        invalid = ({}, {"name": 1}, {"component": False}, {"filename": Path("left.dat")},
                   {"name": "PSD", "filename": "left.dat"},
                   {"name": "PSD", "component": "PSD"})
        for method in (self.data.find, self.data.find_all):
            for selector in invalid:
                with self.subTest(method=method.__name__, selector=selector):
                    with self.assertRaises(TypeError):
                        method(**selector)

    def test_copy_and_pickle(self):
        shallow = copy.copy(self.data)
        self.assertIsInstance(shallow, McStasDataCollection)
        self.assertIs(shallow[0], self.first)
        for restored in (copy.deepcopy(self.data), pickle.loads(pickle.dumps(self.data))):
            self.assertIsInstance(restored, McStasDataCollection)
            self.assertIsNot(restored[0], self.first)
            self.assertIs(restored.find("left.dat"), restored[0])
            np.testing.assert_array_equal(restored[0].Intensity, self.first.Intensity)
            restored[0].Intensity[0] = -1
            self.assertNotEqual(self.first.Intensity[0], -1)

    def test_legacy_search_and_plot_options(self):
        for container in (list, McStasDataCollection):
            with self.subTest(container=container):
                data = container(self.data)
                self.assertIs(name_search("left.dat", data), self.first)
                matches = name_search("PSD", data)
                self.assertIs(type(matches), list)
                self.assertEqual(matches, self.data)
                name_plot_options("PSD", data, colormap="Blues")
                self.assertTrue(all(d.plot_options.colormap == "Blues" for d in data))
                with self.assertRaises(NameError):
                    name_search("absent", data)
                with self.assertRaises(RuntimeError):
                    name_search("PSD", container())
                with self.assertRaises(RuntimeError):
                    name_search("PSD", container([1]))

    def test_plot_wrappers_forward_arguments_and_results(self):
        for method, target in ((self.data.plot, "make_sub_plot"),
                               (self.data.plot_individual, "make_plot")):
            with self.subTest(target=target), patch.object(plotter, target) as plot:
                result = method(return_fig_ax=True, log=True)
                plot.assert_called_once_with(self.data, return_fig_ax=True, log=True)
                self.assertIs(result, plot.return_value)

    def test_real_plotters_accept_collection_and_filter_events(self):
        metadata = McStasMetaData()
        metadata.component_name = "events"
        metadata.info["variables"] = "p x"
        event = McStasDataEvent(metadata, np.array([[1.0, 0.5]]))
        data = McStasDataCollection([self.first, event])
        for plot in (plotter.make_plot, plotter.make_sub_plot,
                     data.plot, data.plot_individual):
            with self.subTest(plot=plot.__name__):
                args = (data,) if plot in (plotter.make_plot, plotter.make_sub_plot) else ()
                fig, axes = plot(*args, return_fig_ax=True, log=False)
                self.addCleanup(plt.close, fig)
                ax = np.asarray(axes).reshape(-1)[0]
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), self.first.Intensity)
                self.assertFalse(self.first.plot_options.log)
                self.assertEqual(len(data), 2)


class TestCollectionLoading(unittest.TestCase):
    def test_ascii_producer_apis(self):
        folder = str(Path(__file__).parent / "test_data_set")
        runner = ManagedMcrun("unused.instr", output_path=folder, executable_path=".")
        for producer in (load_data, load_results, runner.load_results,
                         lambda path: McStasFormat.read(path)["data"]):
            with self.subTest(producer=producer):
                data = producer(folder)
                self.assertIsInstance(data, McStasDataCollection)
                self.assertIsInstance(data, list)
                self.assertEqual([d.name for d in data], ["PSD_4PI", "PSD", "L_mon", "monitor"])
                self.assertIs(data.find("L_mon"), data[2])
                self.assertIsInstance(data[3], McStasDataEvent)
                self.assertTrue(all(d.get_data_location() == folder for d in data))

    def test_nexus_loading(self):
        with TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "mccode.h5", "w") as output:
                entry = output.create_group("entry1")
                entry.create_group("simulation/Param").attrs["wavelength"] = 2.0
                monitor = entry.create_group("data/PSD")
                monitor.attrs.update(type="array_1d(3)", component="PSD", filename="PSD.dat",
                                     xlabel="Position", ylabel="Intensity", title="PSD monitor",
                                     xlimits="0 3", values="6 0.7 12")
                for field, values in (("Position", [0.5, 1.5, 2.5]),
                                      ("data", [1.0, 2.0, 3.0]),
                                      ("errors", [0.1, 0.2, 0.3]), ("ncount", [2, 4, 6])):
                    monitor.create_dataset(field, data=values)
            data = load_data(directory)
            self.assertIsInstance(data, McStasDataCollection)
            self.assertEqual(len(data), 1)
            monitor = data.find("PSD")
            self.assertIsInstance(monitor, McStasDataBinned)
            self.assertEqual(monitor.metadata.parameters, {"wavelength": 2.0})
            self.assertEqual(monitor.get_data_location(), directory)
            np.testing.assert_array_equal(monitor.Intensity, [1, 2, 3])
            np.testing.assert_array_equal(monitor.Error, [0.1, 0.2, 0.3])
            np.testing.assert_array_equal(monitor.Ncount, [2, 4, 6])
            np.testing.assert_array_equal(monitor.xaxis, [0.5, 1.5, 2.5])

    def test_empty_results_and_missing_output(self):
        with TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "mccode.h5", "w") as output:
                entry = output.create_group("entry1")
                entry.create_group("simulation/Param")
                entry.create_group("data")
            data = load_data(directory)
            self.assertIsInstance(data, McStasDataCollection)
            self.assertEqual(data, [])
            with patch("mcstasscript.helper.managed_mcrun.load_metadata", return_value=[]):
                (Path(directory) / "mccode.sim").touch()
                self.assertIsInstance(load_data(directory), McStasDataCollection)
                self.assertEqual(load_data(directory), [])
            missing = str(Path(directory) / "missing")
            runner = ManagedMcrun("unused.instr", output_path=missing, executable_path=".")
            with self.assertWarns(UserWarning):
                self.assertIsNone(runner.load_results())
            with self.assertRaises(RuntimeError):
                load_data(missing)
            with self.assertRaises(NameError):
                load_results(missing)
