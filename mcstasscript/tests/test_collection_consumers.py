"""Regression coverage for list-compatible simulation result consumers."""

import copy
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from mcstasscript import McStasDataCollection
from mcstasscript.data.data import McStasDataBinned, McStasMetaData
from mcstasscript.instrument_diagnostics.beam_diagnostics import BeamDiagnostics
from mcstasscript.instrument_diagnostics.intensity_diagnostics import IntensityDiagnostics
from mcstasscript.jb_interface.plot_interface import PlotInterface
from mcstasscript.jb_interface.simulation_interface import add_data
from mcstasscript.tests.test_add_data import (
    set_dummy_McStasDataBinned_1d as make_live_1d,
    set_dummy_McStasDataBinned_2d as make_live_2d,
)
from mcstasscript.tests.test_dump_and_load import (
    setup_populated_instr_with_dummy_path,
    setup_populated_x_ray_instr_with_dummy_path,
)
from mcstasscript.tests.test_instrument_diagnostics import (
    make_dummy_event_data,
    setup_instr_with_dummy_path,
)
from mcstasscript.tests.test_plot_interface import (
    set_dummy_McStasDataBinned_1d as make_plot_1d,
)


INSTRUMENT_FACTORIES = (
    setup_populated_instr_with_dummy_path,
    setup_populated_x_ray_instr_with_dummy_path,
)


class TestCollectionConsumers(unittest.TestCase):
    def test_backengine_and_deprecated_wrapper_preserve_output_identity(self):
        for factory in INSTRUMENT_FACTORIES:
            for deprecated in (False, True):
                with self.subTest(factory=factory.__name__, deprecated=deprecated):
                    instr = factory()
                    instr.set_parameters(theta=1.25)
                    data = McStasDataCollection([make_plot_1d()])
                    with patch("mcstasscript.interface.instr.ManagedMcrun") as runner:
                        simulation = runner.return_value
                        simulation.simulation_succeeded = True
                        simulation.simulation_performed = True
                        simulation.load_results.return_value = data
                        with patch.object(instr, "write_full_instrument") as write:
                            warning = (
                                self.assertWarnsRegex(UserWarning, "run_full_instrument")
                                if deprecated else nullcontext()
                            )
                            with warning:
                                if deprecated:
                                    result = instr.run_full_instrument(
                                        parameters={"theta": 2.5}, ncount=1234
                                    )
                                else:
                                    result = instr.backengine()
                        write.assert_called_once_with()
                    simulation.run_simulation.assert_called_once_with()
                    simulation.load_results.assert_called_once_with()
                    self.assertIsInstance(result, McStasDataCollection)
                    self.assertIs(result, data)
                    self.assertIs(result[0], data[0])
                    stored = instr.output[instr.output_keys[0]].get_data()["data"]
                    self.assertIs(result, stored)
                    self.assertEqual(
                        runner.call_args.kwargs["parameters"]["theta"],
                        2.5 if deprecated else 1.25,
                    )
                    if deprecated:
                        self.assertEqual(runner.call_args.kwargs["ncount"], 1234)

    def test_dump_round_trip_with_collection_and_legacy_list(self):
        for factory in INSTRUMENT_FACTORIES:
            for container in (McStasDataCollection, list):
                with self.subTest(factory=factory.__name__, container=container):
                    instr = factory()
                    data = container([make_plot_1d(), make_live_2d()])
                    instr.output[instr.output_keys[0]].set_dict({"data": data})
                    with TemporaryDirectory() as directory:
                        filename = str(Path(directory) / "instrument.dmp")
                        instr.dump(filename)
                        restored = type(instr).from_dump(filename)
                    result = restored.output[restored.output_keys[0]].get_data()["data"]
                    self.assertIs(type(result), container)
                    self.assertEqual(restored.name, instr.name)
                    self.assertEqual(len(restored.component_list), 3)
                    self.assertEqual(len(result), 2)
                    for actual, expected in zip(result, data):
                        self.assertIsNot(actual, expected)
                        self.assertEqual(actual.name, expected.name)
                        self.assertEqual(actual.metadata.filename, expected.metadata.filename)
                        for field in ("Intensity", "Error", "Ncount"):
                            np.testing.assert_array_equal(
                                getattr(actual, field), getattr(expected, field)
                            )

    def test_plot_interface_selects_current_monitor_from_collection(self):
        interface = PlotInterface()
        widget = interface.show_interface()
        self.addCleanup(widget.close)
        self.addCleanup(plt.close, interface.fig)
        first, second = make_plot_1d(), make_plot_1d()
        second.name = "second monitor"
        second.Intensity += 100
        data = McStasDataCollection([first, second])
        interface.set_data(data)
        self.assertIs(interface.data, data)
        self.assertIs(interface.monitor_dropdown.data, data)
        self.assertEqual(tuple(interface.monitor_dropdown.widget.options),
                         (first.name, second.name))
        for monitor in (second, first):
            interface.monitor_dropdown.widget.value = monitor.name
            self.assertEqual(interface.current_monitor, monitor.name)
            np.testing.assert_array_equal(interface.ax.lines[0].get_xdata(), monitor.xaxis)
            np.testing.assert_array_equal(interface.ax.lines[0].get_ydata(), monitor.Intensity)

    def test_live_add_data_accepts_collection_and_list_combinations(self):
        for factory in (make_live_1d, make_live_2d):
            for initial_type in (McStasDataCollection, list):
                for source_type in (McStasDataCollection, list):
                    with self.subTest(factory=factory.__name__, initial=initial_type,
                                      source=source_type):
                        initial, source = factory(), factory()
                        initial.Intensity *= 2
                        initial.Error *= 1.5
                        initial.Ncount *= 4
                        initial.metadata.info["Ncount"] = 160
                        source.Intensity *= 3
                        source.Error *= 2
                        original = copy.deepcopy(source)
                        target, incoming = initial_type([initial]), source_type([source])
                        add_data(target, incoming)
                        self.assertIs(target[0], initial)
                        np.testing.assert_allclose(initial.Intensity, 2.2)
                        np.testing.assert_allclose(
                            initial.Error, np.sqrt((0.8 * 1.5) ** 2 + (0.2 * 2) ** 2)
                        )
                        np.testing.assert_array_equal(initial.Ncount, 5)
                        self.assertEqual(initial.metadata.info["Ncount"], 200)
                        self.assertIs(incoming[0], source)
                        self.assertEqual(source.metadata.info, original.metadata.info)
                        for field in ("Intensity", "Error", "Ncount"):
                            np.testing.assert_array_equal(
                                getattr(source, field), getattr(original, field)
                            )
                            self.assertFalse(np.shares_memory(
                                getattr(initial, field), getattr(source, field)
                            ))

    def test_beam_read_data_resolves_collection_filenames(self):
        diag = BeamDiagnostics(setup_instr_with_dummy_path())
        diag.add_point(before="first_component")
        diag.add_point(after="second_component")
        diag.add_point(before="third_component")
        diag.add_flag("two_theta")
        diag.add_monitors()
        data = McStasDataCollection()
        for point, count in zip(diag.ordered_point_list, (3, 7)):
            event_data = make_dummy_event_data(n_events=count)
            event_data.metadata.filename = point.filename
            event_data.metadata.total_N = count
            data.append(event_data)
        diag.data = data
        diag.read_data()
        self.assertEqual([point.recorded_rays for point in diag.ordered_point_list],
                         [3, 7, None])
        self.assertEqual(len(diag.event_plotters), 2)
        for plotter, event_data, point in zip(diag.event_plotters, data,
                                              diag.ordered_point_list):
            self.assertIs(plotter.data, event_data)
            self.assertEqual(plotter.name, point.filename)
            self.assertEqual(plotter.flag_info, ["two_theta"])
            self.assertEqual(len(plotter.data.Events), point.recorded_rays)

    def test_intensity_plot_0d_reads_nonzero_collection_values(self):
        diag = IntensityDiagnostics(setup_instr_with_dummy_path())
        diag.data = McStasDataCollection()
        diag.monitors = []
        for component, intensity, count in zip(diag.component_list[1:],
                                               (120.0, 30.0), (80, 20)):
            name = "I_before_" + component.name
            metadata = McStasMetaData()
            metadata.component_name = name
            metadata.dimension = 0
            metadata.total_I, metadata.total_N = intensity, count
            diag.data.append(McStasDataBinned(
                metadata, np.array([intensity]), np.array([1.0]), np.array([count])
            ))
            diag.monitors.append((name, component.name))
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        diag.plot_0D(fig=fig, ax=ax)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [30, 30, 120])
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_xdata(), [20, 20, 80])
        self.assertTrue(np.all(ax.lines[0].get_xdata() > 0))
        self.assertTrue(np.all(fig.axes[1].lines[0].get_xdata() > 0))


if __name__ == "__main__":
    unittest.main()
