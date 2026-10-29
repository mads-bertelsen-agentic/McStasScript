import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcstasscript import ScanPoint, ScanResult
from mcstasscript.data.scan import ScanMonitor, ScanMonitorSummary


MONITOR = ScanMonitor("detector", "detector.dat")


def point(point_id, parameters, intensity=1.0, error=None, *, available=True,
          loader=None):
    availability = {} if available else {"total_E": False}
    summary = ScanMonitorSummary(MONITOR, total_I=intensity, total_E=error,
                                 availability=availability)
    return ScanPoint(point_id, parameters, summaries=[summary], loader=loader)


class TestScanPlotting(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_line_is_summary_only_and_keeps_missing_intensity_and_errors_masked(self):
        loader = Mock(side_effect=AssertionError("plotting must not load arrays"))
        scan = ScanResult([
            point(0, {"theta": 0.0}, 1.0, 0.0, loader=loader),
            point(1, {"theta": 1.0}, None, None, available=False, loader=loader),
            point(2, {"theta": 2.0}, 3.0, 0.5, loader=loader),
        ])

        figure, axes = scan.plot_overview("detector", show=False)

        np.testing.assert_array_equal(axes.lines[0].get_xdata(), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0, np.nan, 3.0],
                                   equal_nan=True)
        errorbar = axes.containers[0]
        self.assertEqual(errorbar.lines[2][0].get_segments()[0].tolist(),
                         [[0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(loader.call_count, 0)
        self.assertIs(figure, axes.figure)

    def test_zero_error_is_retained_but_unavailable_error_is_not_zero_filled(self):
        scan = ScanResult([
            point(0, {"theta": 0}, 1.0, 0.0),
            point(1, {"theta": 1}, 2.0, None, available=False),
        ])
        _, axes = scan.plot_overview("detector", show=False)
        segments = axes.containers[0].lines[2][0].get_segments()
        self.assertEqual(len([segment for segment in segments if len(segment)]), 1)
        self.assertEqual(segments[0].tolist(), [[0.0, 1.0], [0.0, 1.0]])

    def test_invalid_intensity_is_masked_and_non_intensity_values_have_no_errors(self):
        scan = ScanResult([
            point(0, {"theta": 0}, np.inf, 0.5),
            point(1, {"theta": 1}, 2.0, 0.5),
        ])
        _, axes = scan.plot_overview("detector", show=False)
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [np.nan, 2.0],
                                   equal_nan=True)

        _, axes = scan.plot_overview("detector", value="total_N", show=False)
        self.assertEqual(len(axes.containers), 0)

    def test_one_dimensional_scatter_displays_only_valid_total_i_errors(self):
        scan = ScanResult([
            point(0, {"theta": 0}, 1.0, 0.0),
            point(1, {"theta": 0}, 2.0, 0.5),
            point(2, {"theta": 1}, 3.0, None, available=False),
        ])
        _, axes = scan.plot_overview("detector", show=False)
        self.assertEqual(len(axes.containers), 1)
        self.assertEqual(len(axes.collections[-1].get_offsets()), 3)

    def test_repeated_numeric_coordinates_default_to_scatter(self):
        scan = ScanResult([point(0, {"theta": 2}, 1), point(1, {"theta": 2}, 3),
                           point(2, {"theta": 3}, 4)])
        _, axes = scan.plot_overview("detector", x="theta", show=False)
        self.assertEqual(len(axes.collections), 1)
        np.testing.assert_array_equal(axes.collections[0].get_offsets()[:, 0], [2, 2, 3])

    def test_categorical_coordinates_are_labeled_and_repetitions_remain(self):
        scan = ScanResult([
            point(0, {"mode": "cold"}, 1),
            point(1, {"mode": "hot"}, 2),
            point(2, {"mode": "cold"}, 3),
        ])
        _, axes = scan.plot_overview("detector", show=False)
        self.assertEqual(len(axes.collections[0].get_offsets()), 3)
        self.assertEqual({label.get_text() for label in axes.get_xticklabels()},
                         {"cold", "hot"})

    def test_mixed_numeric_and_string_coordinates_use_explicit_ticks(self):
        scan = ScanResult([
            point(0, {"mode": 1}, 1),
            point(1, {"mode": "one"}, 2),
            point(2, {"mode": 1}, 3),
        ])
        _, axes = scan.plot_overview("detector", show=False)
        np.testing.assert_array_equal(axes.collections[0].get_offsets()[:, 0], [0, 1, 0])
        self.assertEqual([label.get_text() for label in axes.get_xticklabels()],
                         ["1", "one"])

    def test_explicit_two_dimensional_scatter_does_not_infer_heatmap(self):
        scan = ScanResult([
            point(0, {"x": 0, "y": 0}, 1),
            point(1, {"x": 1, "y": 0}, 2),
            point(2, {"x": 0, "y": 1}, 3),
        ])
        _, axes = scan.plot_overview("detector", x="x", y="y", show=False)
        self.assertEqual(len(axes.collections), 1)
        self.assertFalse(axes.collections[0].__class__.__name__ == "QuadMesh")
        self.assertEqual(len(axes.collections[0].get_offsets()), 3)

    def test_explicit_heatmap_masks_unobserved_and_missing_intensity_cells(self):
        scan = ScanResult([
            point(0, {"x": 0, "y": 0}, 1),
            point(1, {"x": 2, "y": 0}, None, available=False),
            point(2, {"x": 0, "y": 3}, 3),
        ])
        _, axes = scan.plot_overview("detector", x="x", y="y", kind="heatmap",
                                     show=False)
        mesh = axes.collections[0]
        np.testing.assert_array_equal(np.ma.getmaskarray(mesh.get_array()),
                                      [[False, True], [False, True]])
        np.testing.assert_allclose(mesh.get_coordinates()[0, :, 0], [-1.0, 1.0, 3.0])
        np.testing.assert_allclose(mesh.get_coordinates()[:, 0, 1], [-1.5, 1.5, 4.5])

    def test_heatmap_rejects_any_invalid_coordinate_before_creating_figure(self):
        for coordinates in (({"x": 0, "y": "zero"}, {"x": 1, "y": 1}),
                            ({"x": 0, "y": 0}, {"x": None, "y": 1}),
                            ({"x": 0, "y": 0}, {"x": np.inf, "y": 1})):
            with self.subTest(coordinates=coordinates):
                scan = ScanResult([point(index, values)
                                   for index, values in enumerate(coordinates)])
                with patch("matplotlib.pyplot.subplots") as subplots:
                    with self.assertRaises(ValueError):
                        scan.plot_overview("detector", x="x", y="y", kind="heatmap",
                                           show=False)
                subplots.assert_not_called()

    def test_heatmap_rejects_repeated_coordinate_pairs_and_nonfinite_coordinates(self):
        duplicate = ScanResult([
            point(0, {"x": 0, "y": 0}), point(1, {"x": 0.0, "y": 0.0})
        ])
        with self.assertRaises(ValueError):
            duplicate.plot_overview("detector", x="x", y="y", kind="heatmap", show=False)
        nonfinite = ScanResult([point(0, {"x": np.nan, "y": 0})])
        with self.assertRaises(ValueError):
            nonfinite.plot_overview("detector", x="x", y="y", kind="heatmap", show=False)

    def test_where_is_exact_nonmutating_and_rejects_unknown_or_empty_slices(self):
        scan = ScanResult([
            point(0, {"x": 0, "mode": "a"}), point(1, {"x": 1, "mode": "a"}),
            point(2, {"x": 2, "mode": "b"}),
        ])
        _, axes = scan.plot_overview("detector", x="x", where={"mode": "a"}, show=False)
        np.testing.assert_array_equal(axes.lines[0].get_xdata(), [0, 1])
        self.assertEqual(scan.points[2].parameters["mode"], "b")
        for where in ({"unknown": 1}, {"mode": "missing"}):
            with self.subTest(where=where), self.assertRaises(ValueError):
                scan.plot_overview("detector", where=where, show=False)

    def test_three_dimensions_require_slice_or_point_index(self):
        scan = ScanResult([point(i, {"x": i, "y": i % 2, "z": i // 2})
                           for i in range(4)])
        with self.assertRaises(ValueError):
            scan.plot_overview("detector", x="x", show=False)
        _, axes = scan.plot_overview("detector", x="x", where={"z": 0}, show=False)
        self.assertEqual(len(axes.lines), 1)
        _, axes = scan.plot_overview("detector", x="point_index", show=False)
        self.assertEqual(len(axes.lines), 1)

    def test_real_point_index_parameter_precedes_synthetic_index_only_when_explicit(self):
        scan = ScanResult([point(0, {"point_index": 100, "fixed": 7}, 1),
                           point(1, {"point_index": 100, "fixed": 7}, 2)])
        _, axes = scan.plot_overview("detector", show=False)
        np.testing.assert_array_equal(axes.lines[0].get_xdata(), [0.0, 1.0])

        _, axes = scan.plot_overview("detector", x="point_index", show=False)
        np.testing.assert_array_equal(axes.collections[0].get_offsets()[:, 0], [100, 100])

    def test_numeric_line_keeps_missing_coordinate_gaps(self):
        scan = ScanResult([point(0, {"theta": 0}, 1), point(1, {}, 2),
                           point(2, {"theta": 2}, 3)])
        _, axes = scan.plot_overview("detector", x="theta", kind="line", show=False)
        np.testing.assert_allclose(axes.lines[0].get_xdata(), [0.0, np.nan, 2.0],
                                   equal_nan=True)
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.0, np.nan, 3.0],
                                   equal_nan=True)

    def test_explicit_x_allows_paired_projection_and_preserves_repeats(self):
        scan = ScanResult([point(0, {"x": 0, "pair": "a"}, 1),
                           point(1, {"x": 0, "pair": "b"}, 2),
                           point(2, {"x": 1, "pair": "a"}, 3)])
        _, axes = scan.plot_overview("detector", x="x", kind="scatter", show=False)
        np.testing.assert_array_equal(axes.collections[0].get_offsets()[:, 0], [0, 0, 1])

    def test_axis_reuse_filename_and_display_control(self):
        scan = ScanResult([point(0, {"theta": 0}), point(1, {"theta": 1})])
        figure, axes = plt.subplots()
        with patch("matplotlib.pyplot.show") as show:
            result_figure, result_axes = scan.plot_overview("detector", ax=axes,
                                                            show=False)
        self.assertIs(result_figure, figure)
        self.assertIs(result_axes, axes)
        show.assert_not_called()
        with patch("matplotlib.pyplot.show") as show:
            scan.plot_overview("detector", show=True)
        show.assert_called_once_with(block=False)

    def test_filename_is_a_monitor_selector(self):
        scan = ScanResult([point(0, {"theta": 0})])
        _, axes = scan.plot_overview(filename="detector.dat", show=False)
        self.assertEqual(len(axes.lines), 1)
        with self.assertRaises(TypeError):
            scan.plot_overview("detector", filename="detector.dat", show=False)

    def test_invalid_arguments_are_rejected(self):
        scan = ScanResult([point(0, {"theta": 0})])
        for kwargs in ({"value": "Intensity"}, {"kind": "bars"},
                       {"kind": "heatmap"}, {"x": "missing"}, {"y": "missing"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                scan.plot_overview("detector", show=False, **kwargs)


if __name__ == "__main__":
    unittest.main()
