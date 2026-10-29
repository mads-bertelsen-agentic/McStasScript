import json
import unittest

import numpy as np

from mcstasscript import ScanPlan
from mcstasscript.data.scan_plan import ScanPlanPoint


class TestScanPlan(unittest.TestCase):
    def test_grid_uses_last_axis_fastest_order_and_lazy_indexing(self):
        plan = ScanPlan.grid({"x": [1, 2], "mode": ["a", "b", "c"]})

        self.assertEqual(plan.kind, "grid")
        self.assertEqual(plan.parameter_names, ("x", "mode"))
        self.assertEqual(plan.axes, {"x": (1, 2), "mode": ("a", "b", "c")})
        self.assertEqual(len(plan), 6)
        self.assertEqual([(point.parameters, point.grid_indices) for point in plan], [
            ({"x": 1, "mode": "a"}, {"x": 0, "mode": 0}),
            ({"x": 1, "mode": "b"}, {"x": 0, "mode": 1}),
            ({"x": 1, "mode": "c"}, {"x": 0, "mode": 2}),
            ({"x": 2, "mode": "a"}, {"x": 1, "mode": 0}),
            ({"x": 2, "mode": "b"}, {"x": 1, "mode": 1}),
            ({"x": 2, "mode": "c"}, {"x": 1, "mode": 2}),
        ])
        self.assertIsInstance(plan[0], ScanPlanPoint)
        self.assertEqual(plan[0].point_id, 0)

    def test_grid_is_lazy_for_large_cartesian_product(self):
        plan = ScanPlan.grid(a=range(1000), b=range(1000), c=range(1000),
                             d=range(1000), e=range(1000))

        self.assertEqual(plan.point_count, 1000 ** 5)
        self.assertEqual(plan[1000].parameters, {"a": 0, "b": 0, "c": 0, "d": 1, "e": 0})

    def test_positional_mapping_and_parameter_named_parameters(self):
        plan = ScanPlan.grid({"x": [1]}, parameters=["a,b", '"quoted"', ""])
        self.assertEqual(plan.parameter_names, ("x", "parameters"))
        self.assertEqual(plan[0].parameters["parameters"], "a,b")
        self.assertEqual(plan.axes["parameters"], ("a,b", '"quoted"', ""))
        with self.assertRaises(ValueError):
            ScanPlan.grid({"x": [1]}, x=[2])

    def test_paired_axes_and_numpy_scalars_are_normalized(self):
        plan = ScanPlan.paired(theta=np.array([np.int64(2), np.int32(3), 4]),
                               mode=(np.float32(0.5), "", "last"))
        self.assertEqual(plan.axes, {"theta": (2, 3, 4), "mode": (0.5, "", "last")})
        self.assertEqual(plan.point_count, 3)
        self.assertEqual([point.point_id for point in plan], [0, 1, 2])
        self.assertEqual(plan[1].parameters, {"theta": 3, "mode": ""})
        self.assertIsNone(plan[1].grid_indices)
        with self.assertRaises(IndexError):
            plan[3]
        with self.assertRaises(ValueError):
            ScanPlan.paired(x=[1], y=[2, 3])

    def test_points_preserve_first_row_order_and_ignore_later_row_order(self):
        plan = ScanPlan.points([
            {"x": 1, "mode": "a"},
            {"mode": "b", "x": 2},
        ])
        self.assertEqual(plan.parameter_names, ("x", "mode"))
        self.assertEqual(plan[1].parameters, {"x": 2, "mode": "b"})
        self.assertIsNone(plan.axes)
        self.assertIsNone(plan[0].grid_indices)

    def test_points_reject_infinite_or_inconsistent_inputs(self):
        for rows in (iter([{"x": 1}]), [], [{}], [{"x": 1}, {"y": 2}], [{"x": [1]}]):
            with self.subTest(rows=rows), self.assertRaises((TypeError, ValueError)):
                ScanPlan.points(rows)

    def test_direct_constructor_validates_and_detaches_raw_inputs(self):
        with self.assertRaises(ValueError):
            ScanPlan("grid", parameter_names=("x",), axes={})
        with self.assertRaises(TypeError):
            ScanPlan("grid", parameter_names=("x",), axes={"x": iter([1, 2])})
        with self.assertRaises(ValueError):
            ScanPlan("paired", axes={"x": [1, 2], "y": [3]})
        with self.assertRaises(TypeError):
            ScanPlan("points", parameter_names=("x",), points=[{"x": None}])

        axes = {"x": [1, 2]}
        plan = ScanPlan("grid", axes=axes)
        axes["x"].append(3)
        self.assertEqual(plan.axes, {"x": (1, 2)})

    def test_invalid_axes_and_values(self):
        with self.assertRaises(ValueError):
            ScanPlan.grid()
        invalid_axes = ["abc", iter([1, 2]), {1, 2}, {"nested": [1]}, np.array([[1, 2]])]
        for axis in invalid_axes:
            with self.subTest(axis=axis), self.assertRaises((TypeError, ValueError)):
                ScanPlan.grid(x=axis)
        for value in (True, None, np.nan, np.inf, 1j, [1]):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                ScanPlan.grid(x=[value])
        for name in ("", 1):
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                ScanPlan.grid({name: [1]})

    def test_negative_and_out_of_range_indices_are_rejected(self):
        plan = ScanPlan.grid(x=[1, 2])
        for index in (-1, 2, np.int64(2)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                plan[index]
        with self.assertRaises(TypeError):
            plan[1.0]
        with self.assertRaises(TypeError):
            plan[:1]

    def test_snapshots_are_detached_and_read_only(self):
        source = {"x": [1, 2]}
        plan = ScanPlan.grid(source)
        source["x"].append(3)
        self.assertEqual(plan.axes, {"x": (1, 2)})
        with self.assertRaises(TypeError):
            plan.axes["x"] = (3,)
        point = plan[0]
        with self.assertRaises(TypeError):
            point.parameters["x"] = 9
        self.assertEqual(plan[0].parameters, {"x": 1})
        with self.assertRaises(AttributeError):
            point.point_id = 7

    def test_serialization_is_normalized_strict_and_json_safe(self):
        plan = ScanPlan.grid({"mode": ["a,b", '"quoted"', ""], "x": [1, 2.5]})
        serialized = plan.to_dict()
        self.assertEqual(serialized, {
            "kind": "grid",
            "parameter_names": ["mode", "x"],
            "axes": {"mode": ["a,b", '"quoted"', ""], "x": [1, 2.5]},
        })
        self.assertEqual(ScanPlan.from_dict(json.loads(json.dumps(serialized))), plan)
        serialized["axes"]["x"].append(99)
        self.assertEqual(plan.axes["x"], (1, 2.5))
        with self.assertRaises(ValueError):
            ScanPlan.from_dict({"kind": "grid", "parameter_names": ["x"],
                                "axes": {"x": [1]}, "extra": True})

        points = ScanPlan.points([{"x": 1, "y": "a"}, {"y": "b", "x": 2}])
        self.assertEqual(ScanPlan.from_dict(points.to_dict()), points)

    def test_preview_is_bounded_and_contains_indices(self):
        plan = ScanPlan.grid(x=range(1000), y=range(1000))
        preview = plan.preview(3)
        self.assertEqual(preview["point_count"], 1000000)
        self.assertEqual(preview["order"], "last_axis_fastest")
        self.assertEqual([point["grid_indices"] for point in preview["points"]], [
            {"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 0, "y": 2},
        ])
        self.assertEqual(len(preview["points"]), 3)
        self.assertEqual(preview["axes"], {
            "x": {"length": 1000, "values": [0, 1, 2], "truncated": True},
            "y": {"length": 1000, "values": [0, 1, 2], "truncated": True},
        })
        preview["axes"]["x"]["values"][0] = 99
        self.assertEqual(plan.axes["x"][0], 0)
        self.assertEqual(plan.preview(0)["points"], [])
        zero_preview = plan.preview(0)
        self.assertEqual(zero_preview["axes"]["x"],
                         {"length": 1000, "values": [], "truncated": True})
        self.assertIn("point_count=1000000", repr(plan))
        self.assertNotIn("999", repr(plan))
        with self.assertRaises(ValueError):
            plan.preview(-1)


if __name__ == "__main__":
    unittest.main()
