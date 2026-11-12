import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mcstasscript.data.data import McStasDataBinned, McStasMetaData
from mcstasscript.data.scan import (ScanMonitor, ScanMonitorSummary, ScanPoint,
                                    ScanPointData, ScanResult)
from mcstasscript.data.scan_plan import ScanPlan
from mcstasscript import load_scan


MONITOR = ScanMonitor('PSD', 'psd.dat')


def point(point_id, parameters, values=(1., 2.), axis=(0., 1.), availability=None):
    meta = McStasMetaData()
    meta.component_name, meta.filename = 'PSD', 'psd.dat'
    meta.dimension, meta.limits = 2, [0., 2.]
    data = McStasDataBinned(meta, np.array(values), np.array(values),
                           np.array(values), xaxis=np.array(axis))
    report = ScanPointData([data], availability={MONITOR: availability or {
        'Intensity': True, 'Error': True, 'Ncount': True}})
    return ScanPoint(point_id, parameters, summaries=[ScanMonitorSummary(MONITOR)], data=report)


class StackTests(unittest.TestCase):
    def test_real_ascii_and_nexus(self):
        fixtures = Path(__file__).parent / 'scan_fixtures'
        for name in ('ascii_paired', 'nexus_paired'):
            with self.subTest(name=name):
                scan = load_scan(fixtures / name)
                stack = scan.dataset(scan.monitors[0]).stack()
                self.assertEqual(stack.Intensity.shape[0], len(scan.points))
                self.assertTrue(stack.validity['Intensity'].any())

    def test_scalar_and_2d(self):
        for dimension, shape, limits in ((0, (), []), ([2, 3], (3, 2), [0, 2, -1, 2])):
            with self.subTest(dimension=dimension):
                meta = McStasMetaData()
                meta.component_name, meta.filename = 'PSD', 'psd.dat'
                meta.dimension, meta.limits = dimension, limits
                data = np.ones(shape)
                item = McStasDataBinned(meta, data, data, data)
                p = ScanPoint(0, {}, summaries=[ScanMonitorSummary(MONITOR)],
                              data=ScanPointData([item], availability={MONITOR: {
                                  'Intensity': True, 'Error': True, 'Ncount': True}}))
                stack = ScanResult([p]).dataset('PSD').stack()
                self.assertEqual(stack.Intensity.shape, (1,) + shape)

    def test_grid_order_missing_and_availability(self):
        plan = ScanPlan.grid(theta=[0, 1], wave=[2, 3, 4])
        points = [point(i, plan[i].parameters) for i in (5, 0, 3)]
        points[0].array_availability[MONITOR]['Error'] = False
        stack = ScanResult(points, plan=plan).dataset('PSD').stack()
        self.assertEqual(stack.dims, ('theta', 'wave', 'detector_x'))
        self.assertEqual(stack.Intensity.shape, (2, 3, 2))
        self.assertEqual(stack.point_ids[1, 2], 5)
        self.assertIsNone(stack.point_ids[0, 1])
        self.assertTrue(np.isnan(stack.Intensity[0, 1]).all())
        self.assertTrue(np.isnan(stack.Error[1, 2]).all())
        self.assertFalse(stack.validity['Error'][1, 2].any())

    def test_incompatible_coordinates_and_memory_limit(self):
        scan = ScanResult([point(0, {}, axis=(0., 1.)), point(1, {}, axis=(0., 2.))])
        with self.assertRaisesRegex(ValueError, 'Point 1'):
            scan.dataset('PSD').stack()
        scan = ScanResult([point(0, {})])
        with self.assertRaises(MemoryError):
            scan.dataset('PSD').stack(max_bytes=1)

    def test_incompatible_units_and_shape(self):
        first, second = point(0, {}), point(1, {})
        first.data[0].metadata.info['unit'] = 'm'
        second.data[0].metadata.info['unit'] = 'cm'
        with self.assertRaisesRegex(ValueError, 'Point 1'):
            ScanResult([first, second]).dataset('PSD').stack()
        second = point(1, {}, values=(1., 2., 3.), axis=(0., 1., 2.))
        second.data[0].metadata.dimension = 3
        with self.assertRaisesRegex(ValueError, 'Point 1'):
            ScanResult([point(0, {}), second]).dataset('PSD').stack()

    def test_repeated_coordinates_remain_distinct(self):
        plan = ScanPlan.grid(theta=[1, 1], wave=[2, 3])
        scan = ScanResult([point(i, plan[i].parameters) for i in range(4)], plan=plan)
        stack = scan.dataset('PSD').stack()
        np.testing.assert_array_equal(stack.point_ids, [[0, 1], [2, 3]])

    def test_conflicting_grid_parameters_are_rejected(self):
        plan = ScanPlan.grid(theta=[0, 1])
        scan = ScanResult([point(0, {'theta': 1})], plan=plan)
        with self.assertRaisesRegex(ValueError, 'conflict'):
            scan.dataset('PSD').stack()

    def test_event_data_rejected(self):
        metadata = SimpleNamespace(component_name='PSD', filename='psd.dat')
        event = SimpleNamespace(metadata=metadata)
        p = ScanPoint(0, {}, summaries=[ScanMonitorSummary(MONITOR)], data=[event])
        with self.assertRaisesRegex(ValueError, 'event or non-binned'):
            ScanResult([p]).dataset('PSD').stack()

    def test_dimension_name_collision_is_disambiguated(self):
        plan = ScanPlan.grid(detector_x=[1])
        scan = ScanResult([point(0, plan[0].parameters)], plan=plan)
        self.assertEqual(scan.dataset('PSD').stack().dims,
                         ('detector_x', 'detector_x_detector'))

    def test_unknown_availability_does_not_invent_data(self):
        p = point(0, {})
        p.array_availability = {}
        stack = ScanResult([p]).dataset('PSD').stack()
        self.assertTrue(np.isnan(stack.Intensity).all())
        self.assertFalse(stack.validity['Intensity'].any())


if __name__ == '__main__':
    unittest.main()
