"""Explicit, conservative stacking of compatible binned scan monitors."""

from dataclasses import dataclass
import numbers

import numpy as np

from .data import McStasDataBinned
from .scan import ScanDataError
from .scan_plan import ScanPlan


@dataclass
class StackedScanMonitor:
    """Dense detector arrays, point identities and quantity-specific validity.

    Scan dimensions precede detector dimensions. ``coordinates`` includes scan
    axes and detector bin centres; ``detector_edges`` contains known limits.
    The original point datasets remain available from the originating scan.
    """

    Intensity: np.ndarray
    Error: np.ndarray
    Ncount: np.ndarray
    dims: tuple
    coordinates: dict
    point_ids: np.ndarray
    validity: dict
    detector_edges: dict
    estimated_bytes: int


def _detector(dataset, point_id):
    if not isinstance(dataset, McStasDataBinned):
        raise ValueError("Point {!r}: event or non-binned data cannot be stacked.".format(point_id))
    meta = dataset.metadata
    shape = np.asarray(dataset.Intensity).shape
    dimension = meta.dimension
    if isinstance(dimension, numbers.Integral) and dimension == 0:
        if np.asarray(dataset.Intensity).size != 1:
            raise ValueError("Point {!r}: invalid 0D detector shape.".format(point_id))
        shape = ()
        coords, edges = (), ()
    elif isinstance(dimension, numbers.Integral) and dimension > 0:
        if shape != (dimension,):
            raise ValueError("Point {!r}: invalid 1D detector shape.".format(point_id))
        axis = np.asarray(dataset.xaxis)
        if axis.shape != shape or axis.dtype.kind not in 'iuf' or not np.all(np.isfinite(axis)):
            raise ValueError("Point {!r}: invalid detector coordinates.".format(point_id))
        if meta.limits and (len(meta.limits) != 2 or not np.all(np.isfinite(meta.limits))):
            raise ValueError("Point {!r}: invalid detector limits.".format(point_id))
        coords = (axis.copy(),)
        edges = (tuple(meta.limits),)
    elif isinstance(dimension, (tuple, list)) and len(dimension) == 2:
        if shape != tuple(reversed(dimension)) or len(meta.limits) != 4:
            raise ValueError("Point {!r}: invalid 2D detector shape or limits.".format(point_id))
        xmin, xmax, ymin, ymax = meta.limits
        if not np.all(np.isfinite([xmin, xmax, ymin, ymax])):
            raise ValueError("Point {!r}: invalid detector limits.".format(point_id))
        # McStas stores 2D arrays in (y, x) order.
        xe = np.linspace(xmin, xmax, shape[1] + 1)
        ye = np.linspace(ymin, ymax, shape[0] + 1)
        coords = ((ye[:-1] + ye[1:]) / 2, (xe[:-1] + xe[1:]) / 2)
        edges = (tuple(ye), tuple(xe))
    else:
        raise ValueError("Point {!r}: unsupported detector dimensions.".format(point_id))
    for name in ("Intensity", "Error", "Ncount"):
        array = np.asarray(getattr(dataset, name))
        if array.dtype.kind not in 'biuf' or (array.shape != shape and not (shape == () and array.size == 1)):
            raise ValueError("Point {!r}: incompatible {} array.".format(point_id, name))
    # Labels and native unit metadata are evidence of compatibility, not invented units.
    info = meta.info
    units = tuple((key, info.get(key)) for key in
                  ("xlabel", "ylabel", "zlabel", "xvar", "yvar", "zvar", "unit", "units"))
    return shape, coords, edges, units


def stack_monitor(view, *, max_bytes=None):
    if max_bytes is not None and (isinstance(max_bytes, bool) or
                                  not isinstance(max_bytes, numbers.Integral) or max_bytes < 0):
        raise ValueError("max_bytes must be a nonnegative integer or None.")
    scan, monitor = view.scan, view.monitor
    plan = scan.plan
    grid = isinstance(plan, ScanPlan) and plan.kind == 'grid'
    if grid:
        scan_dims = tuple(plan.parameter_names)
        scan_shape = tuple(len(plan.axes[name]) for name in scan_dims)
        coordinates = {name: np.asarray(plan.axes[name]) for name in scan_dims}
    else:
        scan_dims, scan_shape = ('point',), (len(scan.points),)
        coordinates = {'point': np.arange(len(scan.points))}
    locations = []
    seen = set()
    for index, point in enumerate(scan.points):
        if grid:
            try:
                planned = plan[point.point_id]
            except (TypeError, IndexError) as error:
                raise ValueError("Point {!r}: not in the grid plan.".format(point.point_id)) from error
            if any(point.parameters.get(name) != value for name, value in planned.parameters.items()):
                raise ValueError("Point {!r}: parameters conflict with grid plan.".format(point.point_id))
            location = tuple(planned.grid_indices[name] for name in scan_dims)
        else:
            location = (index,)
        if location in seen:
            raise ValueError("Point {!r}: duplicate grid index.".format(point.point_id))
        seen.add(location)
        locations.append(location)

    loaded, reference = [], None
    for point, location in zip(scan.points, locations):
        try:
            data = point.data
        except ScanDataError:
            continue
        matches = [item for item in data if item.metadata.component_name == monitor.component
                   and item.metadata.filename == monitor.filename]
        if len(matches) > 1 or (matches and monitor.discriminator is not None):
            raise ValueError("Point {!r}: ambiguous monitor identity.".format(point.point_id))
        if not matches:
            continue
        item = matches[0]
        spec = _detector(item, point.point_id)
        if reference is None:
            reference = spec
        elif (spec[0] != reference[0] or spec[2:] != reference[2:] or
              any(not np.array_equal(a, b) for a, b in zip(spec[1], reference[1]))):
            raise ValueError("Point {!r}: detector shape, coordinates, limits or units differ.".format(point.point_id))
        loaded.append((point, location, item))
    if reference is None:
        raise ValueError("No readable binned data for monitor {!r}.".format(monitor))
    shape, detector_coords, edges, _ = reference
    detector_dims = []
    for axis in (('x',) if len(shape) == 1 else ('y', 'x') if len(shape) == 2 else ()):
        name = 'detector_' + axis
        while name in scan_dims or name in detector_dims:
            name += '_detector'
        detector_dims.append(name)
    detector_dims = tuple(detector_dims)
    dimensions = scan_shape + shape
    # Three float64 arrays and three bool masks; IDs are an object array.
    count = int(np.prod(dimensions, dtype=object))
    estimated = count * (3 * 8 + 3) + int(np.prod(scan_shape, dtype=object)) * np.dtype(object).itemsize
    if max_bytes is not None and estimated > max_bytes:
        raise MemoryError("Stack requires approximately {} bytes (limit {}).".format(estimated, max_bytes))
    arrays = {name: np.full(dimensions, np.nan) for name in ('Intensity', 'Error', 'Ncount')}
    validity = {name: np.zeros(dimensions, dtype=bool) for name in arrays}
    ids = np.full(scan_shape, None, dtype=object)
    for point, location in zip(scan.points, locations):
        ids[location] = point.point_id
    for point, location, item in loaded:
        available = point.array_availability.get(monitor, {})
        for name, array in arrays.items():
            value = np.asarray(getattr(item, name), dtype=float).reshape(shape)
            valid = np.isfinite(value) & (available.get(name) is True)
            array[location] = np.where(valid, value, np.nan)
            validity[name][location] = valid
    coordinates.update(zip(detector_dims, detector_coords))
    return StackedScanMonitor(arrays['Intensity'], arrays['Error'], arrays['Ncount'],
                              scan_dims + detector_dims, coordinates, ids, validity,
                              dict(zip(detector_dims, edges)), estimated)
