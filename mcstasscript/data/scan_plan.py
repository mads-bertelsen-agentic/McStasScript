"""Immutable, execution-independent scan plans."""

from collections.abc import Mapping
import math
import numbers
from types import MappingProxyType

import numpy as np


def _parameter_name(name):
    if not isinstance(name, str):
        raise TypeError("Scan parameter names must be strings.")
    if not name:
        raise ValueError("Scan parameter names must be nonempty.")
    return name


def _scalar(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Scan values cannot be boolean.")
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Scan values must be finite.")
        return result
    raise TypeError("Scan values must be finite integers, real numbers, or strings.")


def _axis_values(values):
    if isinstance(values, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise TypeError("Scan axes must be explicit list, tuple, range, or 1D ndarray values.")
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("Scan axis ndarrays must be one-dimensional.")
        values = values.tolist()
    elif isinstance(values, (list, tuple, range)):
        values = list(values)
    else:
        raise TypeError("Scan axes must be explicit list, tuple, range, or 1D ndarray values.")
    if not values:
        raise ValueError("Scan axes must be nonempty.")
    return tuple(_scalar(value) for value in values)


def _axis_items(mapping, parameters):
    if mapping is None:
        items = []
    elif not isinstance(mapping, Mapping):
        raise TypeError("The positional scan mapping must be a mapping.")
    else:
        items = list(mapping.items())

    mapping_names = {name for name, _ in items}
    duplicate = mapping_names.intersection(parameters)
    if duplicate:
        raise ValueError("Scan parameters cannot be supplied in both mapping and keywords: {}."
                         .format(sorted(duplicate)))
    items.extend(parameters.items())

    result = []
    seen = set()
    for name, values in items:
        _parameter_name(name)
        if name in seen:
            raise ValueError("Duplicate scan parameter name {!r}.".format(name))
        seen.add(name)
        result.append((name, _axis_values(values)))
    return tuple(result)


def _parameter_names(names):
    if isinstance(names, (str, bytes, bytearray)):
        raise TypeError("Parameter names must be a sequence, not a string.")
    if not isinstance(names, (list, tuple)):
        raise TypeError("Parameter names must be a list or tuple.")
    result = tuple(_parameter_name(name) for name in names)
    if len(set(result)) != len(result):
        raise ValueError("Scan parameter names must be unique.")
    return result


def _rows_values(rows, names=None):
    if not isinstance(rows, (list, tuple)):
        raise TypeError("Explicit scan rows must be a list or tuple.")
    if not rows:
        raise ValueError("Explicit scans must contain at least one row.")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("Each explicit scan row must be a mapping.")

    first_names = tuple(rows[0].keys())
    if names is None:
        names = _parameter_names(first_names)
    else:
        names = _parameter_names(names)
        if set(first_names) != set(names):
            raise ValueError("Explicit scan row keys do not match parameter_names.")
    if not names:
        raise ValueError("Explicit scans need at least one parameter.")

    normalized = []
    expected = set(names)
    for row in rows:
        if set(row) != expected:
            raise ValueError("All explicit scan rows must have the same keys.")
        normalized.append(tuple((name, _scalar(row[name])) for name in names))
    return names, tuple(normalized)


class ScanPlanPoint:
    """One lazily materialized point in a :class:`ScanPlan`."""

    __slots__ = ("_point_id", "_parameters", "_grid_indices")

    def __init__(self, point_id, parameters, grid_indices):
        object.__setattr__(self, "_point_id", int(point_id))
        object.__setattr__(self, "_parameters", tuple(parameters.items()))
        object.__setattr__(self, "_grid_indices",
                           None if grid_indices is None else tuple(grid_indices.items()))

    def __setattr__(self, name, value):
        raise AttributeError("ScanPlanPoint objects are immutable.")

    @property
    def point_id(self):
        return self._point_id

    @property
    def parameters(self):
        return MappingProxyType(dict(self._parameters))

    @property
    def grid_indices(self):
        if self._grid_indices is None:
            return None
        return MappingProxyType(dict(self._grid_indices))

    def __repr__(self):
        return "ScanPlanPoint(point_id={!r}, parameters={!r}, grid_indices={!r})".format(
            self.point_id, dict(self.parameters),
            None if self.grid_indices is None else dict(self.grid_indices))


class ScanPlan:
    """A validated grid, paired-axis, or explicit-point scan description.

    Grid points are ordered as ``itertools.product`` orders its inputs: the last
    axis changes fastest. Construction explicitly normalizes each supplied axis,
    so it is O(sum(axis lengths)); no Cartesian product is materialized by this
    class. :meth:`to_dict` is the complete normalized representation.
    """

    __slots__ = ("_kind", "_parameter_names", "_axes", "_rows", "_point_count")

    def __init__(self, kind, parameter_names=None, axes=None, points=None, rows=None):
        if kind not in ("grid", "paired", "points"):
            raise ValueError("Scan plan kind must be 'grid', 'paired', or 'points'.")
        if points is not None and rows is not None:
            raise TypeError("Use either points or rows, not both.")
        if points is not None:
            rows = points

        if kind in ("grid", "paired"):
            if rows is not None:
                raise ValueError("Grid and paired plans cannot contain explicit rows.")
            if not isinstance(axes, Mapping):
                raise TypeError("Grid and paired plans need an axes mapping.")
            if not axes:
                raise ValueError("Grid and paired plans need at least one axis.")
            axis_names = _parameter_names(tuple(axes))
            names = axis_names if parameter_names is None else _parameter_names(parameter_names)
            if set(names) != set(axis_names) or len(names) != len(axis_names):
                raise ValueError("parameter_names must match the axis names exactly.")
            normalized_axes = tuple((name, _axis_values(axes[name])) for name in names)
            if kind == "paired" and len({len(values) for _, values in normalized_axes}) != 1:
                raise ValueError("Paired scan axes must have equal lengths.")
            normalized_rows = None
            count = (len(normalized_axes[0][1]) if kind == "paired" else
                     math.prod(len(values) for _, values in normalized_axes))
        else:
            if axes is not None:
                raise ValueError("Point plans cannot contain axes.")
            names, normalized_rows = _rows_values(rows, parameter_names)
            normalized_axes = None
            count = len(normalized_rows)

        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_parameter_names", tuple(names))
        object.__setattr__(self, "_axes", normalized_axes)
        object.__setattr__(self, "_rows", normalized_rows)
        object.__setattr__(self, "_point_count", count)

    def __setattr__(self, name, value):
        raise AttributeError("ScanPlan objects are immutable.")

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @classmethod
    def _from_axes(cls, kind, mapping, parameters):
        items = _axis_items(mapping, parameters)
        if not items:
            raise ValueError("Scan plans need at least one axis.")
        names = tuple(name for name, _ in items)
        if kind == "paired" and items:
            lengths = {len(values) for _, values in items}
            if len(lengths) != 1:
                raise ValueError("Paired scan axes must have equal lengths.")
        return cls(kind, names, axes=dict(items))

    @classmethod
    def grid(cls, parameters=None, /, **axes):
        """Create a Cartesian grid from a positional mapping and/or keyword axes."""
        return cls._from_axes("grid", parameters, axes)

    @classmethod
    def paired(cls, parameters=None, /, **axes):
        """Create a pointwise paired scan from equal-length axes."""
        return cls._from_axes("paired", parameters, axes)

    @classmethod
    def _from_rows(cls, rows, names=None):
        return cls("points", parameter_names=names, points=rows)

    @classmethod
    def points(cls, rows):
        """Create a scan from a finite, explicitly ordered list or tuple of rows."""
        return cls._from_rows(rows)

    @classmethod
    def from_dict(cls, value):
        """Strictly restore a plan from :meth:`to_dict` output."""
        if not isinstance(value, Mapping):
            raise TypeError("A serialized scan plan must be a mapping.")
        if "kind" not in value or "parameter_names" not in value:
            raise ValueError("Serialized scan plans require kind and parameter_names.")
        kind = value["kind"]
        if kind not in ("grid", "paired", "points"):
            raise ValueError("Serialized scan plan has an invalid kind.")
        names = _parameter_names(value["parameter_names"])

        if kind in ("grid", "paired"):
            if set(value) != {"kind", "parameter_names", "axes"}:
                raise ValueError("Serialized axis plans must contain exactly kind, parameter_names, axes.")
            axes = value["axes"]
            if not isinstance(axes, Mapping) or set(axes) != set(names):
                raise ValueError("Serialized axes must match parameter_names.")
            ordered_axes = {name: axes[name] for name in names}
            return cls._from_axes(kind, ordered_axes, {})

        if set(value) != {"kind", "parameter_names", "rows"}:
            raise ValueError("Serialized point plans must contain exactly kind, parameter_names, rows.")
        return cls._from_rows(value["rows"], names=names)

    @property
    def kind(self):
        return self._kind

    @property
    def parameter_names(self):
        return self._parameter_names

    @property
    def axes(self):
        if self._axes is None:
            return None
        return MappingProxyType({name: tuple(values) for name, values in self._axes})

    @property
    def point_count(self):
        return self._point_count

    def __len__(self):
        return self.point_count

    def _grid_point(self, point_id):
        remainder = point_id
        indices = [0] * len(self._axes)
        for axis_number in range(len(self._axes) - 1, -1, -1):
            values = self._axes[axis_number][1]
            remainder, indices[axis_number] = divmod(remainder, len(values))
        parameters = {name: values[indices[index]]
                      for index, (name, values) in enumerate(self._axes)}
        grid_indices = {name: indices[index]
                        for index, (name, _) in enumerate(self._axes)}
        return ScanPlanPoint(point_id, parameters, grid_indices)

    def _paired_point(self, point_id):
        parameters = {name: values[point_id] for name, values in self._axes}
        return ScanPlanPoint(point_id, parameters, None)

    def __getitem__(self, point_id):
        if isinstance(point_id, bool) or not isinstance(point_id, numbers.Integral):
            raise TypeError("Scan plan indices must be integers.")
        point_id = int(point_id)
        if point_id < 0 or point_id >= self.point_count:
            raise IndexError("Scan plan point index out of range.")
        if self.kind == "grid":
            return self._grid_point(point_id)
        if self.kind == "paired":
            return self._paired_point(point_id)
        return ScanPlanPoint(point_id, dict(self._rows[point_id]), None)

    def __iter__(self):
        for point_id in range(self.point_count):
            yield self[point_id]

    def to_dict(self):
        """Return a compact, detached normalized representation."""
        result = {"kind": self.kind, "parameter_names": list(self.parameter_names)}
        if self._axes is not None:
            result["axes"] = {name: list(values) for name, values in self._axes}
        else:
            result["rows"] = [dict(row) for row in self._rows]
        return result

    def preview(self, max_points=5):
        """Return a bounded, JSON-safe summary without expanding a grid."""
        if isinstance(max_points, bool) or not isinstance(max_points, numbers.Integral):
            raise TypeError("max_points must be a nonnegative integer.")
        max_points = int(max_points)
        if max_points < 0:
            raise ValueError("max_points must be nonnegative.")

        result = {
            "kind": self.kind,
            "parameter_names": list(self.parameter_names),
            "point_count": self.point_count,
            "order": {"grid": "last_axis_fastest", "paired": "paired",
                      "points": "input"}[self.kind],
            "points": [],
        }
        if self._axes is not None:
            result["axes"] = {
                name: {
                    "length": len(values),
                    "values": list(values[:max_points]),
                    "truncated": len(values) > max_points,
                }
                for name, values in self._axes
            }
        for point in (self[index] for index in range(min(max_points, self.point_count))):
            result["points"].append({
                "point_id": point.point_id,
                "parameters": dict(point.parameters),
                "grid_indices": (None if point.grid_indices is None
                                  else dict(point.grid_indices)),
            })
        return result

    def __eq__(self, other):
        if not isinstance(other, ScanPlan):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return "ScanPlan(kind={!r}, point_count={!r}, parameter_names={!r})".format(
            self.kind, self.point_count, self.parameter_names)
