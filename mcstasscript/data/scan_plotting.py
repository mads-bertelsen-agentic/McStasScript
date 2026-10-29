"""Summary-only plotting for :mod:`mcstasscript.data.scan` records."""

from collections.abc import Mapping
import numbers

import numpy as np

from .scan import _TOTALS


_KINDS = ("auto", "line", "scatter", "heatmap")
_POINT_INDEX_NAME = "point_index"
_POINT_INDEX = object()


def _is_finite_number(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and bool(np.isfinite(value)))


def _coordinate_present(value):
    return value is not None and not (
        isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
        and not bool(np.isfinite(value)))


def _same_value(left, right):
    if left is None or right is None:
        return left is right
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, np.ndarray):
        return False
    return bool(result)


def _unique_values(values):
    unique = []
    for value in values:
        if not _coordinate_present(value):
            continue
        if not any(_same_value(value, previous) for previous in unique):
            unique.append(value)
    return unique


def _coordinate_type(values):
    present = [value for value in values if _coordinate_present(value)]
    if present and all(_is_finite_number(value) for value in present):
        return "numeric"
    return "categorical"


def _slice_indices(scan, where):
    if where is None:
        return np.arange(len(scan.points), dtype=int)
    if not isinstance(where, Mapping):
        raise TypeError("where must be a mapping of exact parameter values.")
    unknown = set(where) - set(scan.parameter_names)
    if unknown:
        raise ValueError("Unknown scan slice parameter(s): {}.".format(
            ", ".join(map(str, sorted(unknown, key=str)))))
    indices = []
    for index, point in enumerate(scan.points):
        if all(_same_value(point.parameters.get(name), expected)
               for name, expected in where.items()):
            indices.append(index)
    if not indices:
        raise ValueError("The requested scan slice is empty.")
    return np.asarray(indices, dtype=int)


def _varying_parameters(scan, indices):
    varying = []
    for name in scan.parameter_names:
        values = [scan.points[index].parameters.get(name) for index in indices]
        if len(_unique_values(values)) > 1:
            varying.append(name)
    return varying


def _validate_axis_name(name, parameter_names, label):
    if name is not None and not isinstance(name, str):
        raise TypeError("{} must be a parameter name, 'point_index', or None.".format(label))
    if name is None:
        return name
    if name == _POINT_INDEX_NAME and name not in parameter_names:
        return _POINT_INDEX
    if name not in parameter_names:
        raise ValueError("Unknown {} coordinate {!r}.".format(label, name))
    return name


def _resolve_axes(scan, x, y, varying, kind):
    x = _validate_axis_name(x, scan.parameter_names, "x")
    y = _validate_axis_name(y, scan.parameter_names, "y")
    if x == y and y is not None:
        raise ValueError("x and y must select different coordinates.")

    if y is not None:
        if x is None:
            candidates = [name for name in varying if name != y]
            x = candidates[0] if len(candidates) == 1 else _POINT_INDEX if not candidates else None
            if x is None:
                raise ValueError("Select x explicitly when more than one coordinate varies.")
        is_2d = True
    else:
        is_2d = False
        if x is None:
            if len(varying) == 1:
                x = varying[0]
            elif len(varying) > 1:
                raise ValueError("Multiple parameters vary; select x or x='point_index'.")
            else:
                x = _POINT_INDEX

    if len(varying) > 2 and _POINT_INDEX not in (x, y):
        raise ValueError("More than two parameters vary; slice the scan or use point_index.")
    if is_2d and kind == "line":
        raise ValueError("kind='line' requires a one-dimensional intensity plot.")
    return x, y, is_2d


def _axis_values(scan, indices, name):
    if name is _POINT_INDEX:
        return np.asarray(indices, dtype=float)
    return np.asarray([scan.points[index].parameters.get(name) for index in indices], dtype=object)


def _valid_coordinate_mask(values):
    return np.asarray([_coordinate_present(value) for value in values], dtype=bool)


def _axis_label(name):
    return _POINT_INDEX_NAME if name is _POINT_INDEX else name


def _category_axis_values(values, mask, axes, axis):
    values = np.asarray(values, dtype=object)
    mask = np.asarray(mask, dtype=bool)
    selected = [value for value, keep in zip(values, mask) if keep]
    if _coordinate_type(values) == "numeric":
        return values[mask]

    categories = []
    for value in selected:
        if not any(_same_value(value, category) for category in categories):
            categories.append(value)
    labels = []
    for category in categories:
        label = category if isinstance(category, str) else str(category)
        if label in labels:
            label = "{} ({})".format(label, type(category).__name__)
            while label in labels:
                label += "*"
        labels.append(label)
    positions = np.asarray([
        next(index for index, category in enumerate(categories)
             if _same_value(value, category))
        for value in selected
    ], dtype=float)
    if axis == "x":
        axes.set_xticks(np.arange(len(categories)))
        axes.set_xticklabels(labels)
    else:
        axes.set_yticks(np.arange(len(categories)))
        axes.set_yticklabels(labels)
    return positions


def _repeated_numeric(values, valid):
    if _coordinate_type(values) != "numeric":
        return False
    numeric = [float(value) for value, keep in zip(values, valid) if keep]
    return len(numeric) != len(set(numeric))


def _edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    result = np.empty(centers.size + 1, dtype=float)
    result[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    result[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    result[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return result


def _heatmap_data(x_values, y_values, intensities, valid):
    if (not all(_is_finite_number(value) for value in x_values) or
            not all(_is_finite_number(value) for value in y_values)):
        raise ValueError("Heatmap x and y coordinates must all be finite numbers.")
    records = [(float(x_value), float(y_value), float(intensity), bool(is_valid))
               for index, (x_value, y_value, intensity, is_valid) in enumerate(
                   zip(x_values, y_values, intensities, valid))]

    # A pair is an observation identity, including when its intensity is missing.
    pairs = [(x_value, y_value) for x_value, y_value, _, _ in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Heatmap coordinates must contain unique observed pairs.")
    if not records:
        raise ValueError("Heatmap requires finite numeric x and y coordinates.")

    x_centers = np.asarray(sorted({record[0] for record in records}), dtype=float)
    y_centers = np.asarray(sorted({record[1] for record in records}), dtype=float)
    x_positions = {value: index for index, value in enumerate(x_centers)}
    y_positions = {value: index for index, value in enumerate(y_centers)}
    intensity_grid = np.ma.masked_all((len(y_centers), len(x_centers)), dtype=float)
    for x_value, y_value, intensity, is_valid in records:
        if is_valid:
            intensity_grid[y_positions[y_value], x_positions[x_value]] = intensity
    return _edges(x_centers), _edges(y_centers), intensity_grid


def _heatmap(ax, data, kwargs):
    x_edges, y_edges, intensity_grid = data

    artist_kwargs = dict(kwargs)
    artist_kwargs.setdefault("shading", "flat")
    ax.pcolormesh(x_edges, y_edges, intensity_grid, **artist_kwargs)


def plot_overview(scan, monitor=None, value="total_I", x=None, y=None, ax=None, *,
                  component=None, filename=None, kind="auto", where=None, show=True,
                  **kwargs):
    """Plot scalar monitor totals and return ``(figure, axes)``.

    ``component`` and ``filename`` are monitor selectors passed through to
    :meth:`ScanResult.dataset`.  This function consumes only point parameters and
    detached monitor summaries; it never accesses ``ScanPoint.data``.
    """
    if not isinstance(value, str) or value not in _TOTALS:
        raise ValueError("value must be total_I, total_E, or total_N.")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("kind must be auto, line, scatter, or heatmap.")
    if ax is not None and not all(hasattr(ax, name) for name in ("figure", "scatter", "errorbar")):
        raise TypeError("ax must be a matplotlib Axes object.")
    if "yerr" in kwargs:
        raise TypeError("yerr is derived from the source total_E and cannot be overridden.")

    indices = _slice_indices(scan, where)
    varying = _varying_parameters(scan, indices)
    if kind == "heatmap" and (x is None or y is None):
        raise ValueError("kind='heatmap' requires explicit x and y coordinates.")
    x, y, is_2d = _resolve_axes(scan, x, y, varying, kind)
    if kind == "heatmap" and not is_2d:
        raise ValueError("kind='heatmap' requires explicit x and y coordinates.")

    view = scan.dataset(monitor, component=component, filename=filename)
    intensities = view.total_I if value == "total_I" else getattr(view, value)
    valid = view.validity(value)
    if value == "total_I":
        errors = view.total_E
        error_valid = view.validity("total_E") & np.isfinite(errors) & (errors >= 0)
    else:
        errors = np.full(len(scan.points), np.nan)
        error_valid = np.zeros(len(scan.points), dtype=bool)
    selected = np.asarray(indices, dtype=int)
    intensities = np.asarray(intensities)[selected]
    valid = np.asarray(valid, dtype=bool)[selected]
    errors = np.asarray(errors, dtype=float)[selected]
    error_valid = np.asarray(error_valid, dtype=bool)[selected] & valid
    x_values = _axis_values(scan, selected, x)
    heatmap_data = None

    if is_2d:
        y_values = _axis_values(scan, selected, y)
        render_kind = "scatter" if kind == "auto" else kind
        if render_kind == "heatmap":
            heatmap_data = _heatmap_data(x_values, y_values, intensities, valid)
    else:
        y_values = None
        coordinate_valid = _valid_coordinate_mask(x_values)
        render_kind = kind
        if kind == "auto":
            render_kind = ("scatter" if _coordinate_type(x_values) == "categorical" or
                           _repeated_numeric(x_values, coordinate_valid) else "line")

    # Import matplotlib only after selectors and all summary-side validation pass.
    import matplotlib.pyplot as plt

    figure = ax.figure if ax is not None else plt.subplots()[0]
    axes = ax if ax is not None else figure.axes[0]
    if is_2d:
        if render_kind == "heatmap":
            _heatmap(axes, heatmap_data, kwargs)
            axes.set_xlabel(_axis_label(x))
            axes.set_ylabel(_axis_label(y))
        else:
            coordinate_valid = (_valid_coordinate_mask(x_values) &
                                _valid_coordinate_mask(y_values))
            plot_mask = coordinate_valid & valid
            artist_kwargs = dict(kwargs)
            if "c" not in artist_kwargs and not any(
                    name in artist_kwargs for name in ("color", "facecolor", "facecolors")):
                artist_kwargs["c"] = intensities[plot_mask]
            plotted_x = _category_axis_values(x_values, plot_mask, axes, "x")
            plotted_y = _category_axis_values(y_values, plot_mask, axes, "y")
            axes.scatter(plotted_x, plotted_y, **artist_kwargs)
            axes.set_xlabel(_axis_label(x))
            axes.set_ylabel(_axis_label(y))
    elif render_kind == "line":
        coordinate_valid = _valid_coordinate_mask(x_values)
        if _coordinate_type(x_values) == "numeric":
            plot_x = np.asarray([
                float(value_) if keep else np.nan
                for value_, keep in zip(x_values, coordinate_valid)
            ])
            plot_valid = valid & coordinate_valid
            plot_y = np.where(plot_valid, intensities, np.nan)
            plot_errors = np.ma.masked_where(~(error_valid & coordinate_valid), errors)
            error_present = np.any(error_valid & coordinate_valid)
        else:
            plot_x = x_values[coordinate_valid]
            plot_valid = valid[coordinate_valid]
            plot_y = np.where(plot_valid, intensities[coordinate_valid], np.nan)
            plot_errors = np.ma.masked_where(~error_valid[coordinate_valid],
                                             errors[coordinate_valid])
            plot_x = _category_axis_values(
                plot_x, np.ones(len(plot_x), dtype=bool), axes, "x")
            error_present = np.any(error_valid[coordinate_valid])
        if value == "total_I" and error_present:
            axes.errorbar(plot_x, plot_y, yerr=plot_errors, **kwargs)
        else:
            axes.plot(plot_x, plot_y, **kwargs)
        axes.set_xlabel(_axis_label(x))
        axes.set_ylabel(value)
    else:
        coordinate_valid = _valid_coordinate_mask(x_values)
        plot_mask = coordinate_valid & valid
        plot_x = _category_axis_values(x_values, plot_mask, axes, "x")
        plot_y = intensities[plot_mask]
        plot_errors = np.ma.masked_where(~error_valid[plot_mask], errors[plot_mask])
        if value == "total_I" and np.any(error_valid[plot_mask]):
            axes.errorbar(plot_x, plot_y, yerr=plot_errors, fmt="none")
        axes.scatter(plot_x, plot_y, **kwargs)
        axes.set_xlabel(_axis_label(x))
        axes.set_ylabel(value)

    if show:
        plt.show(block=False)
    return figure, axes
