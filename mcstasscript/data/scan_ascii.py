"""Execution-free native ASCII scan reader and detached metadata parsing."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re
import shlex

import numpy as np

from mcstasscript.data.data import McStasMetaData
from mcstasscript.helper import managed_mcrun

from .scan import (ScanDataError, ScanMonitor, ScanMonitorSummary, ScanPoint,
                   ScanPointData, ScanResult, ScanSource, _TOTALS)


class ScanFormatError(ValueError):
    """Native output cannot be interpreted without guessing."""


@dataclass
class _SummaryRow:
    number: int
    raw: str
    coordinates: dict
    totals: dict
    encoded_point_id: object = None
    encoded_parameters: tuple = ()


@dataclass
class _AsciiSummary:
    path: Path
    headers: dict
    columns: tuple
    parameter_names: tuple
    rows: list


def _read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ScanFormatError("Invalid UTF-8 text in {}.".format(path)) from error


def _headers(text):
    result = {}
    for line in text.splitlines():
        key, colon, value = line.strip().lstrip("#").strip().partition(":")
        if colon:
            result.setdefault(key.lower(), []).append(value.strip())
    return result


def _last(headers, name, default=""):
    return headers.get(name, [default])[-1]


def _names(value):
    return tuple(re.findall(r"[A-Za-z_]\w*", value))


def _parameter_types(declaration):
    return dict(re.findall(r"(\w+)\s*\(\s*([^()]+)\s*\)", declaration))


def _unescape_native_string(value):
    """Decode the quote and backslash escaping used by native McCode output."""
    result = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in '\\"':
            result.append(value[index + 1])
            index += 2
        else:
            result.append(value[index])
            index += 1
    return "".join(result)


def _parse_value(value, declared_type=None):
    value = str(value).strip()
    # Double quotes are the native wrapper used by legacy output; apostrophes
    # can be literal string data in current native metadata.
    quoted = len(value) >= 2 and value[0] == value[-1] == '"'
    if quoted:
        value = value[1:-1]
    if declared_type in ("string", "char*") or (quoted and declared_type is None):
        return _unescape_native_string(value)
    try:
        if declared_type == "int":
            return int(value)
        if declared_type in ("double", "float", "MCNUM"):
            return float(value)
        if re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        return float(value)
    except ValueError as error:
        if declared_type in ("int", "double", "float", "MCNUM"):
            raise ScanFormatError("Invalid {} parameter {!r}.".format(declared_type, value)) from error
        return value


def _metadata_record(info, parameters):
    """Extract ordinary monitor metadata without legacy zero-fill or Param splitting."""
    metadata = McStasMetaData()
    metadata.info = dict(info, Parameters=dict(parameters))
    values = metadata.info.pop("values", None)
    metadata.info.setdefault("filename", "")
    try:
        metadata.extract_info()
        if (metadata.dimension is None or metadata.dimension == [] or
                (isinstance(metadata.dimension, int) and metadata.dimension < 0)):
            raise ValueError("Unsupported monitor type {!r}".format(info.get("type")))
        if values is not None:
            tokens = str(values).split()
            if not 1 <= len(tokens) <= 3:
                raise ValueError("Expected one to three monitor totals")
            metadata.info["values"] = values
            for quantity, value in zip(_TOTALS, tokens):
                setattr(metadata, quantity, float(value))
        if not metadata.component_name:
            raise ValueError("Missing component name")
    except (ValueError, NameError, IndexError, TypeError, AttributeError) as error:
        raise ScanFormatError("Malformed monitor metadata: {}".format(error)) from error
    return metadata


def _summary_from_metadata(metadata, source, monitor):
    totals = {name: getattr(metadata, name) for name in _TOTALS}
    return ScanMonitorSummary(
        monitor, **totals,
        availability={name: value is not None for name, value in totals.items()},
        provenance={name: str(source) + ":values" for name, value in totals.items()
                    if value is not None},
        metadata={"native": dict(metadata.info)},
    )


def _point_records(point_path):
    path = point_path / "mccode.sim"
    text = _read_text(path)
    types = _parameter_types(_last(_headers(text), "parameters"))
    parameters, simulation, records, info = {}, {}, [], None
    in_simulation = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("begin simulation"):
            in_simulation = True
        elif line.startswith("end simulation"):
            in_simulation = False
        elif line == "begin data":
            if info is not None:
                raise ScanFormatError("Nested data section in {}.".format(path))
            info = {}
        elif line == "end data":
            if info is None:
                raise ScanFormatError("Unmatched end data in {}.".format(path))
            records.append(_metadata_record(info, parameters))
            info = None
        elif ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if in_simulation:
                if key == "Param":
                    name, equals, raw = value.partition("=")
                    if not equals:
                        raise ScanFormatError("Malformed Param in {}.".format(path))
                    name = name.strip()
                    parameters[name] = _parse_value(raw, types.get(name))
                else:
                    simulation[key] = value
            elif info is not None:
                info[key] = value
    if info is not None:
        raise ScanFormatError("Incomplete data section in {}.".format(path))
    summaries, inventory = [], {}
    for metadata in records:
        monitor = ScanMonitor(metadata.component_name, metadata.filename or None)
        if monitor in inventory:
            raise ScanFormatError("Duplicate monitor identity {} in {}.".format(monitor, path))
        inventory[monitor] = metadata
        summaries.append(_summary_from_metadata(metadata, path, monitor))
    return ScanPoint(
        int(point_path.name) if point_path.name.isdecimal() else 0,
        parameters, ScanSource(point_path), summaries=summaries,
        metadata={"native": simulation, "seed": _parse_value(simulation["Seed"], "int")
                  if "Seed" in simulation else None, "parameter_types": types},
        loader=partial(_load_point_data, inventory=inventory),
    )


def _validate_data(dataset, availability):
    quantities = ("Events",) if dataset.data_type == "Events" else ("Intensity", "Error", "Ncount")
    if dataset.data_type == "Binned 1D":
        quantities += ("xaxis",)
    for name in quantities:
        array = getattr(dataset, name)
        if dataset.data_type == "Events":
            valid_shape = array.ndim == 2
        elif dataset.metadata.dimension == 0:
            valid_shape = array.size == 1
        else:
            dimension = dataset.metadata.dimension
            expected = (dimension,) if isinstance(dimension, int) else tuple(reversed(dimension))
            valid_shape = array.shape == expected
        if not valid_shape or array.dtype.kind not in "biuf":
            availability[name] = False
            raise ValueError("Malformed {} array: shape {}, dtype {}".format(
                name, array.shape, array.dtype))


def _load_point_data(source, inventory):
    data, diagnostics, availability = [], {}, {}
    missing = True
    for monitor, metadata in inventory.items():
        if not monitor.filename:
            continue
        path = source.path / monitor.filename
        missing = missing and not path.exists()
        try:
            dataset = managed_mcrun.load_monitor_text(metadata, str(source.path))
            if dataset.data_type == "Events":
                available = {"Events": True}
            else:
                available = {"Intensity": True, "Error": True, "Ncount": True}
                if metadata.dimension == 0:
                    available["Error"] = metadata.total_E is not None
                    available["Ncount"] = metadata.total_N is not None
            availability[monitor] = available
            _validate_data(dataset, available)
            dataset.set_data_location(str(source.path))
            data.append(dataset)
            availability[monitor] = available
        except (OSError, ValueError, IndexError, NameError) as error:
            diagnostics[monitor] = str(path) + ": " + str(error)
            availability.setdefault(monitor, dict.fromkeys(("Intensity", "Error", "Ncount"), False))
    if not data and missing:
        raise ScanDataError("No full monitor output in {}.".format(source), "missing",
                            diagnostics=diagnostics, availability=availability)
    return ScanPointData(data, diagnostics, availability)


def _summary_filename(root, headers):
    filename = (_last(headers, "filename", "mccode.dat")
                if _last(headers, "type").startswith("multiarray") else "mccode.dat")
    filename = filename.strip("'\"")
    path = root / filename
    return path if path.is_file() else None


def _parse_summary(path, root_headers=None):
    text = _read_text(path)
    headers = _headers("\n".join(line for line in text.splitlines()
                                  if line.lstrip().startswith("#")))
    if not text.strip():
        return None  # Older NeXus scans have an empty summary, not an empty scan.
    try:
        columns = tuple(shlex.split(_last(headers, "variables")))
    except ValueError as error:
        raise ScanFormatError("Malformed variables in {}.".format(path)) from error
    if not columns or len(set(columns)) != len(columns):
        raise ScanFormatError("Missing or duplicate variables in {}.".format(path))
    yvars = _names(_last(headers, "yvars"))
    monitor_columns = {}
    suffixes = {"I": "total_I", "ERR": "total_E", "N": "total_N"}
    for index, name in enumerate(columns):
        component, _, suffix = name.rpartition("_")
        if component and suffix in suffixes and (not yvars or name in yvars):
            monitor_columns[index] = (component, suffixes[suffix])
    coordinate_indices = [i for i in range(len(columns)) if i not in monitor_columns]
    if not coordinate_indices or not monitor_columns:
        raise ScanFormatError("Not an ASCII scan summary: {}.".format(path))
    names = tuple(columns[i] for i in coordinate_indices)
    # Native xvars/variables use set order. Root xvars and Param preserve CLI order.
    native_order = _names(_last(root_headers or {}, "xvars"))
    if not native_order and "instrument-source" in headers:
        native_order = tuple(dict.fromkeys(re.findall(
            r"(?:^|,\s*)(\w+)\s*=", _last(headers, "param"))))
    order_known = len(native_order) == len(names) and set(native_order) == set(names)
    if order_known:
        names = native_order
    raw_parameters = _last(headers, "param")
    legacy_category = bool(re.search(r"\blist\b", " ".join(headers))) and any(
        isinstance(_parse_value(value), str) for value in
        re.findall(r"(?:^|,\s*)\w+\s*=\s*([^,]*)", raw_parameters))
    categorical_names = tuple(name for name, value in re.findall(
        r"(?:^|,\s*)(\w+)\s*=\s*([^,]*)", raw_parameters)
        if isinstance(_parse_value(value), str))
    if "instrument-source" in headers and len(names) > 1 and not order_known:
        categorical_names = names
    rows, encoded_layout = [], None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=False, posix=False)
            encoded_id = None
            if len(tokens) != len(columns):
                if (legacy_category and len(tokens) == len(monitor_columns) + 1
                        and re.fullmatch(r"\d+", tokens[0])):
                    encoded_id = int(tokens[0])
                    indices = {old: new for new, old in enumerate(sorted(monitor_columns), 1)}
                else:
                    raise ValueError("{} columns; expected {}".format(len(tokens), len(columns)))
            else:
                indices = {i: i for i in monitor_columns}
            if encoded_layout is not None and encoded_layout != (encoded_id is not None):
                raise ValueError("Inconsistent coordinate column counts")
            encoded_layout = encoded_id is not None
            totals = {}
            for index, (component, quantity) in monitor_columns.items():
                totals.setdefault(component, {})[quantity] = float(tokens[indices[index]])
            coordinates = {} if encoded_id is not None else {
                name: _parse_value(tokens[i]) for name, i in zip(names, coordinate_indices)}
        except ValueError as error:
            raise ScanFormatError("Malformed summary {} line {}: {}".format(
                path, number, error)) from error
        rows.append(_SummaryRow(number, line, coordinates, totals, encoded_id,
                                categorical_names))
    return _AsciiSummary(path, headers, columns, names, rows)


def _same_value(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-5, atol=0, equal_nan=True))
    return left == right


def _apply_summary_row(point, row, path):
    for component, totals in row.totals.items():
        matches = [monitor for monitor in point.summaries if monitor.component == component]
        monitor = matches[0] if len(matches) == 1 else ScanMonitor(component)
        current = point.summaries.get(monitor)
        if len(matches) > 1:
            point.metadata.setdefault("diagnostics", []).append(
                "Ambiguous full-data identity for summary component {}; retained "
                "as a summary-only identity.".format(component))
        values = {name: getattr(current, name) if current else None for name in _TOTALS}
        provenance = dict(current.provenance) if current else {}
        for name, value in totals.items():
            if values[name] is not None and not _same_value(values[name], value):
                raise ScanFormatError("Conflicting {} for {} at point {!r} in {}.".format(
                    name, component, point.point_id, path))
            values[name] = value
            provenance[name] = str(path) + ":" + str(row.number)
        point.summaries[monitor] = ScanMonitorSummary(
            monitor, **values,
            availability={name: value is not None for name, value in values.items()},
            provenance=provenance,
            metadata={"raw_row": row.raw, "line": row.number,
                      "point_metadata": current.metadata if current else {}},
        )


def _summary_only_point(row, point_id, summary):
    encoded = {name: value for name, value in row.coordinates.items()
               if name in row.encoded_parameters and not isinstance(value, str)}
    point = ScanPoint(point_id, {name: value for name, value in row.coordinates.items()
                                if name not in encoded}, data_status="missing",
                      metadata={"summary_line": row.number,
                                "encoded_point_id": row.encoded_point_id,
                                "encoded_parameters": encoded,
                                "unavailable_parameters": list(summary.parameter_names)
                                if row.encoded_point_id is not None else list(encoded)})
    _apply_summary_row(point, row, summary.path)
    return point


def _row_matches(point, row):
    return bool(row.coordinates) and all(
        name in point.parameters and (
            isinstance(point.parameters[name], str) and not isinstance(value, str)
            or _same_value(point.parameters[name], value))
        for name, value in row.coordinates.items())


def _attach_summary(points, summary, *, directory_ids=False):
    """Associate only unique evidence; preserve all unresolved rows explicitly."""
    issues, unmatched, assigned = [], [], set()
    if summary is None:
        return issues, unmatched
    for row in summary.rows:
        if row.encoded_point_id is None and any(
                name in row.coordinates and not isinstance(row.coordinates[name], str)
                for name in row.encoded_parameters):
            issues.append("Encoded categorical coordinates at {} line {}; labels "
                          "require point metadata.".format(summary.path, row.number))
            unmatched.append(row)
            continue
        if row.encoded_point_id is not None and directory_ids:
            candidates = [p for p in points if p.point_id == row.encoded_point_id]
        else:
            candidates = [p for p in points if _row_matches(p, row)]
        if len(candidates) > 1:
            # Totals can disambiguate repeats, but successful data is not execution evidence.
            matching = []
            for point in candidates:
                overlaps = [(getattr(s, name), value)
                            for monitor, s in point.summaries.items()
                            for name, value in row.totals.get(monitor.component, {}).items()
                            if getattr(s, name) is not None]
                if overlaps and all(_same_value(left, right) for left, right in overlaps):
                    matching.append(point)
            candidates = matching
        if len(candidates) == 1 and candidates[0].point_id not in assigned:
            _apply_summary_row(candidates[0], row, summary.path)
            assigned.add(candidates[0].point_id)
        elif candidates:
            issues.append("Ambiguous summary association at {} line {}.".format(
                summary.path, row.number))
            unmatched.append(row)
        else:
            unmatched.append(row)
    return issues, unmatched


def _point_directories(root):
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdecimal()
                   and (p / "mccode.sim").is_file()), key=lambda p: int(p.name))


def _load_ascii_scan(root, summary_path=None):
    sim = root / "mccode.sim"
    headers = _headers(_read_text(sim)) if sim.is_file() else {}
    summary_path = summary_path or _summary_filename(root, headers)
    summary = _parse_summary(summary_path, headers) if summary_path else None
    points = [_point_records(path) for path in _point_directories(root)]
    issues, unmatched = _attach_summary(points, summary, directory_ids=True)
    if not points and summary is None:
        raise ScanFormatError("No ASCII scan summary or numbered metadata in {}.".format(root))
    if summary is not None:
        if not points:
            points = [_summary_only_point(row, row.encoded_point_id
                      if row.encoded_point_id is not None else i, summary)
                      for i, row in enumerate(summary.rows)]
            unmatched = []
        elif unmatched:
            unresolved = []
            for row in unmatched:
                if ((row.encoded_point_id is not None and not any(
                        p.point_id == row.encoded_point_id for p in points))
                        or (row.coordinates and not any(_row_matches(p, row) for p in points))):
                    point_id = (row.encoded_point_id if row.encoded_point_id is not None
                                else "summary:{}".format(row.number))
                    points.append(_summary_only_point(row, point_id, summary))
                else:
                    unresolved.append(row)
            unmatched = unresolved
            issues.append("Rows without point metadata became summary-only records; "
                          "ambiguous rows remain in metadata['unassociated_summary_rows'].")
    declared = _last(summary.headers if summary else headers, "numpoints", None)
    if declared and summary and len(summary.rows) != int(declared):
        issues.append("Declared Numpoints differs from actual rows; the native count may "
                      "be per-axis or output may be incomplete. Missing point IDs are unknown.")
    id_basis = "numbered directories" if _point_directories(root) else "summary row order"
    if id_basis == "summary row order" and summary and any(
            row.encoded_point_id is not None for row in summary.rows):
        id_basis = "encoded indices"
    metadata = {"format": "ascii", "source": str(root),
                "point_id_basis": id_basis, "headers": summary.headers if summary else headers,
                "root_headers": headers, "summary": str(summary_path) if summary_path else None,
                "columns": list(summary.columns) if summary else [], "diagnostics": issues,
                "unassociated_summary_rows": [vars(row) for row in unmatched],
                "declared_numpoints": declared}
    return ScanResult(points, metadata=metadata)


def is_ascii_scan_directory(path):
    """Detect explicit scan-root headers, without probing detector arrays."""
    path = Path(path)
    for filename in ("mccode.sim", "mccode.dat"):
        candidate = path / filename
        if candidate.is_file():
            with candidate.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if re.match(r"\s*#?\s*type:\s*multiarray_1d", line):
                        return True
                    if filename.endswith(".dat") and not line.lstrip().startswith("#"):
                        break
    return False


def load_scan(path):
    """Load a native scan directory, ASCII summary, or NeXus file, without execution."""
    path = Path(path).absolute()
    if not path.exists():
        raise FileNotFoundError("Scan path does not exist: {}".format(path))
    from .scan_nexus import load_nexus_scan, _nexus_files
    if path.is_file():
        if path.suffix.lower() in (".h5", ".hdf5"):
            return load_nexus_scan(path)
        return _load_ascii_scan(path.parent, summary_path=path)
    if _nexus_files(path):
        return load_nexus_scan(path)
    return _load_ascii_scan(path)
