"""Detached metadata and entry-qualified lazy loading for native NeXus scans."""

from functools import partial
from pathlib import Path
import re

import h5py
import numpy as np

from mcstasscript.helper import managed_mcrun

from .scan import (ScanDataError, ScanMonitor, ScanPoint, ScanPointData, ScanResult,
                   ScanSource)
from .scan_ascii import (ScanFormatError, _attach_summary, _headers, _last,
                         _metadata_record, _parameter_types, _parse_summary,
                         _parse_value, _point_directories, _point_records,
                         _read_text, _same_value, _summary_filename,
                         _summary_from_metadata, _validate_data)


def _scalar(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S" and value.dtype.itemsize == 1:
            value = b"".join(value.reshape(-1))
        elif value.size == 1:
            value = value.reshape(-1)[0]
        else:
            raise ScanFormatError("Expected a scalar native metadata value.")
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    return value.item() if isinstance(value, np.generic) else value


def _entry_names(handle):
    entries = []
    for name, group in handle.items():
        if not isinstance(group, h5py.Group):
            continue
        nx_class = _scalar(group.attrs.get("NX_class", ""))
        if nx_class != "NXentry" and "simulation/Param" not in group:
            continue
        match = re.fullmatch(r"entry(\d+)", name)
        if not match:
            raise ScanFormatError("Cannot establish native entry order for {}.".format(name))
        entries.append((int(match[1]), name))
    if not entries:
        raise ScanFormatError("No native McCode entries in {}.".format(handle.filename))
    return [name for _, name in sorted(entries)]


def _nexus_files(root):
    files = sorted(p for p in root.iterdir() if p.is_file()
                   and p.suffix.lower() in (".h5", ".hdf5"))
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.isdecimal():
            files.extend(sorted(p for p in path.iterdir() if p.is_file()
                                and p.suffix.lower() in (".h5", ".hdf5")))
    return files


def _load_location(path, entry, point_id):
    with h5py.File(path, "r") as handle:
        group = handle[entry]
        try:
            parameter_group = group["simulation/Param"]
            declaration = _scalar(group["instrument"].attrs.get("Parameters", "")) \
                if "instrument" in group else ""
            types = _parameter_types(declaration)
            raw = {key: _scalar(value) for key, value in parameter_group.attrs.items()
                   if key != "NX_class"}
            for key, value in parameter_group.items():
                observed = _scalar(value[()])
                if key in raw and str(raw[key]) != str(observed):
                    raise ScanFormatError("Conflicting parameter {} in {}::{}.".format(
                        key, path, entry))
                raw[key] = observed
            parameters = {name: _parse_value(value, types.get(name))
                          for name, value in raw.items()}
        except KeyError as error:
            raise ScanFormatError("Missing simulation/Param in {}::{}.".format(path, entry)) from error
        inventory, summaries = {}, []
        for field, detector in group.get("data", {}).items():
            if not isinstance(detector, h5py.Group):
                continue
            info = {key: _scalar(value) for key, value in detector.attrs.items()}
            metadata = _metadata_record(info, parameters)
            metadata.info["NeXus_field"] = field
            monitor = ScanMonitor(metadata.component_name, metadata.filename or None)
            if monitor in inventory:
                raise ScanFormatError("Duplicate monitor identity {} in {}::{}.".format(
                    monitor, path, entry))
            inventory[monitor] = metadata
            summaries.append(_summary_from_metadata(metadata, str(path) + "::" + entry, monitor))
        simulation = {key: _scalar(value) for key, value in group["simulation"].attrs.items()}
    return ScanPoint(
        point_id, parameters, ScanSource(path, "nexus", entry), summaries=summaries,
        metadata={"native": simulation, "parameter_types": types,
                  "seed": _parse_value(simulation["Seed"], "int") if "Seed" in simulation else None},
        loader=partial(_load_point_data, inventory=inventory),
    )


def _load_point_data(source, inventory):
    data, diagnostics, availability = [], {}, {}
    with h5py.File(source.path, "r") as handle:
        if source.entry not in handle:
            raise ScanDataError("Missing NeXus entry {}.".format(source), "missing")
        for monitor, metadata in inventory.items():
            field = source.entry + "/data/" + metadata.info["NeXus_field"]
            detector = handle.get(field)
            if detector is not None and "events" in detector:
                available = {"Events": True}
            elif metadata.dimension == 0:
                available = {"Intensity": metadata.total_I is not None,
                             "Error": metadata.total_E is not None,
                             "Ncount": metadata.total_N is not None}
            else:
                available = {name: detector is not None and key in detector for name, key in
                             (("Intensity", "data"), ("Error", "errors"), ("Ncount", "ncount"))}
            availability[monitor] = available
            try:
                dataset = managed_mcrun.load_monitor_nexus(metadata, handle, source.entry)
                _validate_data(dataset, available)
                dataset.set_data_location(str(source))
                data.append(dataset)
            except (OSError, ValueError, IndexError, KeyError, NameError) as error:
                diagnostics[monitor] = field + ": " + str(error)
    if not inventory:
        raise ScanDataError("No monitor output in {}.".format(source), "missing")
    return ScanPointData(data, diagnostics, availability)


def _merge_ascii_points(points, root):
    ascii_points = [_point_records(path) for path in _point_directories(root)]
    if not ascii_points:
        return points
    remaining = list(points)
    merged = []
    for point in ascii_points:
        matches = [p for p in remaining if p.parameters.keys() == point.parameters.keys()
                   and all(_same_value(p.parameters[k], v) for k, v in point.parameters.items())]
        duplicates = [p for p in ascii_points if p.parameters == point.parameters]
        if len(matches) > 1 or len(duplicates) > 1:
            seed = point.metadata.get("seed")
            if seed is None or len([p for p in duplicates if p.metadata.get("seed") == seed]) != 1:
                raise ScanFormatError("Ambiguous duplicate-coordinate ASCII/NeXus association.")
            matches = [p for p in matches if p.metadata.get("seed") == seed]
        if len(matches) > 1:
            raise ScanFormatError("Ambiguous ASCII/NeXus association at point {}.".format(point.point_id))
        if matches:
            nexus = matches[0]
            if any((point.source.path / monitor.filename).is_file()
                   for monitor in point.summaries if monitor.filename):
                raise ScanFormatError("Competing full ASCII and NeXus output in {}; "
                                      "select the NeXus file explicitly.".format(root))
            for monitor, summary in point.summaries.items():
                if monitor in nexus.summaries:
                    for name in ("total_I", "total_E", "total_N"):
                        left, right = getattr(summary, name), getattr(nexus.summaries[monitor], name)
                        if left is not None and right is not None and not _same_value(left, right):
                            raise ScanFormatError("Conflicting ASCII/NeXus {} at point {}.".format(
                                name, point.point_id))
            combined = dict(point.summaries)
            for monitor, summary in nexus.summaries.items():
                old = combined.get(monitor)
                if old is not None:
                    totals = {name: getattr(summary, name) if getattr(summary, name) is not None
                              else getattr(old, name) for name in ("total_I", "total_E", "total_N")}
                    summary = type(summary)(monitor, **totals,
                        availability={name: value is not None for name, value in totals.items()},
                        provenance={**old.provenance, **summary.provenance})
                combined[monitor] = summary
            merged.append(ScanPoint(point.point_id, point.parameters, nexus.source,
                          summaries=combined.values(),
                          metadata={**point.metadata, "ascii_source": str(point.source)},
                          loader=nexus._loader))
            remaining.remove(nexus)
        else:
            # Coexisting full sources with inconsistent coordinates must not silently diverge.
            if remaining and point.summaries:
                raise ScanFormatError("Conflicting or unassociated ASCII/NeXus point {}.".format(point.point_id))
            merged.append(point)
    if remaining:
        raise ScanFormatError("Unassociated NeXus entries coexist with numbered ASCII output.")
    return merged


def load_nexus_scan(path):
    """Load combined or numbered per-point files; original plan IDs may be unknown."""
    root = path.parent if path.is_file() else path
    files = [path] if path.is_file() else _nexus_files(root)
    if not files:
        raise ScanFormatError("No native NeXus files in {}.".format(root))
    points, layouts = [], set()
    for filename in files:
        with h5py.File(filename, "r") as handle:
            entries = _entry_names(handle)
        numbered = filename.parent != root and filename.parent.name.isdecimal()
        layouts.add("per-point" if numbered else "combined")
        if numbered and len(entries) != 1:
            raise ScanFormatError("Multiple entries in per-point file {}.".format(filename))
        if not numbered and (len(files) != 1):
            raise ScanFormatError("Ambiguous combined and per-point NeXus sources in {}.".format(root))
        for entry in entries:
            point_id = int(filename.parent.name) if numbered else len(points)
            points.append(_load_location(filename, entry, point_id))
    if len({p.point_id for p in points}) != len(points):
        raise ScanFormatError("Duplicate native point directories in {}.".format(root))
    if path.is_dir():
        points = _merge_ascii_points(points, root)
    numbered_ids = (path.is_dir() and bool(_point_directories(root))) or layouts == {"per-point"}
    sim = root / "mccode.sim"
    headers = _headers(_read_text(sim)) if sim.is_file() else {}
    summary_path = _summary_filename(root, headers)
    summary = _parse_summary(summary_path, headers) if summary_path else None
    issues, unmatched = _attach_summary(points, summary, directory_ids=numbered_ids)
    if not numbered_ids:
        issues.append("Entry order identifies storage only; original point IDs and "
                      "requested-but-missing points are unknown.")
    if summary and _last(summary.headers, "numpoints") != str(len(summary.rows)):
        issues.append("Declared Numpoints differs from summary rows; no missing plan IDs inferred.")
    return ScanResult(points, metadata={
        "format": "nexus", "source": str(root),
        "point_id_basis": "numbered directories" if numbered_ids else "entry order",
        "layout": next(iter(layouts)), "diagnostics": issues,
        "summary": str(summary_path) if summary_path else None,
        "headers": summary.headers if summary else headers,
        "unassociated_summary_rows": [vars(row) for row in unmatched],
        "declared_numpoints": _last(summary.headers, "numpoints", None) if summary else None,
    })


def is_nexus_scan_directory(path):
    """Detect multiple entries or numbered point files, not an empty summary alone."""
    root = Path(path)
    for filename in _nexus_files(root):
        if filename.name != "mccode.h5":
            continue
        if filename.parent != root:
            return True
        try:
            with h5py.File(filename, "r") as handle:
                if len(_entry_names(handle)) > 1:
                    return True
        except ScanFormatError:
            return True  # Never silently load entry1 when entry discovery was ambiguous.
        except OSError:
            continue
    return False
