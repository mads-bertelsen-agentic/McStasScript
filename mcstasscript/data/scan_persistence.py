"""Manifest persistence and plan-aware native scan loading.

The native readers deliberately know nothing about a requested scan plan.  This
module is the boundary between those readers and an execution manifest: native
observations are matched to plan points using coordinates, native identity, and
the verified point-offset seed policy, while all unobserved plan points remain
in the returned result.
"""

from copy import deepcopy
import json
import numbers
import os
from pathlib import Path
import tempfile

import numpy as np

from .scan import ScanDataError, ScanPoint, ScanResult
from .scan_manifest import MANIFEST_FILENAME, ScanManifest


def _absolute_path(value, name):
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise TypeError("{} must be path-like.".format(name)) from error
    if "\x00" in os.fsdecode(raw):
        raise ValueError("{} cannot contain NUL characters.".format(name))
    return Path(os.path.abspath(raw))


def _root(value):
    path = _absolute_path(value, "root")
    if path.name == MANIFEST_FILENAME and path.is_file():
        return path.parent
    return path


def write_manifest(manifest, root):
    """Atomically write ``manifest`` as ``root/scan_manifest.json``.

    The temporary file is created beside the destination, flushed to disk, and
    replaced only after serialization succeeds.  Cleanup also covers failures
    from ``fsync`` and ``os.replace``.
    """
    if not isinstance(manifest, ScanManifest):
        raise TypeError("manifest must be a ScanManifest.")
    root = _root(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / MANIFEST_FILENAME
    document = json.dumps(manifest.to_dict(), indent=2, sort_keys=True,
                          allow_nan=False) + "\n"
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + MANIFEST_FILENAME + ".", suffix=".tmp", dir=str(root))
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        try:
            descriptor = os.open(str(root), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            # The file replacement is already complete. Directory fsync is a
            # durability improvement, not a reason to report a partial write.
            pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def read_manifest(root_or_manifestpath):
    """Read and strictly validate a manifest from a root or manifest path."""
    value = _absolute_path(root_or_manifestpath, "manifest path")
    path = value if value.is_file() or value.name == MANIFEST_FILENAME else value / MANIFEST_FILENAME
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return ScanManifest.from_dict(document)


def _same_value(left, right):
    if isinstance(left, numbers.Real) and not isinstance(left, bool) and \
            isinstance(right, numbers.Real) and not isinstance(right, bool):
        return bool(np.isclose(left, right, rtol=1e-5, atol=0, equal_nan=True))
    return left == right


def _parameters_match(expected, observed, names):
    return all(name in observed.parameters and
               _same_value(expected[name], observed.parameters[name])
               for name in names)


def _summary_parameters_match(expected, observed, scanned_names, fixed_parameters):
    if not _parameters_match(expected, observed, scanned_names):
        return False
    return all(name not in observed.parameters or
               _same_value(expected[name], observed.parameters[name])
               for name in fixed_parameters)


def _seed_value(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError:
            return None
        return result if result > 0 else None
    return None


def _seed_policy(manifest):
    tool = manifest.tool
    capabilities = tool.get("capabilities", {})
    policy = capabilities.get("seed_policy") if isinstance(capabilities, dict) else None
    if policy is None:
        policy = tool.get("seed_policy")
    if policy is None:
        policy = manifest.effective_settings.get("seed_policy")
    return policy


def _base_seed(manifest):
    return _seed_value(manifest.effective_settings.get("seed"))


def _plan_id(plan, native, plan_point_id):
    result = plan_point_id(plan, native)
    if result < 0 or result >= plan.point_count:
        raise ValueError("Native point index maps outside the scan plan.")
    return result


def _native_index(plan, point_id, native_index):
    result = native_index(plan, point_id)
    if result < 0:
        raise ValueError("Plan point maps to a negative native index.")
    return result


def _numbered_native_index(point, native_root):
    source = point.source
    if source is None or source.format != "ascii":
        return None
    if source.path.parent == native_root and source.path.name.isdecimal():
        return int(source.path.name)
    return None


def _current_source(point, native_root):
    if point.source is None:
        return None
    source = point.source
    try:
        source.path.relative_to(native_root)
    except ValueError:
        return None
    return source


def _native_observations(manifest, native_root, diagnostics):
    """Load detached native summaries, retaining expected native failures."""
    from .scan_ascii import (_load_ascii_scan, _point_records, _point_directories,
                             is_ascii_scan_directory, load_scan as read_scan)
    from .scan_nexus import _nexus_files

    if not native_root.exists():
        diagnostics.append("Native output is absent: {}".format(native_root))
        return [], None, "missing"
    if not any(native_root.iterdir()):
        diagnostics.append("Native output is absent: {}".format(native_root))
        return [], None, "missing"

    def ascii_fallback(error):
        label = ("Combined NeXus output invalid" if _nexus_files(native_root)
                 else "Native output invalid")
        diagnostics.append("{}: {}: {}".format(label,
            type(error).__name__, error))
        try:
            result = _load_ascii_scan(native_root)
        except (FileNotFoundError, OSError, ValueError) as fallback_error:
            diagnostics.append("ASCII summary fallback unavailable: {}: {}".format(
                type(fallback_error).__name__, fallback_error))
            return [], None, "unreadable"
        metadata = deepcopy(result.metadata)
        metadata["fallback_from"] = "nexus"
        return list(result.points), metadata, "unreadable"

    if (manifest.plan.point_count == 1 and (native_root / "mccode.sim").is_file()
            and not _point_directories(native_root)
            and not _nexus_files(native_root)
            and not is_ascii_scan_directory(native_root)):
        try:
            return ([_point_records(native_root)],
                    {"format": "ascii", "source": str(native_root)}, "available")
        except (FileNotFoundError, OSError, ValueError) as error:
            return ascii_fallback(error)

    try:
        result = read_scan(native_root)
    except (FileNotFoundError, OSError, ValueError) as error:
        return ascii_fallback(error)
    return list(result.points), result.metadata, "available"


def _record_source(manifest, point_id, root):
    if point_id not in manifest.records or manifest.records[point_id]["source"] is None:
        return None
    return manifest.resolve_source(point_id, root)


def _native_seed(observed):
    value = observed.metadata.get("seed")
    if isinstance(value, dict):
        value = value.get("actual")
    return _seed_value(value)


def _source_locator(source):
    if source is None:
        return None
    return {"path": str(source.path), "format": source.format,
            "entry": source.entry}


def _summary_record(summary):
    monitor = summary.monitor
    return {
        "monitor": {"component": monitor.component,
                    "filename": monitor.filename,
                    "discriminator": monitor.discriminator},
        "total_I": summary.total_I,
        "total_E": summary.total_E,
        "total_N": summary.total_N,
        "availability": dict(summary.availability),
        "provenance": deepcopy(summary.provenance),
        "metadata": deepcopy(summary.metadata),
    }


def _unassociated_record(observation):
    """Detach native evidence without accessing a point's lazy data."""
    return {
        "native_point_id": observation.point_id,
        "source": _source_locator(observation.source),
        "parameters": deepcopy(observation.parameters),
        "summaries": [_summary_record(summary)
                      for summary in observation.summaries.values()],
        "metadata": deepcopy(observation.metadata),
        "data_status": observation.data_status,
    }


def _parameter_key(parameters, names):
    """Build a lossless lookup key for the no-identity fallback."""
    values = []
    for name in names:
        if name not in parameters:
            return None
        value = parameters[name]
        if isinstance(value, str):
            values.append(("string", value))
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            values.append(("number", value))
        else:
            values.append((type(value).__name__, repr(value)))
    return tuple(values)


def _record_matches_observation(manifest, root, point_id, observation,
                                actual_seed, intended_seed):
    record = manifest.records.get(point_id)
    if record is None:
        return True, None
    if record["source"] is not None:
        expected_source = manifest.resolve_source(point_id, root)
        if observation.source is not None and observation.source != expected_source:
            return False, "Manifest source conflicts with current native source for point {}.".format(
                point_id)
    recorded_seed = record["seed"]
    if recorded_seed is not None:
        if (recorded_seed["actual"] is not None and actual_seed is not None and
                recorded_seed["actual"] != actual_seed):
            return False, "Manifest actual seed conflicts for point {}.".format(point_id)
        if (recorded_seed["intended"] is not None and
                recorded_seed["intended"] != intended_seed):
            return False, "Manifest intended seed conflicts for point {}.".format(point_id)
    return True, None


def _associate(manifest, observations, native_root, diagnostics):
    plan = manifest.plan
    from mcstasscript.helper.scan_command import native_index, plan_point_id

    names = tuple(manifest.fixed_parameters) + tuple(plan.parameter_names)
    scanned_names = tuple(plan.parameter_names)
    parameter_lookup = None
    association_issues = []

    def issue(message):
        diagnostics.append(message)
        association_issues.append(message)

    def coordinate_candidates(observed):
        nonlocal parameter_lookup
        summary_only = observed.source is None
        if summary_only:
            return [point.point_id for point in plan
                    if _summary_parameters_match(
                        {**manifest.fixed_parameters, **plan[point.point_id].parameters},
                        observed, scanned_names, manifest.fixed_parameters)]
        if parameter_lookup is None:
            parameter_lookup = {}
            for point in plan:
                expected = {**manifest.fixed_parameters, **point.parameters}
                key = _parameter_key(expected, names)
                parameter_lookup.setdefault(key, []).append(point.point_id)
        candidate_ids = parameter_lookup.get(
            _parameter_key(observed.parameters, names), [])
        matched = [point_id for point_id in candidate_ids
                   if _parameters_match(
                       {**manifest.fixed_parameters, **plan[point_id].parameters},
                       observed, names)]
        if matched:
            return matched
        # Native metadata can be rounded across a lookup-key boundary. The
        # tolerance-aware scan is deliberately only the no-identity fallback.
        return [point.point_id for point in plan
                if _parameters_match(
                    {**manifest.fixed_parameters, **point.parameters},
                    observed, names)]

    policy = _seed_policy(manifest)
    supported_seeds = policy == "point_offset_1024"
    base_seed = _base_seed(manifest) if supported_seeds else None
    if policy not in (None, "point_offset_1024"):
        issue("Unsupported native seed policy {!r}; seed matching disabled.".format(policy))

    associated = {}
    for observed in observations:
        known_native = _numbered_native_index(observed, native_root)
        if known_native is None and isinstance(observed.point_id, numbers.Integral):
            if (observed.source is not None and observed.source.format == "nexus"
                    and observed.source.path.parent.name.isdecimal()):
                known_native = int(observed.point_id)
        actual_seed = _native_seed(observed)
        candidate_ids = []
        intended = None
        seed_point_id = None
        seed_native = None
        seed_conflict = False
        if supported_seeds and base_seed is not None and actual_seed is not None:
            delta = actual_seed - base_seed
            if delta < 0 or delta % 1024:
                issue("Native observation {!r} has an invalid seed offset."
                      .format(observed.point_id))
                seed_conflict = True
            else:
                seed_native = delta // 1024
                if seed_native >= plan.point_count:
                    issue("Native observation {!r} has a seed outside the scan plan."
                          .format(observed.point_id))
                    seed_conflict = True
                else:
                    seed_point_id = _plan_id(plan, seed_native, plan_point_id)

        native_point_id = None
        native_plan_point_id = None
        if known_native is not None:
            if known_native < 0 or known_native >= plan.point_count:
                issue("Native point identity {!r} is outside the scan plan."
                      .format(known_native))
                seed_conflict = True
            else:
                native_plan_point_id = _plan_id(plan, known_native, plan_point_id)
                native_point_id = known_native

        identity = (seed_point_id if seed_point_id is not None
                    else native_plan_point_id)
        identity_checked = identity is not None
        summary_only = observed.source is None
        if (seed_native is not None and known_native is not None and
                seed_native != known_native):
            issue("Seed conflict for native point {!r}.".format(observed.point_id))
            seed_conflict = True

        if identity is not None and not seed_conflict:
            expected = {**manifest.fixed_parameters, **plan[identity].parameters}
            if (_summary_parameters_match(expected, observed, scanned_names,
                                          manifest.fixed_parameters)
                    if summary_only else _parameters_match(expected, observed, names)):
                candidate_ids = [identity]
            else:
                issue("Native observation {!r} has no matching plan parameters."
                      .format(observed.point_id))
        elif actual_seed is None and native_point_id is None and not seed_conflict:
            candidate_ids = coordinate_candidates(observed)
        elif actual_seed is not None and seed_point_id is None and native_point_id is None:
            issue("Native observation {!r} has seed evidence but no supported identity."
                  .format(observed.point_id))

        if len(candidate_ids) != 1:
            if len(candidate_ids) > 1:
                issue("Ambiguous native association for observation {!r}; "
                      "coordinates were not guessed.".format(observed.point_id))
            elif (not identity_checked and not seed_conflict and
                  not (actual_seed is not None and native_point_id is None)):
                issue("Native observation {!r} has no matching plan parameters."
                      .format(observed.point_id))
            continue
        point_id = candidate_ids[0]
        if point_id in associated:
            issue("Duplicate native observations for plan point {}.".format(point_id))
            continue
        if supported_seeds and base_seed is not None:
            intended = base_seed + 1024 * _native_index(plan, point_id, native_index)
        consistent, conflict = _record_matches_observation(
            manifest, native_root.parent, point_id, observed, actual_seed, intended)
        if not consistent:
            issue(conflict)
            continue
        if native_point_id is None:
            native_point_id = seed_native
        associated[point_id] = (observed, actual_seed, intended, native_point_id,
                                 seed_native)
    return associated, association_issues


def _point_from_observation(manifest, root, native_root, point_id, observation,
                            actual_seed, intended_seed, native_point_id,
                            seed_native, native_status):
    plan_point = manifest.plan[point_id]
    record = manifest.records.get(point_id)
    status = record["status"] if record is not None else "unknown"
    if observation.source is None and observation.data_status == "missing":
        data_status = "missing"
    else:
        data_status = (record["data_status"] if record is not None else
                       observation.data_status)
    source = _current_source(observation, native_root) or observation.source
    metadata = deepcopy(observation.metadata)
    metadata["native_point_id"] = native_point_id
    metadata["seed_native"] = seed_native
    metadata["storage_point_id"] = observation.point_id
    metadata["storage_entry"] = (observation.source.entry
                                  if observation.source is not None and
                                  observation.source.format == "nexus" else None)
    metadata["native_seed"] = actual_seed
    metadata["intended_seed"] = intended_seed
    metadata["seed"] = {"actual": actual_seed, "intended": intended_seed}
    metadata["native_data_status"] = native_status
    return ScanPoint(point_id, dict(plan_point.parameters), source,
                     summaries=tuple(observation.summaries.values()), status=status,
                     data_status=data_status, metadata=metadata,
                     loader=observation._loader)


def _missing_point(manifest, root, point_id, data_status="missing"):
    plan_point = manifest.plan[point_id]
    record = manifest.records.get(point_id)
    source = _record_source(manifest, point_id, root)
    recorded_seed = deepcopy(record["seed"]) if record is not None else None
    intended = (recorded_seed["intended"] if recorded_seed is not None and
                recorded_seed["intended"] is not None else None)
    if (_seed_policy(manifest) == "point_offset_1024" and
            _base_seed(manifest) is not None):
        from mcstasscript.helper.scan_command import native_index
        if intended is None:
            intended = _base_seed(manifest) + 1024 * native_index(manifest.plan, point_id)
    actual = recorded_seed["actual"] if recorded_seed is not None else None
    seed = recorded_seed if recorded_seed is not None else {
        "actual": None, "intended": intended}
    if seed["intended"] is None:
        seed["intended"] = intended
    metadata = {"native_point_id": None, "seed_native": None,
                "storage_point_id": None, "storage_entry": None,
                "native_seed": actual,
                "intended_seed": intended,
                "seed": seed}
    if record is not None:
        metadata["manifest_record"] = deepcopy(dict(record))
    return ScanPoint(point_id, dict(plan_point.parameters), source,
                     status=record["status"] if record is not None else "unknown",
                     data_status=data_status, metadata=metadata)


def _inspect(points, diagnostics):
    for point in points:
        if point.source is None or point._loader is None:
            continue
        try:
            point.data
        except ScanDataError as error:
            diagnostics.append("Point {} data inspection: {}".format(point.point_id, error))
        if point.diagnostics:
            diagnostics.extend("Point {} monitor {}: {}".format(
                point.point_id, monitor, detail)
                for monitor, detail in point.diagnostics.items())


def reconcile_scan(manifest, root, *, inspect_data=False):
    """Reconcile current ``root/native`` output with a strict scan manifest."""
    if not isinstance(manifest, ScanManifest):
        raise TypeError("manifest must be a ScanManifest.")
    root = _root(root)
    native_root = root / "native"
    diagnostics = ["manifest: {}".format(root / MANIFEST_FILENAME)]
    observations, native_metadata, native_status = _native_observations(
        manifest, native_root, diagnostics)
    if native_metadata is not None:
        diagnostics.extend(message for message in native_metadata.get("diagnostics", [])
                           if message not in diagnostics)
    associated, association_issues = _associate(
        manifest, observations, native_root, diagnostics)
    if native_metadata is not None:
        associated_observations = {id(value[0]) for value in associated.values()}
        unassociated = [observation for observation in observations
                        if id(observation) not in associated_observations
                        ]
        if unassociated:
            native_metadata = deepcopy(native_metadata)
            native_metadata["unassociated_observations"] = [
                _unassociated_record(observation) for observation in unassociated]
            summary_only = [observation for observation in unassociated
                            if observation.source is None]
            native_metadata.setdefault("unassociated_summary_rows", [])
            native_metadata["unassociated_summary_rows"].extend(
                {"point_id": observation.point_id,
                 "parameters": deepcopy(observation.parameters),
                 "metadata": deepcopy(observation.metadata)}
                for observation in summary_only)
    missing_data_status = "unreadable" if native_status == "unreadable" else "missing"
    points = []
    for point_id in range(manifest.plan.point_count):
        if point_id in associated:
            points.append(_point_from_observation(manifest, root, native_root, point_id,
                                                 *associated[point_id], native_status))
        else:
            points.append(_missing_point(manifest, root, point_id, missing_data_status))
    if inspect_data:
        _inspect(points, diagnostics)
    metadata = {
        "format": "manifest",
        "source": str(root),
        "sources": {"manifest": str(root / MANIFEST_FILENAME),
                    "native": str(native_root)},
        "native": deepcopy(native_metadata) if native_metadata is not None else {},
        "diagnostics": diagnostics,
        "association_issues": association_issues,
        "native_data_status": native_status,
    }
    return ScanResult(points, parameter_names=manifest.plan.parameter_names,
                      plan=manifest.plan, metadata=metadata,
                      execution=manifest.execution)


def load_manifest_scan(path):
    """Load a manifest and reconcile it against its containing root."""
    value = _absolute_path(path, "manifest path")
    manifest_path = (value if value.is_file() or value.name == MANIFEST_FILENAME
                     else value / MANIFEST_FILENAME)
    manifest = read_manifest(manifest_path)
    return reconcile_scan(manifest, manifest_path.parent)


def finalize_manifest(manifest, root):
    """Inspect output once and return a fresh persisted manifest plus its scan.

    Execution evidence is copied unchanged.  Only observed points and existing
    records are persisted; missing plan points remain represented by the scan.
    Readability changes only each point's data status and diagnostics; it never
    turns readable data into execution success.
    """
    if not isinstance(manifest, ScanManifest):
        raise TypeError("manifest must be a ScanManifest.")
    root = _root(root)
    scan = reconcile_scan(manifest, root, inspect_data=True)
    updated = ScanManifest(
        plan=manifest.plan,
        fixed_parameters=manifest.fixed_parameters,
        parameter_snapshot=manifest.parameter_snapshot,
        effective_settings=manifest.effective_settings,
        tool=manifest.tool,
        output_root=root,
        errors=manifest.errors,
        warnings=manifest.warnings,
        execution=manifest.execution["process"],
    )
    for point in scan.points:
        record = manifest.records.get(point.point_id)
        if (record is None and point.source is None and not point.summaries):
            continue
        source = point.source
        native_observed = point._loader is not None or bool(point.summaries)
        if record is not None and not native_observed:
            updated.record_point(
                point.point_id,
                source=source,
                status=record["status"],
                data_status=record["data_status"],
                seed=record["seed"],
                diagnostics=record["diagnostics"],
            )
            continue
        seed = point.metadata.get("seed")
        point_diagnostics = [str(value) for value in point.diagnostics.values()]
        if record is not None:
            point_diagnostics = list(record["diagnostics"]) + point_diagnostics
        updated.record_point(
            point.point_id,
            source=source,
            status=point.status,
            data_status=point.data_status,
            seed=seed,
            diagnostics=point_diagnostics,
        )
    return updated, scan


__all__ = ["MANIFEST_FILENAME", "write_manifest", "read_manifest",
           "load_manifest_scan", "reconcile_scan", "finalize_manifest"]
