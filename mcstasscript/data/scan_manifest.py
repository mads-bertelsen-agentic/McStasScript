"""Versioned, execution-free manifests for parameter scans.

The manifest records the requested plan and observations about an execution. It
does not run a simulation, inspect output files, or keep file handles. Point
records are deliberately sparse: a point which has not been observed is not
materialized in memory or in the serialized document.
"""

from copy import deepcopy
from collections.abc import Mapping
import json
import math
import numbers
import os
from pathlib import Path
from types import MappingProxyType

from .scan_plan import ScanPlan
from .scan import ScanSource


SCHEMA = "mcstasscript.scan_manifest"
VERSION = 1
POINT_STATUSES = ("pending", "unknown", "running", "succeeded", "failed",
                  "interrupted", "skipped")
DATA_STATUSES = ("unknown", "available", "partial", "missing", "unreadable")
_TOP_LEVEL_KEYS = {
    "schema", "version", "instrument", "plan", "effective_settings", "tool",
    "output_root", "execution", "points", "errors", "warnings",
}
_POINT_KEYS = {"source", "status", "data_status", "seed", "diagnostics"}
_SOURCE_KEYS = {"path", "format", "entry", "external"}
_SEED_KEYS = {"actual", "intended"}
_PROCESS_KEYS = {"status", "outcome", "returncode", "log", "diagnostics"}
_PROCESS_STATUSES = ("unknown", "pending", "running", "succeeded", "failed",
                     "interrupted")


def _json_value(value, name="value"):
    """Return a detached JSON-safe value, rejecting lossy conversions."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{} cannot contain NaN or infinity.".format(name))
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, "{}[]".format(name)) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("{} mapping keys must be strings.".format(name))
            result[key] = _json_value(item, "{}[{}]".format(name, key))
        return result
    raise TypeError("{} must contain only JSON values.".format(name))


def _json_copy(value, name="value"):
    copied = _json_value(value, name)
    # This also makes the contract explicit if a future JSON-safe type is added.
    json.dumps(copied, allow_nan=False)
    return copied


def _path(value, name):
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise TypeError("{} must be path-like.".format(name)) from error
    if "\x00" in os.fsdecode(raw):
        raise ValueError("{} cannot contain NUL characters.".format(name))
    return Path(os.path.abspath(raw))


def _validate_plan(plan):
    if not isinstance(plan, ScanPlan):
        raise TypeError("plan must be a ScanPlan.")
    if isinstance(plan.point_count, bool) or not isinstance(plan.point_count, numbers.Integral):
        raise TypeError("ScanPlan.point_count must be an integer.")
    if plan.point_count < 0:
        raise ValueError("ScanPlan.point_count cannot be negative.")
    plan_dict = plan.to_dict()
    if not isinstance(plan_dict, dict):
        raise TypeError("ScanPlan.to_dict() must return a mapping.")
    _json_copy(plan_dict, "plan")
    try:
        ScanPlan.from_dict(plan_dict)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("ScanPlan.to_dict() does not describe a valid ScanPlan.") from error
    return plan_dict


def _plan_point(plan, point_id):
    """Resolve one canonical integer ID with no plan expansion."""
    if isinstance(point_id, bool) or not isinstance(point_id, numbers.Integral):
        raise TypeError("Point ID must be a nonnegative integer.")
    point_id = int(point_id)
    if point_id < 0 or point_id >= plan.point_count:
        raise ValueError("Point ID {!r} is not part of the scan plan.".format(point_id))
    point = plan[point_id]
    if point.point_id != point_id:
        raise ValueError("ScanPlan returned a noncanonical point ID.")
    return point


def _record_id(plan, key):
    """Convert one canonical JSON object key to an integer point ID."""
    if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
        raise ValueError("Point record ID {!r} is not canonical.".format(key))
    return _plan_point(plan, int(key)).point_id


def _relative_source(path, root):
    if root is None:
        return None
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    return "." if str(relative) == "." else relative.as_posix()


def _validate_relative_path(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("A relative source path must be a nonempty string.")
    if value == ".":
        return value
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.replace("\\", "/").split("/")):
        raise ValueError("Relative source paths cannot contain traversal components.")
    return value


def _source_dict(source, root):
    if source is None:
        return None
    if not isinstance(source, ScanSource):
        raise TypeError("source must be a ScanSource or None.")
    relative = _relative_source(source.path, root)
    external = relative is None
    result = {
        "path": relative if not external else str(source.path),
        "format": source.format,
        "entry": source.entry,
        "external": external,
    }
    return result


def _source_from_dict(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _SOURCE_KEYS:
        raise ValueError("A source must contain path, format, entry, and external.")
    path = value["path"]
    external = value["external"]
    if not isinstance(external, bool):
        raise TypeError("Source external must be a boolean.")
    if external:
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("External source paths must be absolute.")
    else:
        _validate_relative_path(path)
    if value["format"] not in ("ascii", "nexus"):
        raise ValueError("Invalid source format.")
    if value["format"] == "ascii" and value["entry"] is not None:
        raise ValueError("An ASCII source cannot have a NeXus entry.")
    if value["format"] == "nexus" and not isinstance(value["entry"], str):
        raise ValueError("A NeXus source needs an entry.")
    return value


def _seed(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if set(value) - _SEED_KEYS:
            raise ValueError("Seed records may contain only actual and intended values.")
        result = {"actual": None, "intended": None}
        for key, item in value.items():
            result[key] = _seed_value(item, "seed.{}".format(key))
        return result
    return {"actual": _seed_value(value, "seed"), "intended": None}


def _seed_value(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError("{} must be a positive integer or None.".format(name))
    return int(value)


def _mapping(value, name, default=None):
    if value is None:
        value = {} if default is None else default
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping.".format(name))
    return _json_copy(dict(value), name)


def _process(value):
    defaults = {"status": "unknown", "outcome": None, "returncode": None,
                "log": None, "diagnostics": []}
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        raise TypeError("execution process evidence must be a mapping.")
    if set(value) - _PROCESS_KEYS:
        raise ValueError("Unknown execution process evidence field.")
    result = dict(defaults)
    result.update(value)
    if result["status"] not in _PROCESS_STATUSES:
        raise ValueError("Invalid execution process status.")
    if result["outcome"] is not None and not isinstance(result["outcome"], str):
        raise TypeError("Execution outcome must be a string or None.")
    if (result["returncode"] is not None and
            (isinstance(result["returncode"], bool) or
             not isinstance(result["returncode"], numbers.Integral))):
        raise TypeError("Execution returncode must be an integer or None.")
    if not isinstance(result["log"], (str, type(None))):
        raise TypeError("Execution log must be a string or None.")
    if not isinstance(result["diagnostics"], (list, tuple)):
        raise TypeError("Execution diagnostics must be a sequence.")
    result["returncode"] = (None if result["returncode"] is None
                              else int(result["returncode"]))
    result["diagnostics"] = _json_copy(list(result["diagnostics"]),
                                        "execution.diagnostics")
    return result


def _validate_execution_aggregate(value):
    if not isinstance(value, dict) or set(value) != {
            "process", "point_count", "recorded", "counts"}:
        raise ValueError("Invalid execution aggregate.")
    for name in ("point_count", "recorded"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError("Execution {} must be a nonnegative integer."
                             .format(name))
    if value["recorded"] > value["point_count"]:
        raise ValueError("Execution recorded count cannot exceed point_count.")
    counts = value["counts"]
    if not isinstance(counts, dict) or set(counts) != set(POINT_STATUSES):
        raise ValueError("Execution counts must contain exactly point statuses.")
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("Execution status counts must be nonnegative integers.")
    if not isinstance(value["process"], dict) or set(value["process"]) != _PROCESS_KEYS:
        raise ValueError("Invalid execution process evidence.")


def _messages(value, name):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise TypeError("{} must be a sequence of strings.".format(name))
    return list(value)


class ScanManifest:
    """Strict versioned metadata for one requested scan.

    ``records`` contains only explicitly observed points and never contains
    loaded data. Sources are passive locators. The plan remains the source of
    truth for point parameters, so point records do not duplicate parameters.
    """

    schema = SCHEMA
    version = VERSION

    def __init__(self, *, plan, fixed_parameters=None, parameter_snapshot=None,
                 effective_settings=None, tool=None, output_root=None, errors=(),
                 warnings=(), execution=None):
        _validate_plan(plan)
        self._plan = plan
        self._fixed_parameters = _mapping(fixed_parameters, "fixed_parameters")
        self._parameter_snapshot = _mapping(parameter_snapshot, "parameter_snapshot")
        overlap = set(self._fixed_parameters).intersection(plan.parameter_names)
        if overlap:
            raise ValueError("Fixed parameters cannot contain scanned parameters: {}."
                             .format(sorted(overlap)))
        self._effective_settings = _mapping(effective_settings, "effective_settings")
        self._tool = _mapping(tool, "tool")
        self._output_root = None if output_root is None else _path(output_root, "output_root")
        self._errors = _messages(errors, "errors")
        self._warnings = _messages(warnings, "warnings")
        self._records = {}
        self._process = _process(execution)

    @property
    def plan(self):
        """The original :class:`ScanPlan`, not an expanded list of points."""
        return self._plan

    @property
    def fixed_parameters(self):
        return deepcopy(self._fixed_parameters)

    @property
    def parameter_snapshot(self):
        return deepcopy(self._parameter_snapshot)

    @property
    def effective_settings(self):
        return deepcopy(self._effective_settings)

    @property
    def tool(self):
        return deepcopy(self._tool)

    @property
    def output_root(self):
        return self._output_root

    @property
    def errors(self):
        return list(self._errors)

    @property
    def warnings(self):
        return list(self._warnings)

    @property
    def points(self):
        """Sparse point records keyed by native plan point IDs."""
        return self.records

    @property
    def records(self):
        """Detached read-only point records."""
        return MappingProxyType({point_id: MappingProxyType(deepcopy(record))
                                 for point_id, record in self._records.items()})

    @property
    def execution(self):
        counts = {status: 0 for status in POINT_STATUSES}
        counts["unknown"] = self.plan.point_count
        for record in self._records.values():
            counts["unknown"] -= 1
            counts[record["status"]] += 1
        return {"process": deepcopy(self._process), "point_count": self.plan.point_count,
                "recorded": len(self._records), "counts": counts}

    @classmethod
    def from_preflight(cls, report):
        """Create a manifest from a valid preflight report.

        Attribute access keeps this constructor independent of a report's
        concrete implementation.
        """
        if not bool(getattr(report, "valid", False)):
            raise ValueError("Cannot create a manifest from an invalid preflight report.")
        required = ("plan", "fixed_parameters", "parameter_snapshot",
                    "effective_settings", "tool", "output_path", "errors", "warnings")
        missing = [name for name in required if not hasattr(report, name)]
        if missing:
            raise TypeError("Preflight report is missing: {}.".format(", ".join(missing)))
        return cls(plan=report.plan, fixed_parameters=report.fixed_parameters,
                   parameter_snapshot=report.parameter_snapshot,
                   effective_settings=report.effective_settings, tool=report.tool,
                   output_root=report.output_path, errors=report.errors,
                   warnings=report.warnings)

    def record_point(self, point_id, *, source=None, status="unknown",
                     data_status="unknown", seed=None, diagnostics=()):
        """Add one validated sparse observation and return its detached record.

        Existing point IDs are rejected rather than silently replaced.
        """
        point = _plan_point(self.plan, point_id)
        point_id = point.point_id
        if point_id in self._records:
            raise ValueError("Duplicate point record for {!r}; records are immutable."
                             .format(point_id))
        if status not in POINT_STATUSES:
            raise ValueError("Invalid point execution status.")
        if data_status not in DATA_STATUSES:
            raise ValueError("Invalid point data status.")
        if not isinstance(diagnostics, (list, tuple)):
            raise TypeError("diagnostics must be a sequence.")
        record = {
            "source": _source_dict(source, self.output_root),
            "status": status,
            "data_status": data_status,
            "seed": _seed(seed),
            "diagnostics": _json_copy(list(diagnostics), "diagnostics"),
        }
        self._records[point_id] = record
        return deepcopy(record)

    def record_execution(self, *, status="unknown", outcome=None, returncode=None,
                         log=None, diagnostics=()):
        """Record process evidence independently from per-point availability."""
        self._process = _process({"status": status, "outcome": outcome,
                                  "returncode": returncode, "log": log,
                                  "diagnostics": diagnostics})
        return deepcopy(self._process)

    def resolve_source(self, point_id, root=None):
        """Resolve a stored source, optionally relocating the output directory."""
        point = _plan_point(self.plan, point_id)
        if point.point_id not in self._records:
            raise KeyError("No record for point {!r}.".format(point_id))
        value = self._records[point.point_id]["source"]
        if value is None:
            return None
        if value["external"]:
            path = Path(value["path"])
        else:
            base = self.output_root if root is None else _path(root, "root")
            if base is None:
                raise ValueError("A root is required to resolve a relative source.")
            path = base / value["path"]
        return ScanSource(path, value["format"], value["entry"])

    def to_dict(self):
        """Return the complete JSON-safe manifest document."""
        points = {str(point_id): deepcopy(record)
                  for point_id, record in self._records.items()}
        plan = _json_copy(self.plan.to_dict(), "plan")
        _json_copy(points, "points")
        fixed_parameters = _mapping(self._fixed_parameters, "fixed_parameters")
        parameter_snapshot = _mapping(self._parameter_snapshot, "parameter_snapshot")
        if set(fixed_parameters).intersection(self.plan.parameter_names):
            raise ValueError("Fixed parameters cannot contain scanned parameters.")
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "instrument": {
                "fixed_parameters": fixed_parameters,
                "parameter_snapshot": parameter_snapshot,
            },
            "plan": plan,
            "effective_settings": _mapping(self._effective_settings, "effective_settings"),
            "tool": _mapping(self._tool, "tool"),
            "output_root": None if self._output_root is None else str(self._output_root),
            "execution": self.execution,
            "points": points,
            "errors": _messages(self._errors, "errors"),
            "warnings": _messages(self._warnings, "warnings"),
        }

    @classmethod
    def from_dict(cls, value):
        """Validate and reconstruct a manifest document."""
        if not isinstance(value, dict):
            raise TypeError("A manifest must be a mapping.")
        if set(value) != _TOP_LEVEL_KEYS:
            raise ValueError("Manifest keys do not match the manifest schema.")
        if (value["schema"] != SCHEMA or type(value["version"]) is not int
                or value["version"] != VERSION):
            raise ValueError("Unsupported scan manifest schema or version.")
        if not isinstance(value["instrument"], dict) or set(value["instrument"]) != {
                "fixed_parameters", "parameter_snapshot"}:
            raise ValueError("Invalid instrument snapshot.")
        _validate_execution_aggregate(value["execution"])
        try:
            plan = ScanPlan.from_dict(value["plan"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("Invalid scan plan in manifest.") from error
        manifest = cls(plan=plan,
                       fixed_parameters=value["instrument"]["fixed_parameters"],
                       parameter_snapshot=value["instrument"]["parameter_snapshot"],
                       effective_settings=value["effective_settings"],
                       tool=value["tool"], output_root=value["output_root"],
                       errors=value["errors"], warnings=value["warnings"],
                       execution=value["execution"].get("process"))
        if not isinstance(value["points"], dict):
            raise TypeError("Manifest points must be a mapping keyed by point ID.")
        for key, raw_record in value["points"].items():
            if not isinstance(key, str):
                raise TypeError("Manifest point IDs must be strings in JSON form.")
            if not isinstance(raw_record, dict) or set(raw_record) != _POINT_KEYS:
                raise ValueError("Invalid point record for {!r}.".format(key))
            if (raw_record["seed"] is not None and
                    (not isinstance(raw_record["seed"], dict) or
                     set(raw_record["seed"]) != _SEED_KEYS)):
                raise ValueError("Invalid seed record for {!r}.".format(key))
            point_id = _record_id(plan, key)
            manifest.record_point(point_id, source=manifest._source_for_root(raw_record["source"]),
                                  status=raw_record["status"],
                                  data_status=raw_record["data_status"],
                                  seed=raw_record["seed"],
                                  diagnostics=raw_record["diagnostics"])
        if value["execution"] != manifest.execution:
            raise ValueError("Manifest execution aggregate does not match point records.")
        return manifest

    def _source_for_root(self, value):
        """Turn a serialized source into a ScanSource for record_point."""
        source = _source_from_dict(value)
        if source is None:
            return None
        if source["external"]:
            path = source["path"]
        else:
            if self.output_root is None:
                raise ValueError("Relative source requires output_root.")
            path = self.output_root / source["path"]
        return ScanSource(path, source["format"], source["entry"])


__all__ = ["ScanManifest", "ScanSource", "SCHEMA", "VERSION", "POINT_STATUSES",
           "DATA_STATUSES"]
