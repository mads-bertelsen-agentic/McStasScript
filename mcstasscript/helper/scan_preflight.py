"""Non-mutating validation for native McCode scans."""

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping
import math
import numbers
import os
from pathlib import Path
import re
import shutil

from mcstasscript.data.scan_plan import ScanPlan


_SETTING_DEFAULTS = {
    "ncount": 1000000,
    "seed": None,
    "mpi": None,
    "NeXus": False,
    "openacc": False,
    "gravity": False,
    "force_compile": True,
    "increment_folder_name": True,
    "custom_flags": "",
    "suppress_output": False,
    "checks": True,
    "save_comp_pars": False,
    "executable": None,
    "executable_path": None,
    "package_path": None,
}
_SETTING_KEYS = frozenset(_SETTING_DEFAULTS) | {"output_path"}
_BOOLEAN_SETTINGS = frozenset({
    "NeXus", "openacc", "gravity", "force_compile", "increment_folder_name",
    "suppress_output", "checks", "save_comp_pars",
})
_DECIMAL_INTEGER = re.compile(r"[0-9]+\Z")


class _FrozenDict(dict):
    """Dict-compatible read-only snapshot for report metadata."""

    def _readonly(self, *args, **kwargs):
        raise TypeError("Scan preflight metadata is read-only.")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly
    __ior__ = _readonly


def _readonly(value):
    if isinstance(value, Mapping):
        return _FrozenDict({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_readonly(item) for item in value)
    return value


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON state cannot contain NaN or infinity.")
        return value
    if (type(value).__module__.split(".")[0] == "numpy" and
            hasattr(value, "item")):
        if getattr(value, "ndim", 0) != 0:
            raise TypeError("JSON state cannot contain nonscalar numpy arrays.")
        return _json_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON state mapping keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("JSON state cannot contain sets.")
    if isinstance(value, os.PathLike):
        return str(value)
    raise TypeError("JSON state contains unsupported type {}.".format(type(value).__name__))


def _parameters(instrument):
    container = getattr(instrument, "parameters", {})
    values = getattr(container, "parameters", container)
    return values if isinstance(values, Mapping) else {}


def _version(instrument):
    major = getattr(instrument, "mccode_major_version", None)
    if major is None:
        major = getattr(instrument, "mccode_version", None)
    minor = getattr(instrument, "mccode_minor_version", None)
    patch = getattr(instrument, "mccode_patch_version", None)
    if isinstance(major, str):
        match = re.match(r"^([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?", major)
        if match:
            major, minor, patch = (int(match.group(1)),
                                   int(match.group(2) or 0),
                                   match.group(3) or "0")
    if isinstance(major, numbers.Integral) and isinstance(minor, numbers.Integral):
        if patch is None:
            patch = "0"
        match = re.match(r"^[0-9]+", str(patch))
        if match:
            return int(major), int(minor), int(match.group())
    return None


def _type_for(parameter, candidates, current):
    declared = getattr(parameter, "type", None)
    if declared in ("", "double", "int", "string"):
        return "double" if declared == "" else declared
    if declared is not None and declared not in ("double", "int", "string"):
        return None
    values = [value for value in candidates if value is not None]
    if not values and current is not None:
        values = [current]
    if not values:
        return None
    if any(isinstance(value, str) for value in values):
        return "string" if all(isinstance(value, str) for value in values) else None
    if all(isinstance(value, numbers.Real) and not isinstance(value, bool)
           for value in values):
        return "double"
    return None


def _validate_parameter_value(parameter, value, parameter_type, label):
    if value is None:
        return ["{} is unset.".format(label)]
    if isinstance(value, bool):
        return ["{} must not be bool.".format(label)]
    if parameter_type in ("double", "int"):
        if not isinstance(value, numbers.Real):
            return ["{} must be a real number for type {}.".format(label, parameter_type)]
        if not math.isfinite(float(value)):
            return ["{} must be finite.".format(label)]
        if parameter_type == "int" and not float(value).is_integer():
            return ["{} must be an integral real number for type int.".format(label)]
    elif parameter_type == "string":
        if not isinstance(value, str):
            return ["{} must be a string.".format(label)]
    else:
        return ["{} has no supported McCode type.".format(label)]

    try:
        candidate = deepcopy(parameter)
        candidate.value = value
        legal = candidate.is_legal() if callable(getattr(candidate, "is_legal", None)) else True
        if not legal:
            return ["{} violates its parameter constraints.".format(label)]
    except Exception as error:
        return ["{} is illegal: {}".format(label, error)]
    return []


def _settings(instrument, overrides, errors):
    stored = getattr(instrument, "_run_settings", {})
    stored = stored if isinstance(stored, Mapping) else {}
    effective = deepcopy(_SETTING_DEFAULTS)
    for key in _SETTING_KEYS:
        if key in stored:
            effective[key] = deepcopy(stored[key])
    if "executable" not in stored and effective["executable"] is None:
        effective["executable"] = "mcrun" if getattr(
            instrument, "package_name", "") == "McStas" else "mxrun"
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        errors.append("settings must be a mapping or None.")
        overrides = {}
    for key in overrides:
        if key not in _SETTING_KEYS:
            errors.append("Unsupported scan preflight setting: {!r}.".format(key))
        else:
            effective[key] = deepcopy(overrides[key])
    effective["run_path"] = deepcopy(stored.get(
        "run_path", getattr(instrument, "input_path", ".")))
    return effective


def _validate_settings(effective, errors):
    ncount = effective["ncount"]
    if isinstance(ncount, bool) or not isinstance(ncount, numbers.Real):
        errors.append("ncount must be a positive integral number.")
    elif not math.isfinite(float(ncount)) or ncount <= 0 or not float(ncount).is_integer():
        errors.append("ncount must be a positive integral number.")
    else:
        effective["ncount"] = int(ncount)

    seed = effective["seed"]
    if seed is not None:
        valid_seed = ((isinstance(seed, numbers.Integral) and not isinstance(seed, bool)) or
                      (isinstance(seed, str) and _DECIMAL_INTEGER.fullmatch(seed)))
        if not valid_seed or int(seed) <= 0:
            errors.append("seed must be None, a positive integer, or a decimal integer string.")

    mpi = effective["mpi"]
    if isinstance(mpi, str):
        if mpi != "auto":
            errors.append("mpi must be None, 'auto', or a positive integer.")
    elif mpi is not None and (isinstance(mpi, bool) or
                              not isinstance(mpi, numbers.Integral) or mpi <= 0):
        errors.append("mpi must be None, 'auto', or a positive integer.")

    for key in _BOOLEAN_SETTINGS:
        if not isinstance(effective[key], bool):
            errors.append("{} must be a bool.".format(key))
    flags = effective["custom_flags"]
    if flags is not None and not isinstance(flags, str):
        errors.append("custom_flags must be a string or None.")
    elif flags != "":
        errors.append("Nonempty custom_flags cannot be safely preflighted.")


def _same_runner_value(left, right, path=False):
    if path:
        try:
            left = os.fspath(left)
            right = os.fspath(right)
        except TypeError:
            return left == right
        if left == "" or right == "":
            return _safe_equal(left, right)
        return os.path.abspath(left) == os.path.abspath(right)
    return _safe_equal(left, right)


def _safe_equal(left, right):
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    if (type(result).__module__.split(".")[0] == "numpy" and
            type(result).__name__ == "bool_"):
        return bool(result)
    return False


def _validate_runner(instrument, overrides, effective, errors):
    stored = getattr(instrument, "_run_settings", {})
    if not isinstance(stored, Mapping):
        stored = {}
    if not isinstance(overrides, Mapping):
        overrides = {}
    valid = True
    for key, path in (("executable", False), ("executable_path", True),
                      ("package_path", True)):
        if key in overrides and (key not in stored or
                                 not _same_runner_value(overrides[key], stored[key], path)):
            errors.append("{} override does not match the cached instrument runner."
                          .format(key))
            valid = False

    package = getattr(instrument, "package_name", "")
    expected = {"McStas": "mcrun", "McXtrace": "mxrun"}.get(package)
    executable = effective.get("executable")
    if not isinstance(executable, str) or not executable:
        errors.append("executable must be a nonempty runner name.")
        valid = False
    elif expected is None or executable != expected:
        errors.append("executable must be {} for {}.".format(
            expected or "mcrun or mxrun", package or "the configured package"))
        valid = False

    configured_path = effective.get("executable_path", "")
    resolved = None
    if configured_path is None:
        errors.append("executable_path must be an existing directory.")
        valid = False
    else:
        try:
            configured_path = os.fspath(configured_path)
            if "\x00" in os.fsdecode(configured_path):
                raise ValueError("executable_path cannot contain NUL characters")
            if not configured_path or not os.path.isdir(configured_path):
                errors.append("executable_path must be an existing directory: {}."
                              .format(configured_path))
                valid = False
            elif isinstance(executable, str) and executable:
                resolved = shutil.which(executable, path=configured_path)
                if resolved is None:
                    errors.append("Configured runner executable was not found: {}."
                                  .format(executable))
                    valid = False
                else:
                    resolved = os.path.abspath(resolved)
        except (TypeError, ValueError, OSError) as error:
            errors.append("Invalid executable_path: {}.".format(error))
            valid = False

    package_path = effective.get("package_path")
    if package_path is not None:
        try:
            package_path = os.fspath(package_path)
            if "\x00" in os.fsdecode(package_path):
                raise ValueError("package_path cannot contain NUL characters")
            if package_path and not os.path.isdir(package_path):
                errors.append("package_path must be an existing directory: {}."
                              .format(package_path))
                valid = False
        except (TypeError, ValueError, OSError) as error:
            errors.append("Invalid package_path: {}.".format(error))
            valid = False
    return valid, resolved


def _validate_run_path(effective, errors):
    raw = effective.get("run_path")
    try:
        raw = os.fspath(raw)
        if "\x00" in os.fsdecode(raw):
            raise ValueError("run_path cannot contain NUL characters")
        path = Path(os.path.abspath(raw))
    except (TypeError, ValueError, OSError) as error:
        errors.append("Invalid run_path: {}.".format(error))
        return
    if not path.is_dir():
        errors.append("run_path must be an existing directory: {}.".format(path))


def _output_path(instrument, effective, errors):
    raw = effective.get("output_path")
    if raw is None:
        raw = getattr(instrument, "output_path", None)
    try:
        raw = os.fspath(raw)
        if "\x00" in os.fsdecode(raw):
            raise ValueError("output_path cannot contain NUL characters")
        output = Path(os.path.abspath(raw))
    except (TypeError, ValueError, OSError) as error:
        errors.append("Invalid output_path: {}.".format(error))
        return None
    effective["output_path"] = str(output)
    if output.exists():
        errors.append("output_path already exists: {}.".format(output))
        return str(output)
    parent = output.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        errors.append("The nearest existing output parent is not a directory: {}.".format(parent))
    elif not os.access(str(parent), os.W_OK):
        errors.append("The output parent is not writable: {}.".format(parent))
    return str(output)


def _capabilities(instrument, plan, effective, runner_valid, resolved_runner,
                  errors, warnings):
    package = str(getattr(instrument, "package_name", ""))
    executable = effective.get("executable")
    version = _version(instrument)
    version_text = ".".join(map(str, version)) if version else None
    capabilities = {}
    if runner_valid and package == "McStas" and version == (3, 7, 14):
        capabilities = {"grid_numeric": True, "grid_categorical": False,
                        "paired": True, "points": True, "nexus": True,
                        "seed_policy": "sequential_reuse"}
        warnings.append("McStas 3.7.14 reuses the base seed for sequential points; "
                        "preflight records this policy but does not promise execution behavior.")
    elif runner_valid and package == "McStas" and version == (3, 10, 0):
        capabilities = {"grid_numeric": True, "grid_categorical": True,
                        "paired": True, "points": True, "nexus": True,
                        "seed_policy": "point_offset_1024"}
    elif package == "McStas" and version not in ((3, 7, 14), (3, 10, 0)):
        errors.append("Scan capabilities cannot be verified for cached McStas version {}."
                      .format(version_text or "unknown"))
    elif package != "McStas":
        errors.append("No verified native scan capabilities are available for {}."
                      .format(package or "unknown McCode"))
    if plan.kind == "grid" and capabilities.get("grid_categorical") is False:
        if any(isinstance(value, str) for name in plan.parameter_names
               for value in plan.axes[name]):
            errors.append("McStas 3.7.14 does not have verified categorical grid support.")
    return {
        "name": package.lower() if package else None,
        "package": package or None,
        "executable": executable,
        "executable_path": effective.get("executable_path"),
        "version": version_text,
        "capabilities": capabilities,
        "capability_evidence": "cached instrument version; verified fixtures",
        "path_evidence": {
            "configured_executable_path": effective.get("executable_path"),
            "resolved_executable": resolved_runner,
            "checked_without_execution": True,
            "note": "Point-in-time evidence from mutable instrument settings.",
        },
    }


@dataclass(frozen=True)
class ScanPreflight:
    plan: ScanPlan
    fixed_parameters: dict
    parameter_snapshot: dict
    effective_settings: dict
    tool: dict
    output_path: str
    errors: tuple = ()
    warnings: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "fixed_parameters",
                           _readonly(deepcopy(self.fixed_parameters)))
        object.__setattr__(self, "parameter_snapshot",
                           _readonly(deepcopy(self.parameter_snapshot)))
        object.__setattr__(self, "effective_settings",
                           _readonly(deepcopy(self.effective_settings)))
        object.__setattr__(self, "tool", _readonly(deepcopy(self.tool)))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def valid(self):
        return not self.errors

    def raise_for_errors(self):
        if self.errors:
            raise ValueError("Scan preflight failed:\n" + "\n".join(self.errors))

    def to_dict(self):
        return {
            "plan": self.plan.to_dict(),
            "fixed_parameters": _json_value(self.fixed_parameters),
            "parameter_snapshot": _json_value(self.parameter_snapshot),
            "effective_settings": _json_value(self.effective_settings),
            "tool": _json_value(self.tool),
            "output_path": _json_value(self.output_path),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "valid": self.valid,
        }


def scan_preflight(instrument, plan, *, settings=None):
    """Return a detached validation report without executing or writing."""
    if not isinstance(plan, ScanPlan):
        raise TypeError("plan must be a ScanPlan.")
    errors = []
    warnings = []

    parameters = _parameters(instrument)
    snapshot = {}
    for name, parameter in parameters.items():
        try:
            snapshot[name] = deepcopy(parameter.value)
        except Exception as error:
            snapshot[name] = None
            errors.append("Could not read parameter {!r}: {}.".format(name, error))

    scanned = set(plan.parameter_names)
    for name in scanned - set(parameters):
        errors.append("Unknown scan parameter: {!r}.".format(name))

    candidates = {name: [] for name in plan.parameter_names}
    if plan.kind in ("grid", "paired"):
        for name in plan.parameter_names:
            candidates[name] = list(plan.axes[name])
    else:
        for point_id, point in enumerate(plan):
            for name in plan.parameter_names:
                candidates[name].append(point.parameters[name])

    for name, parameter in parameters.items():
        if name not in scanned and snapshot.get(name) is None:
            errors.append("Fixed parameter {!r} is unset.".format(name))
            continue
        parameter_type = _type_for(parameter, candidates.get(name, []), snapshot.get(name))
        if parameter_type is None:
            errors.append("Parameter {!r} has no supported or inferable McCode type.".format(name))
            continue
        if name not in scanned:
            errors.extend(_validate_parameter_value(
                parameter, snapshot[name], parameter_type,
                "Fixed parameter {!r}".format(name)))
        else:
            for value in candidates[name]:
                errors.extend(_validate_parameter_value(
                    parameter, value, parameter_type,
                    "Scan parameter {!r} value {!r}".format(name, value)))

    effective = _settings(instrument, settings, errors)
    _validate_settings(effective, errors)
    runner_valid, resolved_runner = _validate_runner(
        instrument, settings, effective, errors)
    _validate_run_path(effective, errors)
    output_path = _output_path(instrument, effective, errors)
    if any(getattr(instrument, name, None) is not None
           for name in ("run_from_ref", "run_to_ref", "run_from_component_parameters",
                        "run_to_component_parameters")):
        errors.append("Partial instrument runs via run_from/run_to are not supported by scan preflight.")
    tool = _capabilities(instrument, plan, effective, runner_valid, resolved_runner,
                         errors, warnings)

    fixed = {name: snapshot.get(name) for name in parameters if name not in scanned}
    detached_plan = ScanPlan.from_dict(plan.to_dict())
    return ScanPreflight(plan=detached_plan,
                         fixed_parameters=deepcopy(fixed),
                         parameter_snapshot=deepcopy(snapshot),
                         effective_settings=deepcopy(effective),
                         tool=deepcopy(tool), output_path=output_path,
                         errors=tuple(str(error) for error in errors),
                         warnings=tuple(str(warning) for warning in warnings))
