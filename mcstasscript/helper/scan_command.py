"""Pure construction of native McCode scan command arguments."""

from collections.abc import Mapping
import math
import numbers
import os
import re

from mcstasscript.data.scan_plan import ScanPlan


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SUPPORTED_VERSION = "3.10.0"
_NATIVE_SEED_MAX = 2 ** 63 - 1
_NATIVE_SEED_STEP = 1024


def _check_index(plan, value, label):
    if not isinstance(plan, ScanPlan):
        raise TypeError("plan must be a ScanPlan.")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("{} must be an integer.".format(label))
    value = int(value)
    if value < 0 or value >= plan.point_count:
        raise IndexError("{} is outside the scan plan.".format(label))
    return value


def native_index(plan, point_id):
    """Return the native point index for a canonical plan point ID."""
    point_id = _check_index(plan, point_id, "point_id")
    if plan.kind != "grid":
        return point_id

    axes = plan.axes
    indices = [0] * len(plan.parameter_names)
    remainder = point_id
    # ScanPlan uses the last axis as the least-significant digit.
    for axis_number in range(len(plan.parameter_names) - 1, -1, -1):
        name = plan.parameter_names[axis_number]
        remainder, indices[axis_number] = divmod(remainder, len(axes[name]))

    result = 0
    stride = 1
    # Native -M uses the first command-line axis as the least-significant digit.
    for axis_number, name in enumerate(plan.parameter_names):
        result += indices[axis_number] * stride
        stride *= len(axes[name])
    return result


def plan_point_id(plan, native_index):
    """Return the canonical plan point ID for a native point index."""
    native_index = _check_index(plan, native_index, "native_index")
    if plan.kind != "grid":
        return native_index

    axes = plan.axes
    indices = [0] * len(plan.parameter_names)
    remainder = native_index
    for axis_number, name in enumerate(plan.parameter_names):
        remainder, indices[axis_number] = divmod(remainder, len(axes[name]))

    result = 0
    stride = 1
    for axis_number in range(len(plan.parameter_names) - 1, -1, -1):
        name = plan.parameter_names[axis_number]
        result += indices[axis_number] * stride
        stride *= len(axes[name])
    return result


def _text(value, label):
    try:
        value = os.fsdecode(value)
    except (TypeError, ValueError):
        raise TypeError("{} must be a path-like string.".format(label))
    if (any(ord(character) < 32 or ord(character) == 127 for character in value)
            or value != "".join(value.splitlines())):
        raise ValueError("{} cannot contain control or linebreak characters.".format(label))
    return value


def _scalar_text(value, label):
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
        raise TypeError("{} must be a string or real number.".format(label))
    if isinstance(value, str):
        value = _text(value, label)
        if value != value.strip():
            raise ValueError(
                "{} has leading or trailing whitespace; native metadata strips it."
                .format(label))
        return value
    if not math.isfinite(float(value)):
        raise ValueError("{} must be finite.".format(label))
    return str(value)


def _native_value(value, label, list_mode):
    value = _scalar_text(value, label)
    if isinstance(value, str) and value == "":
        raise ValueError("{} cannot be empty in native execution.".format(label))
    if not list_mode or not isinstance(value, str):
        return value
    if value.endswith("\\"):
        raise ValueError(
            "{} cannot end with '\\' in native list mode; it would escape the delimiter."
            .format(label))
    if ":" in value:
        raise ValueError(
            "{} contains ':' and cannot be represented safely in native list mode."
            .format(label))
    # McStas 3.10 splits on unescaped commas and then removes the escape.
    return value.replace(",", r"\,")


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping.".format(label))
    return value


def _report_value(report, name):
    try:
        return getattr(report, name)
    except AttributeError:
        raise TypeError("report must be a ScanPreflight report.")


def _validated_report(report):
    if getattr(report, "errors", ()):
        raise ValueError("Cannot build a command from an invalid scan preflight.")
    plan = _report_value(report, "plan")
    if not isinstance(plan, ScanPlan):
        raise TypeError("report.plan must be a ScanPlan.")

    tool = _mapping(_report_value(report, "tool"), "report.tool")
    package = tool.get("package")
    version = tool.get("version")
    if package != "McStas":
        raise ValueError("Native scan execution is not verified for {}."
                         .format(package or "the configured McCode tool"))
    if version != _SUPPORTED_VERSION:
        raise ValueError(
            "Native scan execution requires McStas {} (found {})."
            .format(_SUPPORTED_VERSION, version or "unknown"))

    evidence = _mapping(tool.get("path_evidence"), "report.tool.path_evidence")
    runner = _text(evidence.get("resolved_executable"), "resolved executable")
    if not os.path.isabs(runner):
        raise ValueError("report.tool.path_evidence.resolved_executable must be absolute.")
    return plan, runner


def _validate_names(plan, fixed):
    names = list(fixed) + list(plan.parameter_names)
    if len(set(names)) != len(names):
        raise ValueError("Fixed and scanned parameter names must be distinct.")
    for name in names:
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError("Invalid McCode parameter name: {!r}.".format(name))
        _text(name, "parameter name")


def _settings(report):
    settings = _mapping(_report_value(report, "effective_settings"),
                        "report.effective_settings")
    ncount = settings.get("ncount")
    if isinstance(ncount, bool) or not isinstance(ncount, numbers.Real):
        raise ValueError("effective ncount must be a positive integer.")
    if not math.isfinite(float(ncount)) or ncount <= 0 or not float(ncount).is_integer():
        raise ValueError("effective ncount must be a positive integer.")

    for key in ("NeXus", "openacc", "gravity"):
        if not isinstance(settings.get(key, False), bool):
            raise ValueError("effective {} must be a bool.".format(key))

    mpi = settings.get("mpi")
    if isinstance(mpi, str):
        if mpi != "auto":
            raise ValueError("effective mpi must be None, 'auto', or a positive integer.")
    elif mpi is not None and (isinstance(mpi, bool) or
                              not isinstance(mpi, numbers.Integral) or mpi <= 0):
        raise ValueError("effective mpi must be None, 'auto', or a positive integer.")

    custom_flags = settings.get("custom_flags", "")
    if custom_flags not in (None, ""):
        raise ValueError("Nonempty custom_flags cannot be serialized safely.")
    return settings, int(ncount), mpi


def _scan_arguments(plan, list_mode):
    if not list_mode:
        point = plan[0].parameters
        return [(name, _native_value(point[name], "scan parameter {!r}".format(name), False))
                for name in plan.parameter_names]

    if plan.kind == "grid":
        # Native -M performs the Cartesian expansion. Do not materialize the grid.
        return [(name, ",".join(
            _native_value(value, "scan parameter {!r}".format(name), True)
            for value in plan.axes[name])) for name in plan.parameter_names]

    rows = [plan[index].parameters for index in range(plan.point_count)]
    return [(name, ",".join(
        _native_value(row[name], "scan parameter {!r}".format(name), True)
        for row in rows)) for name in plan.parameter_names]


def build_scan_command(report, instrument_file, native_path, *, seed):
    """Build a shell-free McStas 3.10.0 native scan command."""
    plan, runner = _validated_report(report)
    fixed = _mapping(_report_value(report, "fixed_parameters"),
                     "report.fixed_parameters")
    _validate_names(plan, fixed)
    settings, ncount, mpi = _settings(report)

    instrument_file = os.path.abspath(_text(instrument_file, "instrument_file"))
    native_path = os.path.abspath(_text(native_path, "native_path"))
    if not instrument_file or not native_path:
        raise ValueError("Native scan paths must be nonempty.")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        if not (isinstance(seed, str) and seed.isdecimal()):
            raise ValueError("seed must be a positive integer.")
    if int(seed) <= 0:
        raise ValueError("seed must be a positive integer.")
    if int(seed) > _NATIVE_SEED_MAX - _NATIVE_SEED_STEP * (plan.point_count - 1):
        raise ValueError(
            "seed is too large for the native signed-64 seed budget across the scan.")

    list_mode = plan.point_count > 1
    arguments = [(name, _native_value(value, "fixed parameter {!r}".format(name),
                                       list_mode)) for name, value in fixed.items()]
    arguments.extend(_scan_arguments(plan, list_mode))

    command = [runner, "--force-compile",
               "--format={}".format("NeXus" if settings.get("NeXus", False)
                                    else "McCode"),
               "--seed={}".format(seed), "--ncount={}".format(ncount)]
    if mpi is not None:
        command.append("--mpi={}".format(mpi))
    if settings.get("openacc", False):
        command.append("--openacc")
    if settings.get("gravity", False):
        command.append("-g")
    command.extend(["--dir={}".format(native_path), instrument_file])
    if list_mode:
        command.append("-L")
        if plan.kind == "grid":
            command.append("-M")
    command.extend("{}={}".format(name, value) for name, value in arguments)
    return command


__all__ = ["build_scan_command", "native_index", "plan_point_id"]
