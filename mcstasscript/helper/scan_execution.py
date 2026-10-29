"""Execution owner for verified McStas 3.10.0 native scans."""

from copy import deepcopy
import ast
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import threading

from mcstasscript.data.scan import ScanResult
from mcstasscript.data.scan_manifest import ScanManifest
from mcstasscript.data.scan_plan import ScanPlan
from mcstasscript.helper.scan_preflight import ScanPreflight, _json_value
from mcstasscript.helper.mcstas_objects import provide_parameter
from mcstasscript.data import scan_persistence
from mcstasscript.helper import scan_command


class ScanExecutionError(RuntimeError):
    """A scan could not complete or its required output was not readable."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


_SUPPORTED_SETTINGS = frozenset({
    "ncount", "seed", "mpi", "NeXus", "openacc", "gravity", "force_compile",
    "increment_folder_name", "custom_flags", "suppress_output", "checks",
    "save_comp_pars", "executable", "executable_path", "package_path",
    "output_path", "run_path", "parameters",
})


def _safe_name(name):
    value = str(name)
    reserved = {"auto", "break", "case", "char", "const", "continue", "default",
                "do", "double", "else", "enum", "extern", "float", "for", "goto",
                "if", "int", "long", "register", "return", "short", "signed",
                "sizeof", "static", "struct", "switch", "typedef", "union",
                "unsigned", "void", "volatile", "while"}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) or value in reserved:
        raise ValueError("Instrument name must be a non-reserved C identifier: {!r}."
                         .format(name))
    return value


def _result(plan, *, execution=None, metadata=None, points=()):
    return ScanResult(points, parameter_names=plan.parameter_names, plan=plan,
                      metadata=metadata or {}, execution=execution or {})


def _canonical_fixed_string(value, name):
    """Convert a supported source string literal to its native runtime value."""
    if not isinstance(value, str):
        raise TypeError("Fixed string parameter {!r} must be a string.".format(name))
    if value.startswith('"') or value.endswith('"'):
        if not re.fullmatch(r'"(?:\\.|[^"\\])*"', value):
            raise ValueError("Fixed string parameter {!r} is not a supported C string literal."
                             .format(name))
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError("Fixed string parameter {!r} has invalid C escapes."
                             .format(name)) from error
        if not isinstance(value, str):
            raise ValueError("Fixed string parameter {!r} is not a string literal."
                             .format(name))
    elif any(character in value for character in
             ("\\", "'", "+", "-", "*", "/", "%", "(", ")", "[", "]",
              "{", "}", ";", "&", "|", "!", "<", ">", "?", ":", "=")):
        raise ValueError("Fixed string parameter {!r} contains an unsupported C expression."
                         .format(name))
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("Fixed string parameter {!r} contains an unsupported control character."
                         .format(name))
    return value


def _canonical_fixed_parameters(instrument, report):
    parameters = getattr(getattr(instrument, "parameters", None), "parameters", {})
    fixed = {}
    for name, value in report.fixed_parameters.items():
        parameter = parameters.get(name)
        parameter_type = getattr(parameter, "type", None)
        if parameter_type == "string" or (parameter_type in (None, "") and
                                           isinstance(value, str)):
            fixed[name] = _canonical_fixed_string(value, name)
        else:
            fixed[name] = value
    return fixed


def _point_has_issue(point):
    """Return whether required native output for one point is unusable."""
    if point.diagnostics:
        return True
    if point.data_status in ("unreadable", "partial"):
        return True
    if point.data_status == "missing" and point.source is not None:
        return True
    summary_usable = any(
        value is not None and math.isfinite(float(value))
        for summary in point.summaries.values()
        for value in (summary.total_I, summary.total_E, summary.total_N)
    )
    full_usable = point.is_loaded and bool(point.data)
    return not summary_usable and not full_usable


def _attach_execution(result, execution, diagnostics=()):
    metadata = deepcopy(result.metadata)
    if diagnostics:
        metadata["execution_diagnostics"] = list(diagnostics)
    return ScanResult(result.points, parameter_names=result.parameter_names,
                      plan=result.plan, metadata=metadata, execution=execution)


def _copy_report(report, settings, *, fixed_parameters=None):
    document = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    if fixed_parameters is None:
        fixed_parameters = document["fixed_parameters"]
    else:
        fixed_parameters = _json_value(dict(fixed_parameters))
    effective = dict(document["effective_settings"])
    effective.update({key: _json_value(value) for key, value in settings.items()})
    return ScanPreflight(
        plan=ScanPlan.from_dict(document["plan"]),
        fixed_parameters=fixed_parameters,
        parameter_snapshot=document["parameter_snapshot"],
        effective_settings=effective,
        tool=document["tool"], output_path=document["output_path"],
        errors=document["errors"], warnings=document["warnings"])


def _set_copy_values(working, plan):
    parameters = getattr(getattr(working, "parameters", None), "parameters", {})
    first = plan[0].parameters
    scanned = set(plan.parameter_names)
    for name, parameter in parameters.items():
        if getattr(parameter, "type", None) in (None, ""):
            if name in scanned:
                if plan.kind in ("grid", "paired"):
                    candidate = plan.axes[name][0]
                else:
                    candidate = first[name]
                parameter.type = "string" if isinstance(candidate, str) else "double"
            elif getattr(parameter, "type", None) is None and isinstance(parameter.value, str):
                parameter.type = "string"
        if parameter.type == "string":
            replacement = provide_parameter(
                "string", name, value='"__mcstasscript_scan__"',
                comment=getattr(parameter, "comment", "") or "",
                unit=getattr(parameter, "unit", "") or "")
            parameters[name] = replacement
        elif name in scanned:
            parameter.value = first[name]


def _stop_process(process):
    if process is None:
        return
    try:
        pid = process.pid
    except AttributeError:
        pid = None
    group = pid is not None and pid != os.getpid()
    try:
        if not group:
            raise OSError("process group is unavailable")
        os.killpg(pid, signal.SIGTERM)
    except (AttributeError, OSError):
        try:
            process.terminate()
        except (AttributeError, KeyboardInterrupt, OSError):
            pass
    try:
        process.wait(timeout=5)
    except (AttributeError, KeyboardInterrupt, subprocess.TimeoutExpired, OSError):
        pass
    # The leader may have exited while descendants remain in its session.
    # Escalate the process group, not just the leader, after the grace period.
    try:
        if group:
            os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
    except (AttributeError, KeyboardInterrupt, OSError):
        pass
    try:
        process.wait()
    except (AttributeError, KeyboardInterrupt, OSError):
        pass


def _checkpoint(manifest, root):
    return scan_persistence.write_manifest(manifest, root)


def _final_result(manifest, result, diagnostics=()):
    return _attach_execution(result, manifest.execution, diagnostics)


def _close_log(log_handle):
    if log_handle is None:
        return
    try:
        if not log_handle.closed:
            log_handle.close()
    except (AttributeError, KeyboardInterrupt, OSError):
        pass


def _ignore_sigint():
    if threading.current_thread() is not threading.main_thread():
        return False, None
    try:
        return True, signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        return False, None


def _partial_result(manifest, root, diagnostics=()):
    try:
        result = scan_persistence.reconcile_scan(manifest, root, inspect_data=False)
    except KeyboardInterrupt as error:
        diagnostics = list(diagnostics) + [str(error)]
        result = _result(manifest.plan)
    except Exception as error:
        diagnostics = list(diagnostics) + [str(error)]
        result = _result(manifest.plan)
    return _final_result(manifest, result, diagnostics)


def _association_diagnostics(result):
    metadata = result.metadata
    values = list(metadata.get("association_issues", ()))
    if metadata.get("native_data_status") == "unreadable":
        values.append("Native output data is unreadable.")
    return [str(value) for value in values]


def _interrupt(manifest, root, interrupt, process=None, log_handle=None):
    diagnostics = []
    result = _result(manifest.plan)
    handler_changed, previous_handler = _ignore_sigint()
    try:
        try:
            _stop_process(process)
        except (KeyboardInterrupt, OSError) as error:
            diagnostics.append("Process cleanup: {}".format(error))
        _close_log(log_handle)

        try:
            manifest.record_execution(status="interrupted", outcome="interrupted",
                                      log="execution.log", diagnostics=diagnostics)
            _checkpoint(manifest, root)
        except KeyboardInterrupt:
            diagnostics.append("Interrupted while checkpointing interrupted execution.")
            result = _partial_result(manifest, root, diagnostics)
        except Exception as error:
            diagnostics.append(str(error))
            result = _partial_result(manifest, root, diagnostics)
        else:
            try:
                manifest, finalized = scan_persistence.finalize_manifest(manifest, root)
            except KeyboardInterrupt:
                diagnostics.append("Interrupted while inspecting native output.")
                result = _partial_result(manifest, root, diagnostics)
            except Exception as error:
                diagnostics.append(str(error))
                result = _partial_result(manifest, root, diagnostics)
            else:
                result = _final_result(manifest, finalized)
            try:
                _checkpoint(manifest, root)
            except KeyboardInterrupt:
                diagnostics.append("Interrupted while checkpointing interrupted execution.")
                result = _attach_execution(result, manifest.execution, diagnostics)
            except Exception as error:
                diagnostics.append(str(error))
                result = _attach_execution(result, manifest.execution, diagnostics)
    finally:
        if handler_changed:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except (KeyboardInterrupt, OSError, ValueError) as error:
                diagnostics.append("Could not restore SIGINT handler: {}".format(error))
    interrupt.result = result
    raise interrupt


def _execution_failure(message, result, on_error):
    error = ScanExecutionError(message, result)
    if on_error == "continue":
        return result
    raise error


def execute_scan(instrument, plan, *, on_error="raise"):
    """Execute one native scan without changing the supplied instrument."""
    if on_error not in ("raise", "continue"):
        raise ValueError("on_error must be 'raise' or 'continue'.")
    if not isinstance(plan, ScanPlan):
        raise TypeError("plan must be a ScanPlan.")

    report = instrument.scan_preflight(plan)
    report.raise_for_errors()
    # This is deliberately before deepcopy and before the output root exists.
    report.to_dict()
    stored = getattr(instrument, "_run_settings", {})
    unsupported = set(stored) - _SUPPORTED_SETTINGS
    if unsupported:
        raise ValueError("Unsupported instrument run setting(s) for native scan: {}."
                         .format(", ".join(sorted(unsupported))))
    if report.tool.get("package") != "McStas" or report.tool.get("version") != "3.10.0":
        raise ValueError("Native scan execution requires exactly McStas 3.10.0; "
                         "older seed-reuse runtimes are not supported.")

    max_seed = min(2 ** 31 - 1,
                   (2 ** 63 - 1) - 1024 * (plan.point_count - 1))
    if max_seed <= 0:
        raise ValueError("Scan plan is too large for native signed 64-bit seed offsets.")
    effective = dict(report.effective_settings)
    seed = effective.get("seed")
    if seed is None:
        seed = secrets.randbelow(max_seed) + 1
    else:
        seed = int(seed)
    effective["seed"] = seed

    root = Path(report.output_path)
    safe_name = _safe_name(getattr(instrument, "name", "instrument"))
    working = deepcopy(instrument)
    # The isolated source has no reusable binary; the native builder therefore
    # deliberately forces a compile and the manifest records that effective fact.
    effective["force_compile"] = True
    fixed_parameters = _canonical_fixed_parameters(instrument, report)
    command_report = _copy_report(report, effective, fixed_parameters=fixed_parameters)
    instrument_file = root / "source" / (safe_name + ".instr")
    native_path = root / "native"
    command = scan_command.build_scan_command(
        command_report, str(instrument_file), str(native_path), seed=seed)
    if (not isinstance(command, list) or
            any(not isinstance(argument, str) or not argument or "\x00" in argument
                for argument in command)):
        raise ValueError("Native scan command is not serializable.")
    json.dumps(command, allow_nan=False)
    effective["command"] = list(command)
    effective["instrument_file"] = str(instrument_file)
    effective["native_path"] = str(native_path)
    final_report = _copy_report(report, effective, fixed_parameters=fixed_parameters)
    manifest = ScanManifest.from_preflight(final_report)

    # Nothing below this point can silently replace an existing scan.
    try:
        root.mkdir(parents=True)
    except FileExistsError as error:
        raise ValueError("output_path already exists: {}".format(root)) from error
    source = root / "source"
    try:
        source.mkdir()
        manifest.record_execution(status="pending", outcome="pending",
                                  log="execution.log")
        _checkpoint(manifest, root)
    except KeyboardInterrupt as interrupt:
        _interrupt(manifest, root, interrupt)
    except Exception as error:
        result = _partial_result(manifest, root, [str(error)])
        return _execution_failure(
            "Could not create the initial scan checkpoint: {}".format(error),
            result, on_error)

    try:
        working.name = safe_name
        working.input_path = str(source)
        working._run_settings["run_path"] = str(Path(
            os.path.abspath(final_report.effective_settings["run_path"])))
        working._run_settings["checks"] = final_report.effective_settings["checks"]
        working._run_settings["save_comp_pars"] = final_report.effective_settings["save_comp_pars"]
        _set_copy_values(working, plan)
        working.write_full_instrument()
    except KeyboardInterrupt as interrupt:
        _interrupt(manifest, root, interrupt)
    except Exception as error:
        manifest.record_execution(status="failed", outcome="write_failed",
                                  log="execution.log", diagnostics=[str(error)])
        try:
            _checkpoint(manifest, root)
        except Exception:
            pass
        result = _partial_result(manifest, root, [str(error)])
        return _execution_failure("Could not write the copied instrument: {}".format(error),
                                  result, on_error)

    log_handle = None
    process = None
    try:
        log_handle = (root / "execution.log").open("w", encoding="utf-8")
        manifest.record_execution(status="running", outcome="running",
                                  log="execution.log")
        _checkpoint(manifest, root)
        process = subprocess.Popen(
            command, shell=False, stdout=log_handle, stderr=subprocess.STDOUT,
            cwd=str(Path(os.path.abspath(final_report.effective_settings["run_path"]))),
            start_new_session=True)
        returncode = process.wait()
    except KeyboardInterrupt as interrupt:
        _interrupt(manifest, root, interrupt, process, log_handle)
    except Exception as error:
        try:
            _stop_process(process)
        except (KeyboardInterrupt, OSError):
            pass
        _close_log(log_handle)
        manifest.record_execution(status="failed", outcome="launch_failed",
                                  log="execution.log", diagnostics=[str(error)])
        try:
            _checkpoint(manifest, root)
        except Exception:
            pass
        result = _partial_result(manifest, root, [str(error)])
        return _execution_failure("Could not launch the native scan: {}".format(error),
                                  result, on_error)
    finally:
        _close_log(log_handle)

    diagnostics = []
    try:
        manifest.record_execution(
            status="failed" if returncode else "succeeded",
            outcome="nonzero_exit" if returncode else "completed",
            returncode=returncode, log="execution.log")
        _checkpoint(manifest, root)
        manifest, result = scan_persistence.finalize_manifest(manifest, root)
        if returncode:
            diagnostics.append("Native scanner returned {}.".format(returncode))
        diagnostics.extend(_association_diagnostics(result))
        if len(manifest.records) != plan.point_count:
            diagnostics.append("Native output did not contain every planned point.")
        for point in result.points:
            if _point_has_issue(point):
                diagnostics.append("Point {!r} has missing, unreadable, or partial output."
                                   .format(point.point_id))
        if diagnostics:
            outcome = "completed_with_output_issues" if not returncode else "nonzero_exit"
            manifest.record_execution(
                status="failed" if returncode else "succeeded", outcome=outcome,
                returncode=returncode, log="execution.log", diagnostics=diagnostics)
        _checkpoint(manifest, root)
        result = _final_result(manifest, result, diagnostics)
    except KeyboardInterrupt as interrupt:
        _interrupt(manifest, root, interrupt)
    except Exception as error:
        diagnostics.append(str(error))
        try:
            manifest.record_execution(status="failed", outcome="finalization_failed",
                                      returncode=returncode, log="execution.log",
                                      diagnostics=diagnostics)
            _checkpoint(manifest, root)
        except Exception:
            pass
        result = _partial_result(manifest, root, diagnostics)
        return _execution_failure("Could not finalize the native scan: {}".format(error),
                                  result, on_error)

    if diagnostics and on_error == "raise":
        raise ScanExecutionError("Native scan completed with output issues: "
                                 + " ".join(diagnostics), result)
    return result


__all__ = ["ScanExecutionError", "execute_scan", "_point_has_issue"]
