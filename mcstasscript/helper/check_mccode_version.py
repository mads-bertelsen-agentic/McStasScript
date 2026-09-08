import os
import subprocess

def _parse_version(output):
    output = output.decode("utf-8")
    output = output.split(" (", 1)[0]
    output = output.split("version", 1)[1].strip()
    parts = output.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return major, minor, patch

def check_mcstas_major_version(mcstas_bin_path):
    """
    Checks installed McStas version and returns
    (major, minor, patch) tuple.
    """
    mcstas_command = os.path.join(mcstas_bin_path, "mcstas")
    output = subprocess.check_output([mcstas_command, "-v"])
    return _parse_version(output)

def check_mcxtrace_major_version(mcstas_bin_path):
    """
    Checks installed McXtrace version and returns
    (major, minor, patch) tuple.
    """
    mcstas_command = os.path.join(mcstas_bin_path, "mcxtrace")
    output = subprocess.check_output([mcstas_command, "-v"])
    return _parse_version(output)
