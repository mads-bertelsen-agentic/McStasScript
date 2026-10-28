#!/usr/bin/env python3
"""Capture native fixtures, not a McStasScript execution API.

Requires mcrun, mcstas, numpy and h5py in the active environment. All native
compilation happens in a fresh directory under --work-root. Native data files
are copied byte-for-byte; only copied instrument/C sources are discarded.
"""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
WORK_ROOT = Path(tempfile.gettempdir())
SEED = 314159
NCOUNT = 64
DEFAULTS = {"theta": 0, "wavelength": 3, "category": "vanadium", "fail": 0}


def case(name, options, rows, *, parallel=False, failure=False, nexus=False):
    return {
        "name": name,
        "options": options,
        "parallel": parallel,
        "intentional_failure": failure,
        "nexus": nexus,
        "scanned_parameters": [arg.split("=", 1)[0] for arg in options
                               if "=" in arg and not arg.startswith("--")],
        "intended_points": [
            {"point_id": i, "parameters": {**DEFAULTS, **row},
             "seed": SEED + (1024 * i if parallel else 0),
             "expected_status": ("completed" if not failure or i == 0 else
                                 "failed" if i == 1 else "not_attempted")}
            for i, row in enumerate(rows)
        ],
    }


def cases():
    repeated = [
        {"theta": 0, "wavelength": 3, "category": "vanadium"},
        {"theta": 1, "wavelength": 5, "category": "nickel"},
        {"theta": 0, "wavelength": 3, "category": "vanadium"},
    ]
    repeated_options = ["-L", "theta=0,1,0", "wavelength=3,5,3",
                        "category=vanadium,nickel,vanadium"]
    partial = [{"theta": i, "fail": int(i == 1)} for i in range(4)]
    partial_options = ["-L", "theta=0,1,2,3", "fail=0,1,0,0"]
    specs = [
        case("ascii_1d", ["-N", "3", "theta=-1,1"],
             [{"theta": i} for i in (-1, 0, 1)]),
        case("ascii_cartesian_2x3",
             ["-L", "theta=-1,1,-1,1,-1,1", "wavelength=2,2,4,4,6,6"],
             [{"theta": t, "wavelength": w} for w in (2, 4, 6) for t in (-1, 1)]),
        case("ascii_parallel", ["--scan_split=2", "-M", "-N", "2",
                                 "theta=-1,1", "wavelength=3,5"],
             [{"theta": t, "wavelength": w} for w in (3, 5) for t in (-1, 1)],
             parallel=True),
        case("ascii_categorical_repeated", repeated_options, repeated),
        case("ascii_partial", partial_options, partial, failure=True),
        case("ascii_paired", ["-N", "3", "theta=-1,1", "wavelength=2,6"],
             [{"theta": t, "wavelength": w} for t, w in [(-1, 2), (0, 4), (1, 6)]]),
        case("nexus_paired_categorical_repeated", repeated_options, repeated, nexus=True),
        case("nexus_parallel", ["--scan_split=2", "-N", "2", "theta=-1,1"],
             [{"theta": t} for t in (-1, 1)], parallel=True, nexus=True),
        case("nexus_partial", partial_options, partial, failure=True, nexus=True),
    ]
    mirrors = {"ascii_1d", "ascii_cartesian_2x3", "ascii_paired"}
    for ascii_spec in [spec for spec in specs if spec["name"] in mirrors]:
        specs.append(case(ascii_spec["name"].replace("ascii_", "nexus_", 1),
                          ascii_spec["options"],
                          [point["parameters"] for point in ascii_spec["intended_points"]],
                          nexus=True))
    return specs


def run(argv, work):
    print(shlex.join(argv), flush=True)
    return subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=180)


def native_arguments(spec):
    return ["mcrun", "--force-compile", "--format=" + ("NeXus" if spec["nexus"] else "McCode"),
            "--seed=" + str(SEED), "--ncount=" + str(NCOUNT),
            "--dir=" + spec["name"], "tiny_scan.instr", *spec["options"]]


def text_value(dataset):
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return b"".join(value.reshape(-1)).decode().rstrip("\x00")
    if isinstance(value, bytes):
        return value.decode().rstrip("\x00")
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def observe_ascii(directory):
    # Parsing here verifies the generator independently of any package loader.
    points, incomplete = [], []
    for point in sorted(directory.glob("[0-9]*"), key=lambda p: int(p.name)):
        if not (point / "mccode.sim").is_file():
            continue
        metadata = (point / "mccode.sim").read_text()
        parameters = {}
        for key, value in re.findall(r"^\s*Param:\s*(\w+)=(.*)$", metadata, re.M):
            parameters[key] = value.strip().strip('"')
        seed = re.search(r"^\s*Seed:\s*(\d+)", metadata, re.M)
        native = {"point_id": int(point.name), "path": point.name,
                  "parameters": parameters, "seed": int(seed[1]) if seed else None}
        if not (point / "spectrum.dat").exists():
            incomplete.append(native)
            continue
        array = np.loadtxt(point / "spectrum.dat")
        assert array.shape == (4, 4), array.shape
        assert np.isfinite(array).all()
        points.append({**native, "detector_shape": list(array.shape),
                       "event_count": float(array[:, 3].sum())})
    return points, incomplete


def observe_nexus(directory):
    points, incomplete = [], []
    for filename in sorted(directory.rglob("*.h5")):
        with h5py.File(filename) as handle:
            entries = sorted((key for key in handle if key.startswith("entry")),
                             key=lambda key: int(key[len("entry"):]))
            for entry in entries:
                group = handle[entry]
                parameters = {key: text_value(value)
                              for key, value in group["simulation/Param"].items()}
                point_id = (int(filename.parent.name) if filename.parent.name.isdecimal()
                            else int(entry[len("entry"):]) - 1)
                native = {"point_id": point_id,
                          "path": str(filename.relative_to(directory)),
                          "entry": entry, "parameters": parameters,
                          "seed": int(group["simulation"].attrs["Seed"])}
                if "data/spectrum_dat" not in group:
                    incomplete.append(native)
                    continue
                detector = group["data/spectrum_dat"]
                arrays = {key: list(value.shape) for key, value in detector.items()
                          if isinstance(value, h5py.Dataset)}
                for key in ("data", "errors", "ncount", "Wavelength__AA_"):
                    assert np.isfinite(detector[key][()]).all()
                points.append({**native, "detector_datasets": arrays,
                               "event_count": float(detector["ncount"][()].sum())})
    return points, incomplete


def observations(directory, spec):
    points, incomplete = observe_nexus(directory) if spec["nexus"] else observe_ascii(directory)
    summary = directory / "mccode.dat"
    summary_text = summary.read_text() if summary.exists() else ""
    rows = [line for line in summary_text.splitlines() if line.strip() and not line.startswith("#")]
    header = next((line[len("# variables: "):] for line in summary_text.splitlines()
                   if line.startswith("# variables: ")), None)
    completed = {point["point_id"] for point in points}
    return {"points": points, "incomplete_points": incomplete,
            "summary_rows": rows, "summary_variables": header,
            "summary_bytes": summary.stat().st_size if summary.exists() else None,
            "has_root_sim": (directory / "mccode.sim").exists(),
            "missing_point_ids": [p["point_id"] for p in spec["intended_points"]
                                  if p["point_id"] not in completed]}


def verify_case(spec, observed):
    expected = spec["intended_points"]
    if spec["intentional_failure"]:
        expected = expected[:1]
        assert [p["point_id"] for p in observed["incomplete_points"]] == [1]
    else:
        assert not observed["incomplete_points"]
    assert len(observed["points"]) == len(expected), (spec["name"], observed)
    for native, intended in zip(observed["points"], expected):
        assert native["point_id"] == intended["point_id"]
        assert native["seed"] == intended["seed"], (spec["name"], native)
        for key, value in intended["parameters"].items():
            actual = native["parameters"][key]
            assert (actual == value if isinstance(value, str) else float(actual) == value), (
                spec["name"], key, actual, value)
        if spec["nexus"]:
            assert native["detector_datasets"]["data"] == [4]
            assert native["detector_datasets"]["errors"] == [4]
            assert native["detector_datasets"]["ncount"] == [4]
        assert native["event_count"] == NCOUNT
    for native in observed["incomplete_points"]:
        intended = spec["intended_points"][native["point_id"]]
        assert native["seed"] == intended["seed"]
        for key, value in intended["parameters"].items():
            actual = native["parameters"][key]
            assert (actual == value if isinstance(value, str) else float(actual) == value)
    assert len(observed["summary_rows"]) == (0 if spec["nexus"] else len(expected))
    assert observed["has_root_sim"] == (not spec["nexus"])
    if spec["nexus"]:
        assert observed["summary_bytes"] == 0
    for row, intended in zip(observed["summary_rows"], expected):
        numeric = [float(value) for value in row.split()]
        coordinates = [intended["parameters"][key] for key in spec["scanned_parameters"]]
        if any(isinstance(value, str) for value in coordinates):
            coordinates = [intended["point_id"]]
        assert numeric[:-2] == coordinates, (spec["name"], row, coordinates)


def verify_arrays():
    for ascii_name, nexus_name in (
        ("ascii_1d", "nexus_1d"),
        ("ascii_cartesian_2x3", "nexus_cartesian_2x3"),
        ("ascii_paired", "nexus_paired"),
        ("ascii_categorical_repeated", "nexus_paired_categorical_repeated"),
        ("ascii_partial", "nexus_partial"),
    ):
        with h5py.File(HERE / nexus_name / "mccode.h5") as handle:
            for point in sorted((HERE / ascii_name).glob("[0-9]*")):
                if not (point / "spectrum.dat").exists():
                    continue
                native_ascii = np.loadtxt(point / "spectrum.dat")
                native_h5 = handle[f"entry{int(point.name) + 1}/data/spectrum_dat"]
                for column, key in enumerate(("Wavelength__AA_", "data", "errors", "ncount")):
                    np.testing.assert_allclose(native_ascii[:, column], native_h5[key][()], rtol=1e-5, atol=0)
    with h5py.File(HERE / "nexus_paired_categorical_repeated/mccode.h5") as handle:
        for key in ("data", "errors", "ncount"):
            np.testing.assert_array_equal(handle[f"entry1/data/spectrum_dat/{key}"][()],
                                          handle[f"entry3/data/spectrum_dat/{key}"][()])
    print("Verified ASCII/NeXus array agreement and repeated-coordinate identity")


def verify():
    manifest = json.loads((HERE / "expected.json").read_text())
    assert manifest["instrument_sha256"] == hashlib.sha256((HERE / "tiny_scan.instr").read_bytes()).hexdigest()
    for spec in manifest["cases"]:
        observed = observations(HERE / spec["name"], spec)
        verify_case(spec, observed)
        assert observed == spec["observed"], spec["name"]
        assert {str(path.relative_to(HERE / spec["name"]))
                for path in (HERE / spec["name"]).rglob("*") if path.is_file()} == set(spec["sha256"])
        for relative, digest in spec["sha256"].items():
            assert hashlib.sha256((HERE / spec["name"] / relative).read_bytes()).hexdigest() == digest
        print("Verified", spec["name"])
    verify_arrays()


def capture(work_root, replace=False):
    specs = cases()
    if not replace and any((HERE / spec["name"]).exists() for spec in specs):
        raise SystemExit("Fixture directories already exist; remove only these captures before regenerating.")
    work = Path(tempfile.mkdtemp(prefix="scan-fixtures-", dir=work_root))
    print("Native build directory:", work, flush=True)
    shutil.copyfile(HERE / "tiny_scan.instr", work / "tiny_scan.instr")
    manifest = {
        "schema_version": 1,
        "purpose": "Native fixtures only; no McStasScript execution API is used.",
        "seed": SEED, "ncount_per_point": NCOUNT,
        "versions": {
            "mcrun": run(["mcrun", "--version"], work).stdout.strip(),
            "mcstas": run(["mcstas", "--version"], work).stdout.strip(),
            "python": sys.version, "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version, "numpy": np.__version__,
        },
        "build_directory": str(work), "cases": specs,
        "instrument_sha256": hashlib.sha256((HERE / "tiny_scan.instr").read_bytes()).hexdigest(),
    }
    for spec in specs:
        argv = native_arguments(spec)
        result = run(argv, work)
        (work / (spec["name"] + ".log")).write_text(result.stdout + result.stderr)
        spec["command"] = argv
        spec["returncode"] = result.returncode
        if spec["intentional_failure"]:
            assert "SCAN_FIXTURE_INTENTIONAL_FAILURE" in result.stderr, result.stderr
            assert result.returncode != 0
            spec["failure"] = {"native_exit_status": 23,
                               "message": "SCAN_FIXTURE_INTENTIONAL_FAILURE: fail requested"}
            assert "non-zero exit status 23" in result.stderr
        elif result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        native = work / spec["name"]
        spec["observed"] = observations(native, spec)
        verify_case(spec, spec["observed"])
        # Preserve every output except the automatic source-code copies.
        if replace and (HERE / spec["name"]).exists():
            shutil.rmtree(HERE / spec["name"])
        shutil.copytree(native, HERE / spec["name"],
                        ignore=shutil.ignore_patterns("*.c", "*.instr", "*.out", "*.dSYM"))
        spec["sha256"] = {
            str(path.relative_to(native)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(native.rglob("*")) if path.is_file()
            and path.suffix not in (".c", ".instr", ".out")
        }
        spec["artifact_bytes"] = sum((HERE / spec["name"] / p).stat().st_size for p in spec["sha256"])
    (HERE / "expected.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_arrays()
    print("Native data bytes:", sum(spec["artifact_bytes"] for spec in specs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check committed fixtures without running McStas")
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT,
                        help="existing temporary parent (default: tempfile.gettempdir())")
    parser.add_argument("--replace", action="store_true", help="replace only the named generated fixture directories")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        capture(args.work_root, replace=args.replace)
