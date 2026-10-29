#!/usr/bin/env python3
"""Verify the McStas 3.10.0 native fixture set without subprocesses."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value):
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode()
    if isinstance(value, np.bytes_):
        return bytes(value).rstrip(b"\x00").decode()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return b"".join(value.reshape(-1)).rstrip(b"\x00").decode()
        if value.size == 1:
            return decode(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def sim_metadata(path: Path):
    text = path.read_text()
    parameters = {
        key: value.strip().strip('"')
        for key, value in re.findall(r"^\s*Param:\s*(\w+)=(.*)$", text, re.M)
    }
    seed = re.search(r"^\s*Seed:\s*(-?\d+)", text, re.M)
    return parameters, int(seed.group(1)) if seed else None


def verify_file_hashes(case_dir: Path, case: dict) -> int:
    expected = {item["path"]: item for item in case["files"]}
    actual = {
        str(path.relative_to(case_dir)): path
        for path in case_dir.rglob("*")
        if path.is_file()
    }
    assert set(actual) == set(expected), (case_dir, sorted(actual), sorted(expected))
    for relative, item in expected.items():
        path = actual[relative]
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path
    return len(actual)


def verify_summary(case_dir: Path, summary: dict) -> int:
    path = case_dir / summary["path"]
    text = path.read_text()
    headers = {}
    for line in text.splitlines():
        if line.startswith("# ") and ":" in line:
            key, value = line[2:].split(":", 1)
            headers[key] = value.strip()
    expected_headers = summary["header"]
    for key, value in expected_headers.items():
        assert headers.get(key) == value, (path, key, headers.get(key), value)
    actual_rows = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert actual_rows == summary["literal_rows"], (path, actual_rows)
    return len(actual_rows)


def verify_ascii(case_dir: Path, case: dict) -> tuple[int, int]:
    file_count = verify_file_hashes(case_dir, case)
    row_count = verify_summary(case_dir, case["summary"])
    for point in case["points"]:
        point_dir = case_dir / point["path"]
        parameters, seed = sim_metadata(point_dir / "mccode.sim")
        assert parameters == point["parameters"], (point_dir, parameters)
        assert seed == point["seed"], (point_dir, seed)
        detector = point["detector"]
        if detector is None:
            assert not (case_dir / f"{point['path']}/spectrum.dat").exists()
            continue
        array = np.loadtxt(case_dir / detector)
        assert array.shape == tuple(point["detector_shape"]), detector
        assert np.isfinite(array).all(), detector
        assert float(array[:, 3].sum()) == point["event_count"], detector
    if "repeated_point_ids" in case:
        first, second = case["repeated_point_ids"]
        assert case["points"][first]["parameters"] == case["points"][second]["parameters"]
        assert case["points"][first]["seed"] != case["points"][second]["seed"]
    return file_count, row_count


def verify_nexus(case_dir: Path, case: dict) -> tuple[int, int]:
    file_count = verify_file_hashes(case_dir, case)
    row_count = verify_summary(case_dir, case["summary"])
    with h5py.File(case_dir / "mccode.h5", "r") as handle:
        assert decode(handle.attrs["HDF5_Version"]) == case["hdf5_root"]["HDF5_Version"]
        assert decode(handle.attrs["NeXus_version"]) == case["hdf5_root"]["NeXus_version"]
        entry_names = sorted(
            (key for key in handle if key.startswith("entry")),
            key=lambda key: int(key[5:]),
        )
        assert entry_names == [entry["entry"] for entry in case["entries"]]
        for entry_spec in case["entries"]:
            entry = handle[entry_spec["entry"]]
            params = {
                key: decode(value[()])
                for key, value in entry["simulation/Param"].items()
            }
            assert params == entry_spec["parameters"], (entry_spec, params)
            seed = decode(entry["simulation"].attrs["Seed"])
            assert str(seed) == entry_spec["seed"], (entry_spec, seed)
            detector = entry["data/spectrum_dat"]
            assert sorted(detector) == sorted(entry_spec["detector_shapes"])
            for name, shape in entry_spec["detector_shapes"].items():
                values = detector[name][()]
                assert list(values.shape) == shape, (entry_spec, name)
                assert np.isfinite(values).all(), (entry_spec, name)
            assert float(detector["ncount"][()].sum()) == entry_spec["event_count"]
    return file_count, row_count


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert sha256(HERE.parent / "tiny_scan.instr") == manifest["capture"]["instrument_sha256"]
    total_files = 0
    total_rows = 0
    total_points = 0
    for name, case in manifest["cases"].items():
        case_dir = HERE / name
        if case["format"] == "McCode":
            files, rows = verify_ascii(case_dir, case)
            total_points += len(case["points"])
        else:
            files, rows = verify_nexus(case_dir, case)
            total_points += len(case["entries"])
        total_files += files
        total_rows += rows
        print(f"Verified {name}: {files} files, {rows} summary rows")
    assert total_files == 29
    assert total_points == 15
    print(f"Verified {total_files} native files, {total_points} point/entry records, {total_rows} summary rows")


if __name__ == "__main__":
    main()
