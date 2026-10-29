import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mcstasscript import load_data, load_scan
from mcstasscript.data.scan import ScanMonitor, ScanMonitorSummary, ScanPoint, ScanSource
from mcstasscript.data.scan_manifest import MANIFEST_FILENAME, ScanManifest
from mcstasscript.data.scan_persistence import (
    finalize_manifest, load_manifest_scan, reconcile_scan, write_manifest,
)
from mcstasscript.data.scan_plan import ScanPlan
from mcstasscript.helper.scan_command import plan_point_id


FIXTURES = Path(__file__).resolve().parent / "scan_fixtures"


def manifest(root, plan):
    return ScanManifest(
        plan=plan,
        effective_settings={"seed": 314159},
        tool={"capabilities": {"seed_policy": "point_offset_1024"}},
        output_root=root,
    )


class TestScanPersistence(unittest.TestCase):
    def test_single_run_loader_redirects_manifest_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(manifest(root, ScanPlan.points([{"theta": 0}])), root)
            with self.assertRaisesRegex(RuntimeError, "load_scan"):
                load_data(root)

    def test_sparse_manifest_keeps_plan_and_missing_points(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": 0}, {"theta": 1}])
            current = manifest(root, plan)
            write_manifest(current, root)

            scan = load_scan(root)
            self.assertEqual([point.point_id for point in scan.points], [0, 1])
            self.assertEqual([point.status for point in scan.points], ["unknown"] * 2)
            self.assertEqual([point.data_status for point in scan.points], ["missing"] * 2)
            self.assertEqual(scan.plan, plan)

    def test_singleton_ascii_is_lazy_and_survives_a_move(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as relocated:
            root = Path(directory)
            (root / "native").mkdir()
            for source in (FIXTURES / "ascii_1d" / "0").iterdir():
                destination = root / "native" / source.name
                if source.is_file():
                    shutil.copyfile(source, destination)
            plan = ScanPlan.points([{"theta": -1}])
            current = manifest(root, plan)
            write_manifest(current, root)

            scan = load_manifest_scan(root / MANIFEST_FILENAME)
            self.assertFalse(scan.points[0].is_loaded)
            self.assertEqual(scan.points[0].parameters, {"theta": -1})
            self.assertEqual(scan.points[0].metadata["seed"]["actual"], 314159)

            moved = Path(relocated) / "moved"
            shutil.copytree(root, moved)
            moved_scan = load_manifest_scan(moved)
            self.assertEqual(moved_scan.points[0].source.path,
                             moved / "native")

    def test_malformed_manifest_does_not_fall_back_to_native(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / MANIFEST_FILENAME).write_text(json.dumps({}), encoding="utf-8")
            shutil.copytree(FIXTURES / "ascii_1d", root / "native")
            with self.assertRaises(ValueError):
                load_scan(root)

    def test_atomic_failure_leaves_no_temporary_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current = manifest(root, ScanPlan.points([{"theta": 0}]))
            with patch("mcstasscript.data.scan_persistence.os.replace",
                       side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    write_manifest(current, root)
            self.assertFalse((root / MANIFEST_FILENAME).exists())
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_finalize_records_data_status_without_inventing_success(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(FIXTURES / "ascii_1d" / "0", root / "native")
            current = manifest(root, ScanPlan.points([{"theta": -1}]))
            updated, scan = finalize_manifest(current, root)
            self.assertEqual(updated.records[0]["status"], "unknown")
            self.assertEqual(updated.records[0]["data_status"], "available")
            self.assertEqual(scan.execution, current.execution)

    def test_finalize_keeps_sparse_manifest_when_native_output_is_absent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current = manifest(root, ScanPlan.points([{"theta": 0}, {"theta": 1}]))

            updated, scan = finalize_manifest(current, root)

            self.assertEqual(updated.records, {})
            self.assertEqual(len(scan.points), 2)
            self.assertEqual(updated.execution["recorded"], 0)
            self.assertEqual(updated.execution["counts"]["unknown"], 2)

    def test_finalize_preserves_existing_record_without_native_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current = manifest(root, ScanPlan.points([{"theta": 0}, {"theta": 1}]))
            current.record_point(1, status="failed", seed={"actual": 315183,
                                "intended": 315183}, diagnostics=["native failure"])

            updated, scan = finalize_manifest(current, root)

            self.assertEqual(len(scan.points), 2)
            self.assertEqual(updated.records[1]["status"], "failed")
            self.assertEqual(updated.records[1]["seed"],
                             {"actual": 315183, "intended": 315183})
            self.assertEqual(updated.records[1]["diagnostics"], ["native failure"])
            self.assertEqual(updated.execution["recorded"], 1)
            self.assertEqual(updated.execution["counts"]["failed"], 1)
            self.assertEqual(updated.execution["counts"]["unknown"], 1)

    def test_unassociated_observations_retain_detached_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            source = ScanSource(native / "mccode.h5", "nexus", "/entry7")
            monitor = ScanMonitor("spectrum", "spectrum.dat")
            summary = ScanMonitorSummary(
                monitor, total_I=4.5, availability={"total_I": True},
                provenance={"total_I": "native:values"},
                metadata={"raw": "summary"})
            observation = ScanPoint(
                3, {"theta": 9}, source, summaries=[summary],
                metadata={"seed": 123, "raw": "point"},
                loader=lambda unused: (_ for _ in ()).throw(AssertionError("loaded")))
            current = manifest(root, ScanPlan.points([{"theta": 0}]))
            with patch("mcstasscript.data.scan_persistence._native_observations",
                       return_value=([observation], {"format": "nexus"}, "available")):
                scan = reconcile_scan(current, root)

            detached = scan.metadata["native"]["unassociated_observations"]
            self.assertEqual(len(detached), 1)
            self.assertEqual(detached[0]["native_point_id"], 3)
            self.assertEqual(detached[0]["source"], {
                "path": str(source.path), "format": "nexus", "entry": "/entry7"})
            self.assertEqual(detached[0]["parameters"], {"theta": 9})
            self.assertEqual(detached[0]["summaries"][0]["total_I"], 4.5)
            self.assertEqual(detached[0]["summaries"][0]["availability"]["total_I"], True)
            self.assertEqual(detached[0]["summaries"][0]["provenance"]["total_I"],
                             "native:values")
            self.assertEqual(detached[0]["summaries"][0]["metadata"], {"raw": "summary"})
            self.assertEqual(detached[0]["metadata"]["raw"], "point")

    def test_seed_identity_keeps_native_and_storage_ids_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.grid(theta=[0, 1], wavelength=[10, 20])
            current = manifest(root, plan)
            observations = []
            for storage_id, native_id in enumerate((0, 2, 3)):
                point_id = plan_point_id(plan, native_id)
                observations.append(ScanPoint(
                    storage_id,
                    plan[point_id].parameters,
                    ScanSource(root / "native" / "mccode.h5", "nexus",
                               "/entry{}".format(storage_id + 1)),
                    metadata={"seed": 314159 + 1024 * native_id}))

            with patch("mcstasscript.data.scan_persistence._native_observations",
                       return_value=(observations, {"format": "nexus"}, "available")):
                scan = reconcile_scan(current, root)

            by_id = {point.point_id: point for point in scan.points}
            for storage_id, native_id in enumerate((0, 2, 3)):
                point_id = plan_point_id(plan, native_id)
                point = by_id[point_id]
                self.assertEqual(point.metadata["native_point_id"], native_id)
                self.assertEqual(point.metadata["storage_point_id"], storage_id)
                self.assertEqual(point.metadata["storage_entry"],
                                 "/entry{}".format(storage_id + 1))

    def test_missing_points_keep_intended_seed_and_fixed_parameters_are_verified(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = ScanPlan.points([{"theta": -1}, {"theta": 1}])
            current = ScanManifest(
                plan=plan, fixed_parameters={"wavelength": 99},
                effective_settings={"seed": 314159},
                tool={"capabilities": {"seed_policy": "point_offset_1024"}},
                output_root=root,
            )
            write_manifest(current, root)
            (root / "native").mkdir()
            shutil.copyfile(FIXTURES / "ascii_1d" / "mccode.sim",
                            root / "native" / "mccode.sim")
            shutil.copyfile(FIXTURES / "ascii_1d" / "mccode.dat",
                            root / "native" / "mccode.dat")

            scan = load_manifest_scan(root)
            self.assertEqual([point.data_status for point in scan.points],
                             ["missing", "missing"])
            self.assertEqual([point.metadata["intended_seed"] for point in scan.points],
                             [314159, 315183])

    def test_conflicting_record_evidence_does_not_reassociate_native_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(FIXTURES / "ascii_1d" / "0", root / "native")
            current = manifest(root, ScanPlan.points([{"theta": -1}]))
            current.record_point(
                0, source=ScanSource(root / "native"),
                seed={"actual": 999, "intended": 314159},
            )
            write_manifest(current, root)

            scan = load_manifest_scan(root)
            self.assertEqual(scan.points[0].data_status, "missing")
            self.assertTrue(any("actual seed conflicts" in message
                                for message in scan.metadata["diagnostics"]))
            self.assertTrue(any("actual seed conflicts" in message
                                for message in scan.metadata["association_issues"]))

    def test_external_source_inside_output_root_stays_external_after_relocation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current = manifest(root, ScanPlan.points([{"theta": 0}]))
            current.record_point(0, source=ScanSource(root / "point"))
            document = current.to_dict()
            document["points"]["0"]["source"] = {
                "path": str(root / "point"), "format": "ascii",
                "entry": None, "external": True,
            }

            restored = ScanManifest.from_dict(document)
            relocated = restored.resolve_source(0, root / "moved")
            self.assertEqual(relocated.path, root / "point")
            self.assertTrue(restored.to_dict()["points"]["0"]["source"]["external"])

    def test_corrupt_nexus_keeps_bytes_and_uses_unique_ascii_summaries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            shutil.copyfile(FIXTURES / "ascii_1d" / "mccode.sim",
                            native / "mccode.sim")
            shutil.copyfile(FIXTURES / "ascii_1d" / "mccode.dat",
                            native / "mccode.dat")
            corrupt = native / "mccode.h5"
            raw = b"not an hdf5 file"
            corrupt.write_bytes(raw)
            plan = ScanPlan.points([{"theta": -1}, {"theta": 0}, {"theta": 1}])
            current = ScanManifest(
                plan=plan,
                fixed_parameters={"wavelength": 3, "category": "vanadium", "fail": 0},
                effective_settings={"seed": 314159},
                tool={"capabilities": {"seed_policy": "point_offset_1024"}},
                output_root=root,
            )
            current.record_point(0, source=ScanSource(native),
                                 seed={"actual": None, "intended": 314159})
            write_manifest(current, root)

            scan = load_manifest_scan(root)
            self.assertEqual(corrupt.read_bytes(), raw)
            self.assertEqual([point.source for point in scan.points], [None] * 3)
            self.assertEqual([point.data_status for point in scan.points], ["missing"] * 3)
            self.assertIn("Combined NeXus output invalid", " ".join(scan.metadata["diagnostics"]))
            self.assertEqual(scan.metadata["native_data_status"], "unreadable")
            self.assertEqual(scan.dataset("spectrum").total_I.size, 3)

    def test_corrupt_nexus_ambiguous_ascii_rows_are_not_guessed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            (native / "mccode.sim").write_text(
                "# type: multiarray_1d(1)\n# filename: mccode.dat\n",
                encoding="utf-8")
            (native / "mccode.dat").write_text(
                "# Numpoints: 2\n# xvars: x\n"
                "# yvars: (detector_I,detector_ERR)\n"
                "# variables: x detector_I detector_ERR\n"
                "0 1 0.1\n0 2 0.2\n",
                encoding="utf-8")
            corrupt = native / "mccode.h5"
            raw = b"broken"
            corrupt.write_bytes(raw)
            plan = ScanPlan.points([{"x": 0}, {"x": 0}])
            write_manifest(ScanManifest(
                plan=plan, fixed_parameters={"wavelength": 3},
                effective_settings={"seed": 314159},
                tool={"capabilities": {"seed_policy": "point_offset_1024"}},
                output_root=root,
            ), root)

            scan = load_manifest_scan(root)
            self.assertEqual(corrupt.read_bytes(), raw)
            self.assertEqual([point.data_status for point in scan.points],
                             ["unreadable", "unreadable"])
            self.assertTrue(any("Ambiguous native association" in message
                                for message in scan.metadata["association_issues"]))
            self.assertEqual(len(scan.metadata["native"]["unassociated_summary_rows"]), 2)


if __name__ == "__main__":
    unittest.main()
