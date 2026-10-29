import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from mcstasscript.data.scan_manifest import ScanManifest
from mcstasscript.data.scan import ScanSource
from mcstasscript.data.scan_plan import ScanPlan


def make_plan(count=4):
    if count == 4:
        return ScanPlan.grid(angle=[0, 1], wavelength=[10, 20])
    side = int(count ** 0.5)
    assert side * side == count
    return ScanPlan.grid(angle=range(side), wavelength=range(side))


def report(plan, output_path):
    return SimpleNamespace(
        plan=plan,
        fixed_parameters={"source": "sample"},
        parameter_snapshot={"source": "sample", "angle": "scanned"},
        effective_settings={"ncount": 1000},
        tool={"name": "mcstas", "version": "3.7"},
        output_path=output_path,
        errors=[],
        warnings=[],
        valid=True,
    )


class TestScanManifest(unittest.TestCase):
    def make_manifest(self, count=4, output_path="/tmp/scan"):
        return ScanManifest.from_preflight(report(make_plan(count), output_path))

    def test_huge_plan_stays_compact(self):
        manifest = self.make_manifest(1_000_000)
        document = manifest.to_dict()
        self.assertEqual(manifest.plan.point_count, 1_000_000)
        self.assertEqual(len(document["plan"]["axes"]["angle"]), 1000)
        self.assertEqual(document["points"], {})
        self.assertEqual(document["execution"]["recorded"], 0)

        manifest.record_point(999_999, status="succeeded", data_status="unknown")
        self.assertEqual(len(manifest.to_dict()["points"]), 1)

    def test_json_roundtrip_and_portable_nexus_locator(self):
        manifest = self.make_manifest(output_path="/original/results")
        source = ScanSource("/original/results/point-0.h5", "nexus", "/entry_7")
        manifest.record_point(0, source=source, status="failed",
                              data_status="available")
        encoded = json.dumps(manifest.to_dict(), allow_nan=False)
        restored = ScanManifest.from_dict(json.loads(encoded))
        relocated = restored.resolve_source(0, "/moved/results")

        self.assertEqual(relocated.path, Path("/moved/results/point-0.h5"))
        self.assertEqual(relocated.format, "nexus")
        self.assertEqual(relocated.entry, "/entry_7")
        self.assertEqual(restored.records[0]["status"], "failed")
        self.assertEqual(restored.records[0]["data_status"], "available")

    def test_external_source_remains_explicit(self):
        manifest = self.make_manifest(output_path="/results")
        manifest.record_point(0, source=ScanSource("/outside/point"))
        value = manifest.to_dict()["points"]["0"]["source"]
        self.assertTrue(value["external"])
        self.assertEqual(manifest.resolve_source(0).path, Path("/outside/point"))

    def test_seed_absence_and_actual_intended_values(self):
        manifest = self.make_manifest()
        manifest.record_point(0)
        manifest.record_point(1, seed={"actual": 23, "intended": 17})
        self.assertIsNone(manifest.records[0]["seed"])
        self.assertEqual(manifest.records[1]["seed"], {"actual": 23, "intended": 17})

    def test_invalid_schema_version_path_status_and_id(self):
        manifest = self.make_manifest()
        document = manifest.to_dict()
        for field, value in (("schema", "other"), ("version", 99), ("version", 1.0)):
            invalid = dict(document)
            invalid[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    ScanManifest.from_dict(invalid)

        with self.assertRaises(ValueError):
            manifest.record_point(99)
        with self.assertRaises(TypeError):
            manifest.record_point("0")
        with self.assertRaises(ValueError):
            manifest.record_point(0, status="available")
        manifest.record_point(0)
        invalid = manifest.to_dict()
        invalid["points"]["0"]["source"] = {
            "path": "../escape", "format": "ascii", "entry": None, "external": False}
        with self.assertRaises(ValueError):
            ScanManifest.from_dict(invalid)

    def test_preflight_rejects_invalid_report_and_accepts_namespace(self):
        invalid = report(make_plan(1), "/results")
        invalid.valid = False
        with self.assertRaises(ValueError):
            ScanManifest.from_preflight(invalid)
        manifest = ScanManifest.from_preflight(
            SimpleNamespace(**report(ScanPlan.grid(angle=[1]), "/results").__dict__))
        self.assertEqual(manifest.plan.point_count, 1)

    def test_process_evidence_is_independent_from_point_data(self):
        manifest = self.make_manifest()
        manifest.record_point(0, status="failed", data_status="available")
        manifest.record_execution(status="failed", outcome="nonzero_exit",
                                  returncode=17, log="run.log",
                                  diagnostics=["process failed"])
        execution = manifest.to_dict()["execution"]
        self.assertEqual(execution["process"]["returncode"], 17)
        self.assertEqual(execution["process"]["outcome"], "nonzero_exit")
        self.assertEqual(execution["counts"]["failed"], 1)
        self.assertEqual(manifest.records[0]["data_status"], "available")

    def test_seed_values_are_positive_integers(self):
        manifest = self.make_manifest()
        for seed in (0, -1, 1.0, "7", {"actual": 4, "intended": 0}):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                manifest.record_point(0, seed=seed)

    def test_fixed_parameters_cannot_contain_scanned_names(self):
        with self.assertRaises(ValueError):
            ScanManifest.from_preflight(SimpleNamespace(
                plan=make_plan(), fixed_parameters={"angle": 1},
                parameter_snapshot={"angle": 2}, effective_settings={}, tool={},
                output_path="/results", errors=[], warnings=[], valid=True))

    def test_metadata_properties_are_detached(self):
        manifest = self.make_manifest()
        fixed = manifest.fixed_parameters
        fixed["changed"] = object()
        snapshot = manifest.parameter_snapshot
        snapshot["changed"] = object()
        self.assertNotIn("changed", manifest.to_dict()["instrument"]["fixed_parameters"])
        self.assertNotIn("changed", manifest.to_dict()["instrument"]["parameter_snapshot"])

    def test_execution_aggregate_requires_exact_integer_counts(self):
        document = self.make_manifest().to_dict()
        document["execution"]["recorded"] = 0.0
        with self.assertRaises(ValueError):
            ScanManifest.from_dict(document)
        document = self.make_manifest().to_dict()
        document["execution"]["counts"]["unknown"] = True
        with self.assertRaises(ValueError):
            ScanManifest.from_dict(document)
        document = self.make_manifest().to_dict()
        del document["execution"]["counts"]["pending"]
        with self.assertRaises(ValueError):
            ScanManifest.from_dict(document)


if __name__ == "__main__":
    unittest.main()
