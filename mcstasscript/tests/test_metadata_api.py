import os
import unittest

from mcstasscript.interface.instr import McStas_instr
from mcstasscript.helper.mcstas_objects import MetadataBlock
from mcstasscript.tests.helpers_for_tests import WorkInTestDir


def _make_instr():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    dummy_path = os.path.join(THIS_DIR, "dummy_mcstas")
    with WorkInTestDir():
        instr = McStas_instr("TestInstr",
                             package_path=dummy_path,
                             checks=False)
    instr.add_component("Origin", "test_for_reading")
    instr.add_component("Sample", "test_for_reading")
    return instr


class TestInstrMetadataAPI(unittest.TestCase):

    def setUp(self):
        self.instr = _make_instr()
        origin = self.instr.get_component("Origin")
        origin.add_METADATA("stored", "txt", "some text")
        origin.add_METADATA("info", "JSON", '{"a": 1}')
        sample = self.instr.get_component("Sample")
        sample.add_METADATA("note", "txt", "hello")

    def test_list_METADATA(self):
        result = self.instr.list_METADATA()
        self.assertIn("Origin", result)
        self.assertIn("Sample", result)
        self.assertEqual(result["Origin"], ["stored", "info"])
        self.assertEqual(result["Sample"], ["note"])

    def test_list_METADATA_empty(self):
        instr = _make_instr()
        self.assertEqual(instr.list_METADATA(), {})

    def test_get_METADATA_names_only(self):
        names = self.instr.get_METADATA("Origin")
        self.assertEqual(names, ["stored", "info"])

    def test_get_METADATA_specific_block(self):
        block = self.instr.get_METADATA("Origin", "stored")
        self.assertIsInstance(block, MetadataBlock)
        self.assertEqual(block.type, "txt")
        self.assertEqual(block.value, "some text")

    def test_get_METADATA_specific_block_json(self):
        block = self.instr.get_METADATA("Origin", "info")
        self.assertEqual(block.type, "JSON")
        self.assertEqual(block.value, '{"a": 1}')

    def test_get_METADATA_component_not_found(self):
        with self.assertRaises(NameError):
            self.instr.get_METADATA("Nonexistent")

    def test_get_METADATA_name_not_found(self):
        with self.assertRaises(KeyError):
            self.instr.get_METADATA("Origin", "nonexistent")

    def test_metadata_type(self):
        self.assertEqual(self.instr.metadata_type("Origin", "stored"), "txt")
        self.assertEqual(self.instr.metadata_type("Sample", "note"), "txt")

    def test_metadata_data(self):
        self.assertEqual(self.instr.metadata_data("Origin", "stored"),
                         "some text")
        self.assertEqual(self.instr.metadata_data("Sample", "note"),
                         "hello")

    def test_metadata_type_not_found(self):
        with self.assertRaises(KeyError):
            self.instr.metadata_type("Origin", "nonexistent")

    def test_metadata_data_not_found(self):
        with self.assertRaises(NameError):
            self.instr.metadata_data("Nonexistent", "stored")


if __name__ == "__main__":
    unittest.main()
