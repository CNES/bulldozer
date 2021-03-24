# coding: utf-8

import unittest
import os.path
from bulldozer.config_parser import ConfigParser

class TestConfigParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = ConfigParser(False)
        self.path = os.path.join(os.path.dirname(__file__), 'data/')

    def test_format_check(self):
        # should raise exception because it's not an YAML file
        path = "test.txt"
        self.assertRaises(ValueError, lambda: self.parser.read(path))
        # should raise exception for file not found but should pass the format check
        path = "test.yaml"
        self.assertRaises(FileNotFoundError, lambda: self.parser.read(path))
        # should raise exception for file not found but should pass the format check
        path = "test.yml"
        self.assertRaises(FileNotFoundError, lambda: self.parser.read(path))

    def test_existence_check(self):
        # should raise FileNotFoundException since the file doesn't exist
        path = "test.yml"
        self.assertRaises(FileNotFoundError, lambda: self.parser.read(path))
        
        # shouldnt raise FileNotFoundException, if it raises an exception the unit test framework will flag this as an error
        path = self.path + "test.yaml"
        self.parser.read(path)


    def test_read(self):
        path = self.path + "test.yaml"
        cfg = self.parser.read(path)
        # check type
        self.assertIsInstance(cfg, dict)

        # check dict size (expected 5)
        self.assertEqual(len(cfg), 5)

        # check string element read
        self.assertIsInstance(cfg['str_test'], str)
        self.assertIsInstance(cfg['str_test2'], str)
        self.assertEqual(cfg['str_test'], 'test')
        self.assertEqual(cfg['str_test2'], 'test2')

        # check integer element read
        self.assertIsInstance(cfg['int_test'], int)
        self.assertEqual(cfg['int_test'], 100)

        # check boolean element read
        self.assertIsInstance(cfg['boolean_test'], bool)
        self.assertEqual(cfg['boolean_test'], True)

        # check float sub-element read
        self.assertIsInstance(cfg['parent'], dict)
        self.assertIsInstance(cfg['parent']['child'], float)
        self.assertEqual(cfg['parent']['child'], 10.3)
        self.assertEqual(cfg['parent']['child2'], 13)        