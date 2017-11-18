import os
import unittest

def GetTestRootDir():
    git_root = os.path.abspath("../..") # this contains py_proto, testdir, ...
    return os.path.join(git_root, "testdata")

class SurikoTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.tests_data_dir = GetTestRootDir()

