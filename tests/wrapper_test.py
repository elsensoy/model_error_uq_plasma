import unittest

# Test import of hallthruster_jl_wrapper
try:
    from hallmd.models.thruster import hallthruster_jl_wrapper
    wrapper_imported = True
except ImportError:
    wrapper_imported = False

class TestHallThrusterWrapper(unittest.TestCase):

    def test_import(self):
        """
        Test to check if hallthruster_jl_wrapper is imported successfully.
        """
        self.assertTrue(wrapper_imported, "hallthruster_jl_wrapper should be imported successfully.")

# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()
