"""
Unit tests for main.py anomaly detection CLI.
"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestMainCLI(unittest.TestCase):
    """Test cases for the main.py command line interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.main_script = self.project_root / "main.py"
    
    def test_run_all_methods_with_limit(self):
        """Test running all detection methods with limit 1 and save disabled."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.main_script),
                "--method", "all",
                "--limit", "1",
                "--save", "0"
            ],
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Print output for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # Check that the command completed successfully
        self.assertEqual(
            result.returncode, 
            0, 
            f"Command failed with return code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
