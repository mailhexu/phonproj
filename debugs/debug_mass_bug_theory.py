#!/usr/bin/env python3
"""
Minimal test to verify the mass consistency bug in decompose_displacement_to_modes.

The suspected issue:
1. Target displacement normalization uses target_supercell masses (line 358)
2. Mode displacement normalization uses source_supercell masses (line 434)
3. Projection uses target_supercell masses (line 251)

These should all use the SAME reference structure masses.
"""

import numpy as np
from phonproj.core.structure_analysis import decompose_displacement_to_modes


def test_mass_consistency_bug():
    """Test if the mass consistency bug is the root cause of sum=2.17."""

    print("Testing mass consistency bug theory...")

    # Run the step10 example and capture the projection result
    print("Running step10 example...")

    import subprocess
    import sys

    # Run step10 and capture stdout
    result = subprocess.run(
        [sys.executable, "examples/step10_yajun_analysis.py"],
        capture_output=True,
        text=True,
        cwd="/Users/hexu/projects/phonproj",
    )

    if result.returncode != 0:
        print(f"Error running step10: {result.stderr}")
        return

    # Look for the sum in the output
    lines = result.stdout.split("\n")
    sum_line = None
    for line in lines:
        if "Sum of squared projections" in line:
            sum_line = line
            break

    if sum_line:
        print(f"Found: {sum_line}")
        # Extract the sum value
        import re

        match = re.search(r"(\d+\.\d+)", sum_line)
        if match:
            sum_value = float(match.group(1))
            print(f"Extracted sum: {sum_value}")

            if abs(sum_value - 2.17) < 0.1:
                print("✗ Confirmed: Sum is ~2.17, indicating mass consistency bug")
                return True
            elif abs(sum_value - 1.0) < 0.1:
                print("✓ Good: Sum is ~1.0, mass consistency appears fixed")
                return False
            else:
                print(f"? Unexpected sum value: {sum_value}")
                return None

    print("Could not find sum in output")
    print("Full output:", result.stdout[-500:])  # Last 500 chars
    return None


if __name__ == "__main__":
    test_mass_consistency_bug()
