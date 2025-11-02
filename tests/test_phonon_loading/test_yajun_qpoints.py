#!/usr/bin/env python3
"""Test loading yajun data to check available q-points"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from phonproj.modes import PhononModes


def test_yajun_data():
    """Test loading yajun data"""
    print("=== Testing Yajun Dataset ===\n")

    try:
        # Try loading the yajun phonopy.yaml file
        modes = PhononModes.from_phonopy_yaml(
            "data/yajundata/0.02-P4mmm-PTO/phonopy.yaml",
            None,  # Load all available q-points
        )
        print(f"✅ Loaded {len(modes.qpoints)} q-points from yajun data")
        print(f"Q-points:")
        for i, q in enumerate(modes.qpoints):
            print(f"  {i:2d}: [{q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}]")

        # Test commensurate q-points for 2x2x2
        supercell_2x2x2 = np.eye(3) * 2
        step7_indices = modes.get_commensurate_qpoints(supercell_2x2x2)
        print(f"\nStep 7 commensurate q-points for 2x2x2 supercell:")
        print(f"Indices: {step7_indices}")
        for i in step7_indices:
            q = modes.qpoints[i]
            print(f"  {i:2d}: [{q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}]")

        return modes

    except Exception as e:
        print(f"❌ Failed to load yajun data: {e}")
        return None


if __name__ == "__main__":
    modes = test_yajun_data()
