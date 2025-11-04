#!/usr/bin/env python3
"""
Test the fixed force_near_0 function directly.
"""

import numpy as np
from ase.io import read
from phonproj.core.structure_analysis import force_near_0

# Load the target structure
target_structure = read("data/yajundata/disp.vasp")

print("Before force_near_0:")
scaled_before = target_structure.get_scaled_positions()
print(f"  Atom 124, dim 0: {scaled_before[124, 0]:.8f}")

# Apply force_near_0
forced_structure = force_near_0(target_structure, threshold=0.001)

print("\nAfter force_near_0:")
scaled_after = forced_structure.get_scaled_positions()
print(f"  Atom 124, dim 0: {scaled_after[124, 0]:.8f}")

# Check Cartesian positions to see if the shift actually happened
print("\nCartesian positions:")
cart_before = target_structure.get_positions()
cart_after = forced_structure.get_positions()
print(
    f"  Atom 124, dim 0: {cart_before[124, 0]:.8f} -> {cart_after[124, 0]:.8f} (diff: {cart_after[124, 0] - cart_before[124, 0]:.8f})"
)

# Let's also check what the expected Cartesian shift should be
cell = target_structure.get_cell()
expected_shift = -1.0 * cell[0, 0]  # Shift by -1 unit cell in x-direction
print(f"  Expected Cartesian shift: {expected_shift:.8f}")
