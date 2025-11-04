#!/usr/bin/env python3
"""
Find the atom with y-coordinate 4.909171.
"""

from ase.io import read

target_structure = read("data/yajundata/disp.vasp")
scaled_positions = target_structure.get_scaled_positions()
cartesian_positions = target_structure.get_positions()

print("Looking for atom with y-coordinate around 4.909171...")

for i, pos in enumerate(cartesian_positions):
    if abs(pos[1] - 4.909171) < 0.001:
        print(f"\nAtom {i}:")
        print(f"  Cartesian: {pos}")
        print(f"  Scaled: {scaled_positions[i]}")
        print(f"  Species: {target_structure.get_chemical_symbols()[i]}")
        break
