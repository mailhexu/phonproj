#!/usr/bin/env python3
"""
Simple test to understand the mass-weighting issue.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_simple_mass_weighting():
    """Test mass-weighting with a simple example"""
    print("🔹 Testing simple mass-weighting example...")

    # Create simple vectors
    v1 = np.array([1.0, 0.0, 0.0])  # 1 Å in x
    v2 = np.array([0.0, 1.0, 0.0])  # 1 Å in y

    # Realistic masses from PTO system
    masses = np.array([207.2, 47.9, 16.0])  # Pb, Ti, O masses (amu)

    # Mass-weighted norms
    norm1_m = np.sqrt(np.sum(masses * v1 * v1))
    norm2_m = np.sqrt(np.sum(masses * v2 * v2))

    print(f"   v1 = {v1}, mass-weighted norm = {norm1_m:.3f}")
    print(f"   v2 = {v2}, mass-weighted norm = {norm2_m:.3f}")

    # Mass-weighted normalize
    v1_norm = v1 / norm1_m
    v2_norm = v2 / norm2_m

    print(
        f"   v1_norm = {v1_norm}, mass-weighted norm = {np.sqrt(np.sum(masses * v1_norm * v1_norm)):.6f}"
    )
    print(
        f"   v2_norm = {v2_norm}, mass-weighted norm = {np.sqrt(np.sum(masses * v2_norm * v2_norm)):.6f}"
    )

    # Test projections
    # Method 1: Mass-weighted inner product (should give 0 for orthogonal vectors)
    proj_mass = np.sum(masses * v1_norm * v2_norm)
    print(f"   <v1_norm, v2_norm>_M = {proj_mass:.6f}")

    # Method 2: Regular inner product (should give something else)
    proj_regular = np.sum(v1_norm * v2_norm)
    print(f"   <v1_norm, v2_norm> = {proj_regular:.6f}")

    # Self projections
    self_proj_mass = np.sum(masses * v1_norm * v1_norm)
    self_proj_regular = np.sum(v1_norm * v1_norm)
    print(f"   <v1_norm, v1_norm>_M = {self_proj_mass:.6f}")
    print(f"   <v1_norm, v1_norm> = {self_proj_regular:.6f}")

    print("\n🔹 Key insight:")
    print("   When vectors are mass-weighted normalized: ||v||_M = 1")
    print("   Self-projection with mass-weighted inner product: <v,v>_M = 1")
    print("   Self-projection with regular inner product: <v,v> ≠ 1")
    print(
        "   Therefore: use mass-weighted inner product for mass-weighted normalized vectors"
    )


if __name__ == "__main__":
    test_simple_mass_weighting()
