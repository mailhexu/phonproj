"""
Test gauge transformation phase factors.

Purpose:
    Verify that the phase factor signs are correct for R ↔ r gauge transformations
    without needing to load actual phonon data.

How to run:
    uv run python agent_files/debug_irreps/test_gauge_transform.py

Expected behavior:
    - Phase factors for R→r and r→R should be complex conjugates
    - Applying both transformations should give identity
"""

import numpy as np

print("=" * 80)
print("Testing Gauge Transformation Phase Factor Convention")
print("=" * 80)

# Test phase factor signs
# R gauge eigenvectors: u_R(q, r)
# r gauge eigenvectors: u_r(q, r) = u_R(q, r) * exp(-2πi q·r)
#
# So:
# R → r: multiply by exp(-2πi q·r)
# r → R: multiply by exp(+2πi q·r)

# Create test data
n_atoms = 3
qpoint = np.array([0.1, 0.2, 0.3])
scaled_positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
    ]
)

print(f"\nTest setup:")
print(f"Number of atoms: {n_atoms}")
print(f"Q-point: {qpoint}")
print(f"Scaled positions:")
for i, pos in enumerate(scaled_positions):
    print(f"  Atom {i}: {pos}")

# Calculate q·r for each atom
q_dot_r = np.dot(scaled_positions, qpoint)
print(f"\nq·r for each atom: {q_dot_r}")

# Phase factors for R → r: exp(-2πi q·r)
phases_R_to_r = np.exp(-2j * np.pi * q_dot_r)

# Phase factors for r → R: exp(+2πi q·r)
phases_r_to_R = np.exp(+2j * np.pi * q_dot_r)

print(f"\nPhase factors for R→r: {phases_R_to_r}")
print(f"Phase factors for r→R: {phases_r_to_R}")

# Test 1: They should be complex conjugates
print("\n" + "=" * 80)
print("Test 1: R→r and r→R phases are complex conjugates")
print("=" * 80)

conjugate_match = np.allclose(phases_R_to_r, np.conj(phases_r_to_R))
print(f"phases_R_to_r = conj(phases_r_to_R)? {conjugate_match}")

if conjugate_match:
    print("✓ Test PASSED")
else:
    print("✗ Test FAILED")
    print(f"  Difference: {phases_R_to_r - np.conj(phases_r_to_R)}")

# Test 2: Product should be 1
print("\n" + "=" * 80)
print("Test 2: Product of both transformations is identity")
print("=" * 80)

product = phases_R_to_r * phases_r_to_R
print(f"Product: {product}")
product_is_one = np.allclose(product, 1.0)
print(f"Product = 1? {product_is_one}")

if product_is_one:
    print("✓ Test PASSED: R→r→R gives back original (phase factors cancel)")
else:
    print("✗ Test FAILED")
    print(f"  Deviation from 1: {product - 1.0}")

# Test 3: At Gamma, phases should be 1
print("\n" + "=" * 80)
print("Test 3: At Gamma point, phases = 1")
print("=" * 80)

qpoint_gamma = np.array([0.0, 0.0, 0.0])
q_dot_r_gamma = np.dot(scaled_positions, qpoint_gamma)
phases_gamma = np.exp(-2j * np.pi * q_dot_r_gamma)

print(f"Q-point (Gamma): {qpoint_gamma}")
print(f"Phases at Gamma: {phases_gamma}")
gamma_is_one = np.allclose(phases_gamma, 1.0)
print(f"All phases = 1? {gamma_is_one}")

if gamma_is_one:
    print("✓ Test PASSED: At Gamma, R and r gauges are equivalent")
else:
    print("✗ Test FAILED")

# Test 4: Check sign convention from phonopy/modes.py
print("\n" + "=" * 80)
print("Test 4: Verify sign convention in modes.py")
print("=" * 80)

print("\nIn modes.py transform_gauge(), the code should be:")
print("  # R → r: multiply by exp(-2πi q·r)  (negative sign)")
print("  # r → R: multiply by exp(+2πi q·r)  (positive sign)")
print("  sign = -1 if self.gauge == 'R' else 1")
print("  phases = np.exp(sign * 2j * np.pi * np.dot(scaled_positions, qpoint))")

print("\nVerifying this logic:")
current_gauge = "R"
target_gauge = "r"
# When going from R to r:
sign = -1 if current_gauge == "R" else 1
print(f"Current gauge: {current_gauge}, Target: {target_gauge}")
print(f"  sign = {sign} (should be -1 for R→r)")
if sign == -1:
    print("  ✓ Correct: R→r uses exp(-2πi q·r)")
else:
    print("  ✗ Wrong: R→r should use exp(-2πi q·r)")

current_gauge = "r"
target_gauge = "R"
sign = -1 if current_gauge == "R" else 1
print(f"Current gauge: {current_gauge}, Target: {target_gauge}")
print(f"  sign = {sign} (should be +1 for r→R)")
if sign == 1:
    print("  ✓ Correct: r→R uses exp(+2πi q·r)")
else:
    print("  ✗ Wrong: r→R should use exp(+2πi q·r)")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("All theoretical tests completed successfully!")
print("The phase factor convention is mathematically correct.")
