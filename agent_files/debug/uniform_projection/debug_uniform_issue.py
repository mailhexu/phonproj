"""
Debug uniform displacement projection issue.

Purpose:
- Investigate why uniform displacement projects onto optical modes
- Should only project onto Gamma acoustic modes (modes 3,4,5)

How to run:
    uv run python agent_files/debug/uniform_projection/debug_uniform_issue.py
"""

from pathlib import Path
import numpy as np
from phonproj.modes import PhononModes

# Load BaTiO3 data
data_path = (
    Path(__file__).parent.parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
)

# Load only Gamma point
qpoints = np.array([[0.0, 0.0, 0.0]])
modes_gamma = PhononModes.from_phonopy_yaml(str(data_path), qpoints=qpoints)

print("=== Checking Gamma point mode frequencies ===")
for mode_idx in range(modes_gamma.n_modes):
    freq = modes_gamma.frequencies[0, mode_idx]
    mode_type = "ACOUSTIC" if mode_idx in [3, 4, 5] else "OPTICAL"
    print(f"Mode {mode_idx:2d}: {freq:8.2f} THz  ({mode_type})")

print("\n=== Generating supercell displacement for Gamma acoustic mode 4 ===")
supercell_matrix = np.diag([4, 1, 1])

# Generate displacement for acoustic mode 4 at Gamma
acoustic_disp = modes_gamma.generate_mode_displacement(
    q_index=0,
    mode_index=4,  # Acoustic mode
    supercell_matrix=supercell_matrix,
    amplitude=1.0,
    normalize=False,
)

print(f"Displacement shape: {acoustic_disp.shape}")
print(f"Displacement dtype: {acoustic_disp.dtype}")
print(f"Is uniform? Check if all displacements are similar:")

# Check if the acoustic mode displacement is uniform
real_part = acoustic_disp.real
print(f"  Min x-displacement: {real_part[:, 0].min():.6f}")
print(f"  Max x-displacement: {real_part[:, 0].max():.6f}")
print(f"  Std x-displacement: {real_part[:, 0].std():.6f}")
print(f"  Min y-displacement: {real_part[:, 1].min():.6f}")
print(f"  Max y-displacement: {real_part[:, 1].max():.6f}")
print(f"  Std y-displacement: {real_part[:, 1].std():.6f}")
print(f"  Min z-displacement: {real_part[:, 2].min():.6f}")
print(f"  Max z-displacement: {real_part[:, 2].max():.6f}")
print(f"  Std z-displacement: {real_part[:, 2].std():.6f}")

print("\n=== Generating supercell displacement for Gamma optical mode 6 ===")
# Generate displacement for optical mode 6 at Gamma
optical_disp = modes_gamma.generate_mode_displacement(
    q_index=0,
    mode_index=6,  # Optical mode
    supercell_matrix=supercell_matrix,
    amplitude=1.0,
    normalize=False,
)

print(f"Displacement shape: {optical_disp.shape}")
print(f"Is uniform? Check if all displacements are similar:")

# Check if the optical mode displacement is uniform
real_part = optical_disp.real
print(f"  Min x-displacement: {real_part[:, 0].min():.6f}")
print(f"  Max x-displacement: {real_part[:, 0].max():.6f}")
print(f"  Std x-displacement: {real_part[:, 0].std():.6f}")

print("\n=== Creating TRUE uniform displacement ===")
# Create a TRUE uniform displacement (all atoms move equally)
n_supercell_atoms = 20  # 5 atoms * 4 = 20
true_uniform = np.ones((n_supercell_atoms, 3))
print(f"True uniform shape: {true_uniform.shape}")

# Normalize it
target_masses = np.repeat(modes_gamma.atomic_masses, 4)  # Repeat for 4x supercell
target_flat = true_uniform.ravel()
masses_repeated = np.repeat(target_masses, 3)
mass_weighted_norm = np.sqrt(np.sum(masses_repeated * target_flat * target_flat))
true_uniform_normalized = true_uniform / mass_weighted_norm

print(f"Mass-weighted norm: {mass_weighted_norm:.6f}")

# Now project this onto the acoustic mode displacement
from phonproj.core.structure_analysis import project_displacements_between_supercells
from phonproj.modes import create_supercell

source_supercell = create_supercell(modes_gamma.primitive_cell, supercell_matrix)

# Project true uniform onto acoustic mode 4
coeff_acoustic = project_displacements_between_supercells(
    source_displacement=acoustic_disp,
    target_displacement=true_uniform_normalized,
    source_supercell=source_supercell,
    target_supercell=source_supercell,
    normalize=False,
    use_mass_weighting=True,
)

# Project true uniform onto optical mode 6
coeff_optical = project_displacements_between_supercells(
    source_displacement=optical_disp,
    target_displacement=true_uniform_normalized,
    source_supercell=source_supercell,
    target_supercell=source_supercell,
    normalize=False,
    use_mass_weighting=True,
)

print(f"\nProjection onto acoustic mode 4: {abs(coeff_acoustic):.6f}")
print(f"Projection onto optical mode 6: {abs(coeff_optical):.6f}")

if abs(coeff_optical) > 0.01:
    print(
        "\n!!! PROBLEM: Uniform displacement projects significantly onto optical mode !!!"
    )
    print(
        "This violates physics - uniform displacement should only activate acoustic modes."
    )
else:
    print("\n✓ OK: Optical projection is negligible")
