#!/usr/bin/env python3
"""
Check if the generated phonon mode displacements are orthonormal in the supercell.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes, create_supercell


def test_mode_orthonormality():
    """Test if phonon modes are orthonormal in supercell"""
    print("🔹 Testing phonon mode orthonormality in supercell...")

    # Load phonon modes
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    # Create supercell
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    # Get commensurate q-points
    result = phonon_modes.get_commensurate_qpoints(supercell_matrix, detailed=True)
    matched_indices = result.get("matched_indices", [])

    print(f"   Found {len(matched_indices)} commensurate q-points")

    # Generate a few mode displacements and test orthonormality
    mode_displacements = []
    mode_info = []

    # Generate first 10 modes for testing
    count = 0
    for q_index in matched_indices[:3]:  # First 3 q-points
        for mode_index in range(3):  # First 3 modes each
            try:
                mode_disp = phonon_modes.generate_mode_displacement(
                    q_index=q_index,
                    mode_index=mode_index,
                    supercell_matrix=supercell_matrix,
                    amplitude=1.0,
                )

                # Mass-weighted normalize
                masses = np.repeat(supercell.get_masses(), 3)
                mode_flat = mode_disp.ravel()
                mode_norm = np.sqrt(np.sum(masses * mode_flat * mode_flat))

                if mode_norm > 1e-10:
                    mode_disp_norm = mode_disp / mode_norm
                    mode_displacements.append(mode_disp_norm)
                    mode_info.append((q_index, mode_index, mode_norm))
                    count += 1
                    if count >= 9:
                        break
            except Exception as e:
                print(f"   Failed to generate mode Q={q_index}, M={mode_index}: {e}")

        if count >= 9:
            break

    print(f"   Generated {len(mode_displacements)} mode displacements")

    # Test orthonormality
    masses = np.repeat(supercell.get_masses(), 3)

    orthonormality_matrix = np.zeros((len(mode_displacements), len(mode_displacements)))

    for i in range(len(mode_displacements)):
        for j in range(len(mode_displacements)):
            disp_i = mode_displacements[i].ravel()
            disp_j = mode_displacements[j].ravel()

            # Mass-weighted inner product
            inner_product = np.sum(masses * disp_i.conj() * disp_j)
            orthonormality_matrix[i, j] = inner_product.real

    print(f"\n   Orthonormality matrix (should be identity):")
    print(f"   Diagonal elements (should be 1.0):")
    for i in range(len(mode_displacements)):
        q_idx, mode_idx, norm = mode_info[i]
        diag_element = orthonormality_matrix[i, i]
        print(
            f"     Q={q_idx:2d}, M={mode_idx:2d}: {diag_element:.6f} (norm was {norm:.3f})"
        )

    print(f"\n   Off-diagonal elements (should be 0.0):")
    max_off_diag = 0.0
    for i in range(len(mode_displacements)):
        for j in range(i + 1, len(mode_displacements)):
            off_diag = abs(orthonormality_matrix[i, j])
            if off_diag > max_off_diag:
                max_off_diag = off_diag
            if off_diag > 1e-6:
                q_i, m_i, _ = mode_info[i]
                q_j, m_j, _ = mode_info[j]
                print(
                    f"     Q={q_i:2d},M={m_i:2d} x Q={q_j:2d},M={m_j:2d}: {off_diag:.6f}"
                )

    print(f"   Maximum off-diagonal element: {max_off_diag:.2e}")

    if max_off_diag < 1e-10:
        print(f"   ✓ Modes are orthonormal")
    else:
        print(f"   ✗ Modes are NOT orthonormal")

    return orthonormality_matrix


def test_mode_completeness():
    """Test if the mode set spans the full space"""
    print(f"\n🔹 Testing mode completeness...")

    # Load phonon modes
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    # Create supercell
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    total_dof = len(supercell) * 3
    print(f"   Total degrees of freedom: {total_dof}")
    print(
        f"   Total available modes: {len(phonon_modes.qpoints)} q-points × {phonon_modes.n_modes} modes = {len(phonon_modes.qpoints) * phonon_modes.n_modes}"
    )

    # Get commensurate q-points
    result = phonon_modes.get_commensurate_qpoints(supercell_matrix, detailed=True)
    matched_indices = result.get("matched_indices", [])
    total_commensurate_modes = len(matched_indices) * phonon_modes.n_modes

    print(
        f"   Commensurate modes: {len(matched_indices)} q-points × {phonon_modes.n_modes} modes = {total_commensurate_modes}"
    )

    if total_commensurate_modes == total_dof:
        print(f"   ✓ Mode count matches degrees of freedom")
    else:
        print(
            f"   ✗ Mode count mismatch: {total_commensurate_modes} modes vs {total_dof} DOF"
        )

    return total_commensurate_modes == total_dof


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Phonon Mode Orthonormality and Completeness")
    print("=" * 80)

    try:
        orthonormality_matrix = test_mode_orthonormality()
        completeness_ok = test_mode_completeness()

        print(f"\n" + "=" * 80)
        print("Summary:")
        print(
            f"  - Mode orthonormality: {'✓ PASS' if np.max(np.abs(orthonormality_matrix - np.eye(len(orthonormality_matrix)))) < 1e-10 else '✗ FAIL'}"
        )
        print(f"  - Mode completeness: {'✓ PASS' if completeness_ok else '✗ FAIL'}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
