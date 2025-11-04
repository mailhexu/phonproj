# Phonon Mode Decomposition Methodology

This document describes the mathematical methodology and algorithms used in `phonproj/cli.py` for calculating and decomposing structural displacements into phonon mode contributions.

## Overview

The phonon mode decomposition analysis decomposes an arbitrary structural displacement into contributions from individual phonon modes at commensurate q-points. This is useful for understanding which phonon modes dominate a structural transformation, distortion, or relaxation.

## Table of Contents

1. [Coordinate System Transformation](#1-coordinate-system-transformation)
2. [Atom Mapping](#2-atom-mapping)
3. [Center of Mass Alignment](#3-center-of-mass-alignment)
4. [Displacement Calculation](#4-displacement-calculation)
5. [Periodic Boundary Conditions](#5-periodic-boundary-conditions)
6. [Acoustic Mode Projection](#6-acoustic-mode-projection)
7. [Mass-Weighted Normalization](#7-mass-weighted-normalization)
8. [Phonon Mode Decomposition](#8-phonon-mode-decomposition)
9. [Complete Algorithm](#9-complete-algorithm)

---

## 0. ISODISTORT File Processing

### ISODISTORT Format

ISODISTORT files provide a convenient way to specify both undistorted and distorted structures in a single file. This format is commonly used for analyzing symmetry-lowering distortions and mode-selected structural changes.

### File Structure

An ISODISTORT file contains two complete structure definitions:

1. **First structure**: Undistorted (reference) configuration
2. **Second structure**: Distorted configuration

Both structures use the same format as standard structure files (typically VASP POSCAR format), allowing for direct comparison.

### Processing Steps

When using `--isodistort` option:

1. **Parse file format**: Identify structure boundaries and extract both configurations
2. **Load undistorted structure**: Use as reference for displacement calculation
3. **Load distorted structure**: Use as displaced structure for displacement calculation
4. **Validate compatibility**: Ensure both structures have same atom count and types

### Advantages

- **Single file**: No need to manage separate reference/displaced files
- **Guaranteed compatibility**: Both structures designed to be related
- **Common format**: Standard in distortion analysis workflows
- **Reduced errors**: No risk of mismatched structure files

### Implementation

```python
# Load ISODISTORT structures
undistorted, distorted = load_isodistort_structures(isodistort_file)

# Use in standard displacement workflow
displacement = calculate_displacement_vector(
    undistorted, distorted, **kwargs
)
```

---

## 1. Coordinate System Transformation

### Problem Statement

When comparing two structures (reference and displaced), they may have different lattice parameters due to:
- Thermal expansion
- Pressure effects
- Structural relaxation
- Cell optimization

To calculate meaningful displacements, both structures must be expressed in a **common coordinate system**.

### Mathematical Formulation

Given:
- Reference structure with cell matrix $\mathbf{A}_{\text{ref}}$ (3×3 matrix where columns are lattice vectors)
- Displaced structure with cell matrix $\mathbf{A}_{\text{disp}}$
- Atomic positions in Cartesian coordinates: $\mathbf{r}_i^{\text{ref}}$ and $\mathbf{r}_i^{\text{disp}}$

**Step 1: Convert to fractional coordinates**

Each structure is converted to fractional coordinates in its own cell basis:

$$
\mathbf{s}_i^{\text{ref}} = (\mathbf{A}_{\text{ref}})^{-1} \mathbf{r}_i^{\text{ref}}
$$

$$
\mathbf{s}_i^{\text{disp}} = (\mathbf{A}_{\text{disp}})^{-1} \mathbf{r}_i^{\text{disp}}
$$

**Step 2: Transform to common coordinate system**

Express both structures in the reference cell coordinate system by transforming the displaced structure's fractional coordinates:

$$
\tilde{\mathbf{r}}_i^{\text{disp}} = \mathbf{A}_{\text{ref}} \cdot \mathbf{s}_i^{\text{disp}}
$$

$$
\tilde{\mathbf{r}}_i^{\text{ref}} = \mathbf{r}_i^{\text{ref}}
$$

Now both structures are in the same coordinate system (reference cell basis).

### Implementation

```python
# Convert to fractional coordinates
reference_positions_frac = reference_positions @ np.linalg.inv(ref_cell_array)
displaced_positions_frac = displaced_positions @ np.linalg.inv(disp_cell_array)

# Transform displaced structure to reference cell coordinates
displaced_positions_in_ref_cell = displaced_positions_frac @ ref_cell_array
reference_positions_in_ref_cell = reference_positions
```

### Why This Matters

This transformation ensures that:
1. Cell strain/deformation is properly accounted for as part of the displacement
2. Fractional coordinates (relative positions in the unit cell) are preserved
3. The displacement is calculated in a consistent coordinate system

---

## 2. Atom Mapping

### Problem Statement

The reference and displaced structures may have:
- Different atom orderings
- Species substitutions (e.g., Pb → Sr)
- Swapped atomic positions due to symmetry

We need to establish a one-to-one correspondence between atoms.

### Algorithm

Uses the Hungarian algorithm (linear assignment problem) to find the optimal atom mapping that minimizes total displacement:

$$
\text{mapping} = \underset{\pi}{\arg\min} \sum_{i=1}^{N} \|\mathbf{r}_i^{\text{ref}} - \mathbf{r}_{\pi(i)}^{\text{disp}}\|^2
$$

where $\pi$ is a permutation of atom indices.

### Species Substitution

When species substitution occurs (e.g., `--species-map "Pb:Sr"`):
- Atoms of type A in reference are matched to atoms of type B in displaced
- The mass of the reference atom is used for all calculations
- Mapping only considers valid species pairs

### Implementation

```python
mapping, _ = create_atom_mapping(
    reference_atoms,
    displaced_atoms,
    max_cost=100.0,
    warn_threshold=0.5,
    species_map=species_map,
)
```

---

## 3. Center of Mass Alignment

### Problem Statement

Bulk translation (acoustic modes at Γ point) can dominate the displacement, obscuring optical mode contributions. We need to remove this rigid translation.

### Mathematical Formulation

**Step 1: Calculate center of mass**

After transforming to common coordinates, calculate COM for both structures:

$$
\mathbf{R}_{\text{COM}}^{\text{ref}} = \frac{1}{M_{\text{total}}} \sum_{i=1}^{N} m_i \tilde{\mathbf{r}}_i^{\text{ref}}
$$

$$
\mathbf{R}_{\text{COM}}^{\text{disp}} = \frac{1}{M_{\text{total}}} \sum_{i=1}^{N} m_i \tilde{\mathbf{r}}_i^{\text{disp}}
$$

where $M_{\text{total}} = \sum_{i=1}^{N} m_i$

**Step 2: Calculate and remove COM shift**

$$
\Delta\mathbf{R}_{\text{COM}} = \mathbf{R}_{\text{COM}}^{\text{disp}} - \mathbf{R}_{\text{COM}}^{\text{ref}}
$$

$$
\tilde{\mathbf{r}}_i^{\text{disp, aligned}} = \tilde{\mathbf{r}}_i^{\text{disp}} - \Delta\mathbf{R}_{\text{COM}}
$$

### Critical Implementation Detail

⚠️ **WARNING**: COM alignment must be done AFTER coordinate transformation to the reference cell. If done before, the transformation will undo the alignment.

```python
# CORRECT: Align after transformation
displaced_positions_in_ref_cell = displaced_positions_frac @ ref_cell_array
reference_com = sum(masses * reference_positions_in_ref_cell) / total_mass
displaced_com = sum(masses * displaced_positions_in_ref_cell) / total_mass
com_shift = displaced_com - reference_com
displaced_positions_in_ref_cell -= com_shift
```

---

## 4. Displacement Calculation

### Mathematical Formulation

After coordinate transformation and COM alignment, displacement is simply:

$$
\Delta\mathbf{r}_i = \tilde{\mathbf{r}}_i^{\text{disp, aligned}} - \tilde{\mathbf{r}}_i^{\text{ref}}
$$

This gives a displacement vector for each atom:

$$
\mathbf{u} = \{\Delta\mathbf{r}_1, \Delta\mathbf{r}_2, \ldots, \Delta\mathbf{r}_N\} \in \mathbb{R}^{3N}
$$

### Implementation

```python
displacement = displaced_positions_in_ref_cell - reference_positions_in_ref_cell
```

---

## 5. Periodic Boundary Conditions

### Problem Statement

Atoms may cross periodic boundaries, leading to large apparent displacements. We need to find the **minimum image** displacement.

### Mathematical Formulation

**Step 1: Convert displacement to fractional coordinates**

$$
\Delta\mathbf{s}_i = (\mathbf{A}_{\text{ref}})^{-1} \Delta\mathbf{r}_i
$$

**Step 2: Wrap to nearest periodic image**

$$
\Delta\mathbf{s}_i^{\text{wrapped}} = \Delta\mathbf{s}_i - \text{round}(\Delta\mathbf{s}_i)
$$

This maps fractional displacements to the range $[-0.5, 0.5]$.

**Step 3: Convert back to Cartesian**

$$
\Delta\mathbf{r}_i^{\text{wrapped}} = \mathbf{A}_{\text{ref}} \cdot \Delta\mathbf{s}_i^{\text{wrapped}}
$$

### Implementation

```python
# Apply periodic boundary conditions
displacement_frac = displacement @ np.linalg.inv(ref_cell_array)
displacement_frac = displacement_frac - np.round(displacement_frac)
displacement = displacement_frac @ ref_cell_array
```

---

## 6. Acoustic Mode Projection

### Problem Statement

Even after COM alignment, small residual acoustic components may remain due to:
- Numerical precision
- Non-uniform mass distribution
- Cell transformation artifacts

### Mathematical Formulation

**Acoustic mode basis vectors**

For a system with masses $\{m_i\}$, the three acoustic translation modes are:

$$
\mathbf{e}_x = \frac{1}{\sqrt{\sum_i m_i}} \begin{pmatrix} \sqrt{m_1} & 0 & 0 & \sqrt{m_2} & 0 & 0 & \cdots \end{pmatrix}^T
$$

$$
\mathbf{e}_y = \frac{1}{\sqrt{\sum_i m_i}} \begin{pmatrix} 0 & \sqrt{m_1} & 0 & 0 & \sqrt{m_2} & 0 & \cdots \end{pmatrix}^T
$$

$$
\mathbf{e}_z = \frac{1}{\sqrt{\sum_i m_i}} \begin{pmatrix} 0 & 0 & \sqrt{m_1} & 0 & 0 & \sqrt{m_2} & \cdots \end{pmatrix}^T
$$

These are orthonormal mass-weighted translation vectors.

**Projection and removal**

For each acoustic mode $\mathbf{e}_\alpha$ ($\alpha = x, y, z$):

$$
c_\alpha = \sum_{i=1}^{N} \sqrt{m_i} \, \mathbf{e}_\alpha \cdot \Delta\mathbf{r}_i
$$

$$
\mathbf{u}_{\text{acoustic}} = \sum_{\alpha=x,y,z} c_\alpha \mathbf{e}_\alpha
$$

$$
\mathbf{u}_{\text{optical}} = \mathbf{u} - \mathbf{u}_{\text{acoustic}}
$$

### Implementation

```python
displacement, acoustic_projections = project_out_acoustic_modes(
    displacement, reference_atoms
)
```

The `acoustic_projections` array contains $[c_x, c_y, c_z]$.

### Expected Behavior

After proper COM alignment, residual acoustic projections should satisfy:

$$
|c_\alpha| < 10^{-2} \text{ for all } \alpha
$$

If values are larger, COM alignment may have failed.

---

## 7. Mass-Weighted Normalization

### Mathematical Formulation

The mass-weighted norm of a displacement is:

$$
\|\mathbf{u}\|_m = \sqrt{\sum_{i=1}^{N} m_i \|\Delta\mathbf{r}_i\|^2}
$$

For normalized displacement:

$$
\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_m}
$$

This ensures:

$$
\sum_{i=1}^{N} m_i \|\Delta\hat{\mathbf{r}}_i\|^2 = 1
$$

### When to Normalize

- **Use normalization** (`--normalize`): When comparing relative mode contributions across different structures
- **Don't normalize**: When absolute displacement magnitudes are important

### Implementation

```python
# Calculate mass-weighted norm
mass_weighted_norm_sq = np.sum(masses[:, np.newaxis] * np.abs(displacement)**2)
displacement_norm = np.sqrt(mass_weighted_norm_sq)

# Normalize if requested
if normalize:
    displacement_vector = displacement_vector / displacement_norm
```

---

## 8. Phonon Mode Decomposition

### Commensurate Q-Points

For a supercell with dimensions $N_1 \times N_2 \times N_3$, commensurate q-points are:

$$
\mathbf{q}_{ijk} = \left(\frac{i}{N_1}, \frac{j}{N_2}, \frac{k}{N_3}\right)
$$

where $i \in [0, N_1-1]$, $j \in [0, N_2-1]$, $k \in [0, N_3-1]$.

Total q-points: $N_q = N_1 \times N_2 \times N_3 = \det(\mathbf{S})$

### Phonon Mode Basis

Each phonon mode $\nu$ at q-point $\mathbf{q}$ has an eigenvector $\boldsymbol{\epsilon}_{\mathbf{q}\nu}$ (primitive cell).

The supercell mode pattern is:

$$
\mathbf{U}_{\mathbf{q}\nu}(i, \mathbf{R}_p) = \frac{1}{\sqrt{N_q}} \frac{\boldsymbol{\epsilon}_{\mathbf{q}\nu}(i)}{\sqrt{m_i}} e^{i\mathbf{q} \cdot \mathbf{R}_p}
$$

where:
- $i$ indexes atoms in primitive cell
- $\mathbf{R}_p$ is the position of supercell replica $p$
- $N_q$ is the number of q-points

### Projection

The projection coefficient is:

$$
c_{\mathbf{q}\nu} = \sum_{p=1}^{N_{\text{cell}}} \sum_{i=1}^{N_{\text{prim}}} \sqrt{m_i} \, \mathbf{U}_{\mathbf{q}\nu}^*(i, \mathbf{R}_p) \cdot \Delta\mathbf{r}(i, \mathbf{R}_p)
$$

### Contribution

The fractional contribution of mode $(\mathbf{q}, \nu)$ is:

$$
\text{Contribution}_{\mathbf{q}\nu} = \frac{|c_{\mathbf{q}\nu}|^2}{\sum_{\mathbf{q}'\nu'} |c_{\mathbf{q}'\nu'}|^2}
$$

### Completeness

Completeness measures how well the phonon modes span the displacement:

$$
\text{Completeness} = \frac{\sum_{\mathbf{q}\nu} |c_{\mathbf{q}\nu}|^2}{\|\mathbf{u}\|_m^2}
$$

For a complete orthonormal basis, completeness = 1 (or 100%).

---

## 9. Complete Algorithm

### Full Workflow

```
INPUT: reference structure, displaced structure (or ISODISTORT file)

STEP 0: ISODISTORT Processing (if --isodistort used)
├─ Parse ISODISTORT file format
├─ Extract undistorted structure (reference)
├─ Extract distorted structure (displaced)
└─ Proceed with standard workflow

STEP 1: Atom Mapping
├─ Find optimal correspondence between atoms
├─ Handle species substitutions
└─ Reorder displaced structure

STEP 2: Coordinate Transformation
├─ Convert reference to fractional: s_ref = A_ref^(-1) * r_ref
├─ Convert displaced to fractional: s_disp = A_disp^(-1) * r_disp
└─ Transform displaced to ref cell: r_disp' = A_ref * s_disp

STEP 3: Center of Mass Alignment (if --remove-com)
├─ Calculate COM_ref in reference cell coordinates
├─ Calculate COM_disp in reference cell coordinates
├─ Compute shift: Δ_COM = COM_disp - COM_ref
└─ Apply: r_disp'' = r_disp' - Δ_COM

STEP 4: Displacement Calculation
└─ Δr = r_disp'' - r_ref

STEP 5: Periodic Boundary Conditions
├─ Convert to fractional: Δs = A_ref^(-1) * Δr
├─ Wrap: Δs_wrapped = Δs - round(Δs)
└─ Convert back: Δr_wrapped = A_ref * Δs_wrapped

STEP 6: Acoustic Mode Projection (if --remove-com)
├─ Create acoustic basis: e_x, e_y, e_z
├─ Project: c_α = e_α · Δr
└─ Remove: Δr_optical = Δr - Σ_α c_α e_α

STEP 7: Mass-Weighted Normalization (if --normalize)
├─ Calculate: ||u||_m = sqrt(Σ_i m_i |Δr_i|^2)
└─ Normalize: u_hat = u / ||u||_m

STEP 8: Phonon Mode Decomposition
├─ Generate commensurate q-points
├─ Calculate phonon eigenvectors at each q
├─ Project displacement onto each mode
├─ Calculate contributions and completeness
└─ Generate summary tables

OUTPUT: Mode contributions, completeness, q-point analysis
```

### Command Line Usage

```bash
# Basic usage with separate structure files
phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp

# ISODISTORT file analysis (recommended for specific distortions)
phonproj-decompose -p phonopy_params.yaml -s 4x4x2 -i isodistort.txt \
    --remove-com --quiet

# With COM removal and species substitution
phonproj-decompose -p phonopy_params.yaml -s 16x1x1 -d CONTCAR \
    --remove-com --species-map "Pb:Sr"

# Normalized displacement with output file
phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp \
    --normalize --remove-com -o results.txt
```

### Key Parameters

| Parameter | Description | When to Use |
|-----------|-------------|-------------|
| `-i, --isodistort` | Use ISODISTORT file format (contains both structures) | For analyzing specific symmetry-lowering distortions |
| `--remove-com` | Align by COM and project out acoustic modes | Always for optical mode analysis |
| `--normalize` | Mass-weight normalize displacement | For comparing relative contributions |
| `--species-map` | Map species substitutions | For doping/substitution analysis |
| `--no-sort` | Don't sort by contribution | For q-point ordered output |
| `--quiet` | Reduce output verbosity | For large structures or automated processing |

---

## Mathematical Notes

### Orthonormality of Acoustic Modes

The acoustic translation modes satisfy:

$$
\mathbf{e}_\alpha \cdot \mathbf{e}_\beta = \delta_{\alpha\beta}
$$

$$
\sum_{i=1}^{N} m_i = M_{\text{total}}
$$

### Gauge Convention

Phonopy uses the "R gauge" (real gauge) where eigenvectors are chosen to be real when possible. This is handled automatically by the `PhononModes` class.

### Mass Weighting Convention

Phonopy convention for mass-weighted eigenvectors:

$$
\tilde{\boldsymbol{\epsilon}}_{\mathbf{q}\nu}(i) = \frac{\boldsymbol{\epsilon}_{\mathbf{q}\nu}(i)}{\sqrt{m_i}}
$$

This convention is consistently applied throughout the code.

### Numerical Precision

- Tolerance for mode identification: $10^{-6}$
- Minimum contribution for printing: $10^{-10}$
- Expected residual acoustic: $< 10^{-2}$ after COM alignment

---

## References

1. **Phonopy**: A. Togo and I. Tanaka, Scr. Mater. 108, 1-5 (2015)
2. **Zone folding**: Supercell phonon modes from primitive cell
3. **Hungarian algorithm**: Optimal assignment for atom mapping
4. **Projection formalism**: Standard phonon decomposition theory

---

## Implementation Files

- **Main implementation**: `phonproj/cli.py`
- **Core algorithms**: `phonproj/core/structure_analysis.py`
- **Phonon modes**: `phonproj/modes.py`
- **I/O functions**: `phonproj/core/io.py`

---

*Last updated: 2025-11-02*
