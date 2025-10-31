# Mathematical Equations for Phonon Eigenmode Analysis

This document contains the mathematical foundations and equations used in the phonon eigenmode analysis, particularly for Step 7 supercell displacement generation and completeness testing.

## Table of Contents

1. [Basic Phonon Theory](#basic-phonon-theory)
2. [Mass-Weighted Coordinates](#mass-weighted-coordinates)
3. [Supercell Construction](#supercell-construction)
4. [Eigenmode Orthogonality](#eigenmode-orthogonality)
5. [Completeness Relations](#completeness-relations)
6. [Normalization Conventions](#normalization-conventions)
7. [Supercell Displacement Generation](#supercell-displacement-generation)

---

## Basic Phonon Theory

### Dynamical Matrix
The dynamical matrix in reciprocal space is defined as:

$$D_{\alpha\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_\alpha M_\beta}} \sum_{\mathbf{R}} \Phi_{\alpha\beta}(\mathbf{R}) e^{i\mathbf{q} \cdot \mathbf{R}}$$

where:
- $D_{\alpha\beta}(\mathbf{q})$ is the dynamical matrix element between atoms $\alpha$ and $\beta$ at wavevector $\mathbf{q}$
- $M_\alpha, M_\beta$ are the masses of atoms $\alpha$ and $\beta$
- $\Phi_{\alpha\beta}(\mathbf{R})$ are the force constants between atoms $\alpha$ and $\beta$ separated by lattice vector $\mathbf{R}$
- $\mathbf{q}$ is the wavevector in reciprocal space

### Eigenvalue Problem
The phonon eigenvalue problem is:

$$\sum_\beta D_{\alpha\beta}(\mathbf{q}) e_\beta(\mathbf{q},\nu) = \omega^2(\mathbf{q},\nu) e_\alpha(\mathbf{q},\nu)$$

where:
- $e_\alpha(\mathbf{q},\nu)$ is the eigenvector component for atom $\alpha$, wavevector $\mathbf{q}$, and mode $\nu$
- $\omega(\mathbf{q},\nu)$ is the phonon frequency for mode $\nu$ at wavevector $\mathbf{q}$

---

## Mass-Weighted Coordinates

### Mass-Weighted Displacement
For a displacement vector $\mathbf{u}$, the mass-weighted displacement is:

$$u^{\text{MW}}_\alpha = \sqrt{M_\alpha} \cdot \mathbf{u}_\alpha$$

### Mass-Weighted Inner Product
The mass-weighted inner product between two displacement vectors $\mathbf{u}$ and $\mathbf{v}$ is:

$$\langle\mathbf{u}|\mathbf{v}\rangle_M = \sum_\alpha M_\alpha (\mathbf{u}_\alpha \cdot \mathbf{v}_\alpha)$$

### Mass-Weighted Norm
The mass-weighted norm of a displacement vector $\mathbf{u}$ is:

$$\|\mathbf{u}\|_M = \sqrt{\langle\mathbf{u}|\mathbf{u}\rangle_M} = \sqrt{\sum_\alpha M_\alpha \|\mathbf{u}_\alpha\|^2}$$

---

## Supercell Construction

### Supercell Matrix
A supercell is defined by a $3 \times 3$ transformation matrix $\mathbf{S}$:

$$\mathbf{a}'_i = \sum_j S_{ij} \mathbf{a}_j$$

where $\mathbf{a}_i$ are the primitive lattice vectors and $\mathbf{a}'_i$ are the supercell lattice vectors.

### Number of Primitive Cells
The number of primitive cells in the supercell is:

$$N = |\det(\mathbf{S})|$$

### Commensurate Q-points
Q-points that are commensurate with a supercell satisfy:

$$\mathbf{S}^T \cdot \mathbf{q} = \mathbf{k}$$

where $\mathbf{k}$ is an integer vector and $\mathbf{S}^T$ is the transpose of the supercell matrix.

**Commensurability Check**: For a q-point $\mathbf{q}$ to be commensurate with supercell matrix $\mathbf{S}$, the vector $\mathbf{S}^T \cdot \mathbf{q}$ must have integer components (within numerical tolerance). Non-commensurate combinations are automatically detected and blocked with helpful error messages.

For a supercell with matrix $\mathbf{S}$, the commensurate q-points are:

$$\mathbf{q}_{mnp} = \left(\frac{m}{N_1}, \frac{n}{N_2}, \frac{p}{N_3}\right)$$

where $N_i$ are the diagonal elements of $\mathbf{S}$ for diagonal supercell matrices.

---

## Eigenmode Orthogonality

### Orthogonality Within Q-point
Eigenmodes at the same q-point are orthogonal:

$$\langle\mathbf{e}(\mathbf{q},\mu)|\mathbf{e}(\mathbf{q},\nu)\rangle = \delta_{\mu\nu}$$

where $\delta_{\mu\nu}$ is the Kronecker delta.

### Orthogonality Between Q-points (Bloch Theorem)
According to Bloch theorem, eigenmodes from different commensurate q-points are orthogonal in supercell space:

$$\langle\mathbf{u}(\mathbf{q}_1,\mu)|\mathbf{u}(\mathbf{q}_2,\nu)\rangle_M = 0 \quad \text{for } \mathbf{q}_1 \neq \mathbf{q}_2$$

This orthogonality is exact in the supercell representation, as the phase factors $e^{i\mathbf{q} \cdot \mathbf{R}}$ associated with different q-points are orthogonal when summed over the supercell lattice vectors. Small deviations from perfect orthogonality (< 1e-12) may occur only due to numerical precision.

---

## Completeness Relations

### Perfect Orthonormal Basis
For a complete orthonormal basis $\{\boldsymbol{\phi}_i\}$ where $\|\boldsymbol{\phi}_i\| = 1$, any vector $\mathbf{v}$ can be expanded as:

$$\sum_i |\langle\mathbf{v}|\boldsymbol{\phi}_i\rangle|^2 = \|\mathbf{v}\|^2$$

### Phonon Eigenmodes in Supercells
For phonon eigenmodes normalized to unit mass-weighted norm ($\|\boldsymbol{\phi}_i\| = 1$), the completeness relation for a normalized test vector $\|\mathbf{v}\| = 1$ is:

$$\sum_i |\langle\mathbf{v}|\boldsymbol{\phi}_i\rangle|^2 = 1$$

According to Bloch theorem, the set of eigenmodes from all commensurate q-points forms a complete orthonormal basis for the supercell displacement space.

### Practical Completeness Test
For a normalized vector $\|\mathbf{v}\| = 1$ and eigenmodes with norm $1$, the sum of projection squares should equal exactly:

$$\sum_{\mathbf{q},\nu} |\langle\mathbf{v}|\boldsymbol{\phi}(\mathbf{q},\nu)\rangle|^2 = 1.0$$

This reflects the fundamental property that phonon eigenmodes from commensurate q-points form a complete orthonormal basis, ensuring perfect completeness in the supercell representation.

---

## Normalization Conventions

### Step 7 Normalization
In the Step 7 implementation, each supercell eigenmode is normalized to:

$$\|\mathbf{u}(\mathbf{q},\nu)\|_M = \text{amplitude}$$

where:
- $\text{amplitude}$ is the input amplitude parameter (typically 1.0)
- This creates a proper orthonormal basis for the supercell displacement space

### Orthogonality Test Requirements
The orthogonality tests require:

$$\|\mathbf{u}(\mathbf{q},\nu)\|_M = 1.0$$

This ensures that modes form an orthonormal basis for the supercell space.

### Completeness Test Adaptation
For a complete orthonormal basis, the completeness test expects the exact sum:

$$\sum_{\mathbf{q},\nu} |\langle\mathbf{v}|\mathbf{u}(\mathbf{q},\nu)\rangle|^2 = 1.0$$

This reflects Bloch theorem: eigenmodes from commensurate q-points form a complete orthonormal basis for the supercell displacement space, ensuring perfect completeness without correction factors.

---

## Supercell Displacement Generation

### ASE Supercell Construction
The supercell atomic positions are generated using ASE's `make_supercell()` method:

```python
supercell_atoms = make_supercell(primitive_atoms, supercell_matrix)
```

### Eigenmode Phase Application
For a primitive cell eigenmode $\mathbf{e}(\mathbf{q},\nu)$, the supercell displacement is:

$$\mathbf{u}_{\text{sc}}(\mathbf{R}_i + \mathbf{r}_\alpha) = \mathbf{e}_\alpha(\mathbf{q},\nu) \times e^{i\mathbf{q} \cdot \mathbf{R}_i} \times \text{amplitude}$$

where:
- $\mathbf{R}_i$ is the lattice vector to primitive cell $i$
- $\mathbf{r}_\alpha$ is the position of atom $\alpha$ within the primitive cell
- $\mathbf{e}_\alpha(\mathbf{q},\nu)$ is the eigenmode component for atom $\alpha$

### Mass Assignment
The mass array for the supercell is constructed by tiling the primitive cell masses:

$$\mathbf{M}_{\text{supercell}} = \text{tile}(\mathbf{M}_{\text{primitive}}, N)$$

### Final Normalization
The generated displacement is normalized to the target norm:

$$\mathbf{u}_{\text{normalized}} = \mathbf{u}_{\text{sc}} \times \frac{\text{target\_norm}}{\|\mathbf{u}_{\text{sc}}\|_M}$$

where $\text{target\_norm} = \text{amplitude}$ (typically 1.0).

---

## Implementation Notes

### Precision Requirements
- Orthogonality precision: $< 1 \times 10^{-6}$ (within q-point), $< 1 \times 10^{-12}$ (between q-points)
- Normalization precision: $< 1 \times 10^{-10}$
- Completeness tolerance: $|1.0 - \text{actual}| < 1 \times 10^{-4}$ (exact completeness expected)

### Key Algorithmic Choices
1. **ASE vs Phonopy**: ASE's `make_supercell()` preserves atom ordering better than Phonopy's supercell API
2. **Mass-weighted calculations**: All inner products and norms use mass weighting
3. **Unit normalization**: Each eigenmode normalized to mass-weighted norm = 1.0 for proper orthonormal basis

### Physical Interpretation
The unit normalization ensures that the eigenmodes form a complete orthonormal basis for the supercell displacement space. According to Bloch theorem, this basis is exactly complete, meaning the sum of projection squares equals 1.0 exactly for any normalized displacement vector. This perfect completeness reflects the fundamental orthogonality of Bloch waves from different q-points in the supercell representation.