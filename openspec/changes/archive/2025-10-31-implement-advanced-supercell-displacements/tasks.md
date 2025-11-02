## 1. Core Implementation

- [ ] 1.1 Replace `_calculate_supercell_displacements` with phonopy API calls
- [ ] 1.2 Implement function to generate supercell displacements for all modes at given q-point
- [ ] 1.3 Implement function to get commensurate q-points for supercell
- [ ] 1.4 Implement function to generate displacement lists for all commensurate q-points

## 2. Testing Implementation

- [ ] 2.1 Test Gamma q-point (1x1x1) orthonormality with mass-weighted inner product
- [ ] 2.2 Test Gamma q-point (1x1x1) completeness verification 
- [ ] 2.3 Test displacement generation for all commensurate q-points
- [ ] 2.4 Test non-Gamma q-point (2x2x2) orthogonality and norm verification
- [ ] 2.5 Test completeness for all commensurate q-points (2x2x2 supercell)

## 3. Integration and Validation

- [ ] 3.1 Ensure backward compatibility with existing eigendisplacement functionality
- [ ] 3.2 Run all existing tests to ensure no regressions
- [ ] 3.3 Update documentation and examples as needed