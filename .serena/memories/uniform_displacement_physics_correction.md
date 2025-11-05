## Uniform Displacement Physics Correction

**Problem:** The original test was incorrectly allowing uniform displacement to project onto optical modes in supercells, which violates fundamental phonon physics.

**Physics Principle:** A uniform displacement corresponds to translation of the entire crystal, which should ONLY project onto acoustic modes at the Gamma point (q=0). All other projections should be zero.

**Changes Made:**

1. **Updated test expectations** in `test_uniform_displacement_decomposition_4x1x1_supercell`:
   - Gamma acoustic modes should have significant projection (> 0.5)
   - Non-Gamma acoustic modes should have zero projection (< 0.001) 
   - Optical modes should have zero projection (< 0.001)
   - Total magnitude should be close to 1 (complete basis)

2. **Updated docstring** to clarify correct physics and current implementation status

3. **Removed duplicate code section** that was redundant

**Current Status:**
- Test now correctly FAILS because implementation has bug
- Output shows: Gamma acoustic = 0.024, Optical = 0.882 (should be reversed)
- Test properly enforces that only Gamma acoustic modes should be non-zero

**Next Steps:**
The failing test correctly identifies that the current `decompose_displacement` method has a physics bug where uniform displacement incorrectly projects onto optical modes instead of only Gamma acoustic modes.

This is now a proper regression test that will catch implementation bugs.