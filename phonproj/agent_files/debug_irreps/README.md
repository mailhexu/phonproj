# Debug: IrReps Analysis & Thermal Displacements

## What was debugged/implemented

### Issue 1: Signed Structure Generation (IMPLEMENTED ✓)

**File**: `example_mode_summary_and_thermal.py:154-178`

**Feature**: Generate both positive (+) and negative (-) displacement structures for each phonon mode.

**Implementation**:
```python
for mode_idx in range(n_modes):
    mode_disp = thermal_displacements[mode_idx].real
    
    # Generate both + and - displacements
    for sign, sign_str in [(+1, "+"), (-1, "-")]:
        displaced_positions = equilibrium_positions + sign * mode_disp
        filename = f"vasp_mode{mode_idx + 1}{sign_str}.vasp"
        save_structure(displaced_positions, filename)
```

**Output**: For N modes, generates 2N + 1 structures:
- `vasp_mode0.vasp` (undisplaced)
- `vasp_mode1+.vasp`, `vasp_mode1-.vasp` (mode 1, both signs)
- `vasp_mode2+.vasp`, `vasp_mode2-.vasp` (mode 2, both signs)
- ... up to N

### Issue 2: Acoustic Mode Divergence (FIXED ✓)

**File**: `phonproj/modes.py:518-568` (`_calculate_thermal_amplitudes`)

**Problem**: Acoustic phonon modes (frequency ≈ 0) caused divergence in thermal displacement calculations because the formula contains `1/√ω`. When ω → 0, this diverges to infinity.

**Solution**: Added frequency threshold check (lines 539-544):
```python
frequency_threshold = 0.1  # THz (roughly 3.3 cm^-1)
if abs(frequency_thz) < frequency_threshold:
    # Return zero displacement for acoustic modes
    return np.zeros((self.n_atoms, 3), dtype=complex)
```

**Physics rationale**: 
- Acoustic modes at Gamma (q=0) represent rigid translations of the entire crystal
- They don't contribute to internal thermal motion
- In the thermodynamic limit, they don't contribute to thermal properties
- Setting amplitude=0 prevents unphysical divergence

### Issue 2: IrReps Analysis Labels (REFACTORED ✓)

**File**: `phonproj/modes.py:845-925` (`_run_irreps_analysis`)

**Status**: Code refactored to match `IrRepsEigen` pattern from `irreps_anaddb.py`

The `_IrRepsLocal` class now properly inherits from `IrReps` and `IrRepLabels` following the working pattern. Needs testing with real phonon data to verify label extraction works correctly.

## Files in this directory

- `test_signed_structures.py`: Documentation showing +/- structure generation **Status: DOCUMENTED ✓**
- `test_acoustic_modes.py`: Test acoustic mode threshold logic **Status: ALL TESTS PASS ✓**
- `test_gauge_transform.py`: Test gauge transformation phase factors **Status: ALL TESTS PASS ✓**
- `test_irreps_simple.py`: Test refactored irreps implementation **Status: PARTIAL ✓**
- `test_irreps_debug.py`: Original debug script (needs updating)

## How to run

### Test gauge transformation (WORKING):
```bash
cd /Users/hexu/projects/phonproj/phonproj
uv run python agent_files/debug_irreps/test_gauge_transform.py
```

### Test irreps analysis (NOT YET WORKING):
```bash
cd /Users/hexu/projects/phonproj/phonproj  
uv run python agent_files/debug_irreps/test_irreps_debug.py
```

## Expected behavior

### test_acoustic_modes.py (WORKING ✓)
The script verifies:
1. Frequencies < 0.1 THz correctly return zero amplitude ✓
2. Frequencies ≥ 0.1 THz return non-zero amplitude ✓
3. Threshold logic prevents divergence ✓

### test_gauge_transform.py (WORKING ✓)
The script verifies:
1. R→r and r→R phase factors are complex conjugates ✓
2. Applying both transformations gives identity (R→r→R = I) ✓
3. At Gamma point, all phases equal 1 ✓
4. Sign convention in modes.py is mathematically correct ✓

### test_irreps_simple.py (PARTIAL)
Tests the refactored `_IrRepsLocal` implementation:
1. Shows the analysis runs without errors ✓
2. Demonstrates that random eigenvectors don't produce valid labels (expected)
3. Needs real phonon data to verify proper label extraction

### test_irreps_debug.py (NOT UPDATED)
Original debug script - needs updating to work with current code structure.
