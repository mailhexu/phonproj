# Symmetry Precision (symprec) Parameter

## Summary of Changes

Added a `symprec` parameter to control the symmetry precision when determining point groups and mode labels in `modes.py`.

## Updated Methods

### 1. `_determine_point_group(symprec=0.001)`
**Location**: `modes.py:778`

```python
def _determine_point_group(self, symprec: float = 0.001):
    """
    Determine point group using phonopy symmetry analysis.
    
    Args:
        symprec: Symmetry precision for finding point group (default: 0.001)
    
    Returns:
        str: Point group symbol
    """
```

- **Default value**: `0.001` (1e-3)
- Passed directly to `phonopy.structure.symmetry.Symmetry()`
- Controls how strictly symmetry operations are matched

### 2. `get_mode_summary_table(q_index=0, symprec=0.001)`
**Location**: `modes.py:825`

```python
def get_mode_summary_table(self, q_index: int = 0, symprec: float = 0.001):
    """
    Get a summary table of modes with frequencies and labels.
    
    Args:
        q_index: Index of q-point to analyze (default: 0)
        symprec: Symmetry precision for finding labels (default: 0.001)
    
    Returns:
        List of dictionaries with mode information
    """
```

- **Default value**: `0.001` (1e-3)
- Currently used for future irrep label determination
- Passes through to underlying symmetry analysis

### 3. `print_mode_summary_table(q_index=0, include_header=True, symprec=0.001)`
**Location**: `modes.py:892`

```python
def print_mode_summary_table(
    self, q_index: int = 0, 
    include_header: bool = True, 
    symprec: float = 0.001
) -> str:
    """
    Print a formatted table of modes with frequencies and labels.
    
    Args:
        q_index: Index of q-point to analyze (default: 0)
        include_header: Whether to include column header (default: True)
        symprec: Symmetry precision for finding labels (default: 0.001)
    
    Returns:
        Formatted table as a string
    """
```

- **Default value**: `0.001` (1e-3)
- Passes `symprec` to both:
  - `get_mode_summary_table(q_index, symprec=symprec)`
  - `_determine_point_group(symprec=symprec)`

## Usage Examples

### Default behavior (symprec = 0.001)
```python
# Uses default symprec=0.001
table = modes.print_mode_summary_table(q_index=0)
```

### Custom symmetry precision
```python
# Tighter tolerance (finds more symmetries)
table = modes.print_mode_summary_table(q_index=0, symprec=1e-5)

# Looser tolerance (finds fewer symmetries, more robust for noisy structures)
table = modes.print_mode_summary_table(q_index=0, symprec=0.01)
```

### In the example script
```python
# Can now specify symprec when calling
table_output = modes.print_mode_summary_table(
    q_index=0, 
    include_header=True,
    symprec=0.001  # Default, but can be customized
)
```

## When to Adjust symprec

### Use **tighter tolerance** (smaller symprec, e.g., 1e-5):
- High-quality, well-converged structures
- Need precise symmetry identification
- Academic/publication work requiring exact point groups

### Use **looser tolerance** (larger symprec, e.g., 0.01):
- Structures from molecular dynamics or relaxation
- Structures with thermal noise
- Quick exploratory analysis
- When symmetry detection is failing with default

### Keep **default** (0.001):
- General use cases
- Reasonable balance between precision and robustness
- Most phonopy calculations

## Rationale for Default Value

- **0.001 Å** is a common default in phonopy and other codes
- More robust than very tight tolerances (e.g., 1e-5)
- Still precise enough for most crystallographic applications
- Matches typical DFT relaxation convergence criteria

## Backward Compatibility

✅ **Fully backward compatible**
- All parameters have default values
- Existing code continues to work without changes
- Default behavior matches reasonable expectations
