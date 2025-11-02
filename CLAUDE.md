<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Python Environment Management

**Always use `uv` to manage Python in this project.**

This project uses `uv` for Python environment and dependency management:
- Use `uv run python` instead of `python` for all script execution
- Use `uv add <package>` to install new dependencies
- Use `uv pip install -e .` for editable installs
- Use `uv sync` to sync dependencies from pyproject.toml

Example commands:
```bash
# Run Python scripts
uv run python examples/phonon_band_structure_simple_example.py

# Install new dependencies
uv add matplotlib numpy

# Install project in development mode
uv pip install -e .

# Run tests
uv run python -m pytest tests/
```

# References Directory

The `References/` directory contains reference code that should **NOT** be edited. These files can be read and copied from but must remain intact:

- `anamode.py` - analysis modes
- `cells.py` - crystallographic cell handling
- `domain.py` - domain analysis
- `frozen_mode.py` - frozen phonon modes
- `isodistort_parser.py` - ISODISTORT parsing functionality
- `perovskite_mode.py` - perovskite-specific mode analysis
- `phonmode.py` - phonon mode analysis
- `phonon_projection.py` - phonon projection techniques
- `pydistort.py` - structural distortion analysis
- `symbol.py` - symmetry symbols

These are reference implementations for phonon analysis algorithms and should be used as guidance when implementing PhononTools functionality.

# Code Optimization Guidelines

## Writing Simple, Efficient Code

### 1. Avoid Over-Engineering
- **Keep it simple**: Implement only what's requested, not what might be useful
- **Minimal solutions**: Use the simplest approach that works
- **Avoid premature optimization**: Don't optimize until performance is an issue
- **Single purpose**: Each function should do one thing well
- **YAGNI principle**: "You Aren't Gonna Need It" - don't build features not requested

### 2. Code Simplicity Rules
- **Avoid unnecessary abstractions**: Direct code is better than complex hierarchies
- **Use built-in functions**: Prefer Python/NumPy built-ins over custom implementations
- **Clear variable names**: Use descriptive names, but don't over-abbreviate
- **Avoid over-documentation**: Document complex logic, not obvious code
- **Prefer explicit over implicit**: Clear code is better than clever code

### 3. Error Handling Guidelines
- **Avoid excessive try/except**: Only use when error is expected and recoverable
- **Never wrap imports in try/except**: Let import errors surface immediately
- **Don't over-validate inputs**: Trust calling code, validate only when necessary
- **Meaningful error messages**: Explain what went wrong and how to fix it
- **Fail fast**: Let errors propagate when they indicate real problems

### 4. Additional Best Practices
- **Use existing implementations**: Leverage already-implemented functionality
- **Avoid redundant code**: Don't reimplement what already exists
- **Keep functions small**: Prefer 10-20 lines per function
- **Avoid deep nesting**: More than 3 levels of nesting is too complex
- **Test with real data**: Use actual files/data, not mock data
- **Prefer composition over inheritance**: Use composition when possible

### 5. Performance Considerations
- **Profile first**: Don't optimize without measuring
- **Use vectorization**: Prefer NumPy operations over Python loops
- **Memory efficient**: Avoid unnecessary data copying
- **Lazy evaluation**: Compute only when needed
- **Cache results**: Only when computation is expensive

### 6. When to Break These Rules
- **Complex domains**: Some problems inherently need more structure
- **Security concerns**: Add validation when handling external input
- **Production code**: More error handling for critical systems
- **User-facing APIs**: Better error messages and validation
- **Performance critical**: Optimize when profiling shows bottlenecks

### Examples

**Good (Simple):**
```python
# Direct approach using existing functionality
modes = PhononModes.from_phonopy_directory(directory, qpoints)
displacements = modes.generate_mode_displacement(0, 0, supercell_matrix)
```

**Bad (Over-engineered):**
```python
# Unnecessary abstraction
class PhononModeLoader:
    def __init__(self, directory, qpoints):
        self.directory = directory
        self.qpoints = qpoints

    def load_modes(self):
        try:
            # Excessive error handling
            if not os.path.exists(self.directory):
                raise DirectoryNotFoundError(f"Directory {self.directory} not found")
            # ... unnecessary complexity
        except Exception as e:
            logger.error(f"Error loading modes: {e}")
            raise
```

**Good (Minimal Error Handling):**
```python
# Only handle expected, recoverable errors
def load_data(filename):
    try:
        return np.loadtxt(filename)
    except FileNotFoundError:
        print(f"File {filename} not found, using default values")
        return default_values
```

**Bad (Excessive Error Handling):**
```python
# Don't wrap imports
try:
    import numpy as np
except ImportError as e:
    raise ImportError(f"NumPy is required: {e}")

# Don't over-validate obvious inputs
def add_numbers(a, b):
    if not isinstance(a, (int, float)):
        raise TypeError("a must be a number")
    if not isinstance(b, (int, float)):
        raise TypeError("b must be a number")
    return a + b
```

Remember: **Simple code is maintainable code**. Start simple and add complexity only when genuinely needed.