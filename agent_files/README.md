# Agent Files Directory

This directory contains debugging, testing, and exploration files created by AI agents working on this project.

## Directory Structure

```
agent_files/
├── debug_[topic]/          # One directory per debugging session
│   ├── README.md          # Purpose and file descriptions for this session
│   └── *.py               # Debug scripts with descriptive docstrings
└── README.md              # This file - master index
```

## Usage Guidelines

### For Agents
1. **Create a new directory** for each debugging session: `debug_[topic]`
2. **Always include a README.md** in your debug directory explaining:
   - What problem you're investigating
   - What each file does
   - How to run the scripts
   - Expected results

3. **Every Python file must have a comprehensive docstring** explaining:
   - Purpose of the script
   - What it tests or demonstrates
   - How to run it
   - Expected output or behavior
   - Related files it works with

### For Humans
- Each `debug_[topic]` directory is self-contained
- Start by reading the README.md in the relevant directory
- Scripts are designed to be run with `uv run python`
- Most scripts use real data from the `data/` directory

## Current Debug Sessions

*(This section will be updated as agents create new debug directories)*

### debug_example
- **Purpose**: Example template for future debug sessions
- **Files**: 
  - `debug_example.py`: Template script showing proper docstring format
- **Usage**: `uv run python agent_files/debug_example/debug_example.py`
- **Status**: Template example - demonstrates proper structure and documentation

## File Naming Conventions

- **Directories**: `debug_[specific_topic]` (kebab-case)
- **Python files**: `debug_[specific_issue].py` or `test_[feature].py`
- **README files**: Always `README.md`

## Important Notes

- **Do not modify files in other directories** unless specifically instructed
- **Use existing project functionality** - don't reimplement what's already available
- **Keep scripts simple and focused** - one main purpose per file
- **Document assumptions and limitations** in your README files
- **Clean up after yourself** - remove temporary files and unnecessary outputs

## Related Directories

- `../debugs/` - Legacy debug files (preserve existing content)
- `../examples/` - Educational examples and usage patterns
- `../tests/` - Formal test suite
- `../refs/` - Reference implementations (read-only)

---

For more detailed guidelines, see the main `../AGENTS.md` file.