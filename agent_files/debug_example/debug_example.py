"""
Debug example script demonstrating proper documentation format.

PURPOSE:
    This script serves as a template for debug scripts, showing the proper
    docstring format and structure that all debug scripts should follow.

USAGE:
    uv run python agent_files/debug_example/debug_example.py

EXPECTED OUTPUT:
    Prints example debug messages demonstrating the format:
    - Starting debug session message
    - Example investigation steps
    - Completion message

FILES USED:
    - This script is self-contained and doesn't require external files
    - Real debug scripts would list related files here

DEBUG NOTES:
    This is a template example. When creating real debug scripts:
    1. Replace this content with your actual debugging code
    2. Use real data from the data/ directory when applicable
    3. Include meaningful debug output and logging
    4. Document any findings or conclusions in the script comments
    5. Update this docstring to reflect the actual purpose
"""

import numpy as np


def main():
    """Main debug function demonstrating proper structure."""
    print("=== Debug Example Session ===")
    print("Purpose: Demonstrating proper debug script format")
    print()

    # Example investigation step
    print("Step 1: Checking environment...")
    print(f"NumPy version: {np.__version__}")
    print("✓ Environment check passed")
    print()

    # Example data investigation
    print("Step 2: Example data analysis...")
    sample_data = np.array([1, 2, 3, 4, 5])
    print(f"Sample data: {sample_data}")
    print(f"Mean: {np.mean(sample_data)}")
    print(f"Std: {np.std(sample_data)}")
    print("✓ Data analysis completed")
    print()

    # Example conclusion
    print("Step 3: Debug conclusions...")
    print("- Environment is properly configured")
    print("- Data analysis functions work correctly")
    print("- Template structure is valid")
    print()

    print("=== Debug Session Complete ===")
    print("This template can be used for future debug sessions")


if __name__ == "__main__":
    main()
