#!/usr/bin/env python3
"""
Debug script to understand why phase-scan gives small projections.
"""

import numpy as np
import sys

# Simple test
phases_rad = np.linspace(0, np.pi, 4, endpoint=True)
phases_deg = phases_rad * 180.0 / np.pi

print("Phases to scan:")
for i, (rad, deg) in enumerate(zip(phases_rad, phases_deg)):
    print(f"  {i}: {rad:.4f} rad = {deg:.1f}°")

# Check the conversion in generate_mode_displacement
print("\nAfter convert in generate_mode_displacement (phase = π * argument / 180):")
for deg in phases_deg:
    phase_final = np.pi * deg / 180
    print(
        f"  argument={deg:.1f}° → phase={phase_final:.4f} rad ({phase_final * 180 / np.pi:.1f}°)"
    )

print("\nThis shows the conversion is correct!")
print("Phase=0° stays at 0°, phase=180° becomes 180°")
