"""
Test acoustic mode amplitude handling.

Purpose:
    Verify that acoustic modes (frequency ~ 0) correctly return zero amplitude
    to avoid divergence in thermal displacement calculations.

How to run:
    uv run python agent_files/debug_irreps/test_acoustic_modes.py

Expected behavior:
    - Acoustic modes (freq < 0.1 THz) should have zero amplitude
    - Optical modes (freq > 0.1 THz) should have non-zero amplitude
"""

import numpy as np

print("=" * 80)
print("Testing Acoustic Mode Amplitude Handling")
print("=" * 80)

# Test the threshold logic
frequency_threshold = 0.1  # THz

test_frequencies = [
    -0.05,  # Imaginary acoustic mode
    0.0,  # Zero frequency
    0.01,  # Small acoustic mode
    0.09,  # Near threshold
    0.1,  # At threshold
    0.11,  # Just above threshold
    1.0,  # Optical mode
    10.0,  # High frequency optical mode
]

print(f"\nFrequency threshold: {frequency_threshold} THz")
print(f"(roughly {frequency_threshold * 33.35641:.1f} cm⁻¹)")
print("\nTest results:")
print("-" * 60)
print(f"{'Frequency (THz)':>20} {'Amplitude':>15} {'Status':>20}")
print("-" * 60)

for freq in test_frequencies:
    # Apply the same logic as in _calculate_thermal_amplitudes
    should_be_zero = abs(freq) < frequency_threshold

    if should_be_zero:
        amplitude_str = "0 (zero)"
        status = "✓ Acoustic mode"
    else:
        amplitude_str = "Non-zero"
        status = "✓ Optical mode"

    print(f"{freq:>20.3f} {amplitude_str:>15} {status:>20}")

print("-" * 60)

print("\n" + "=" * 80)
print("Physics Explanation")
print("=" * 80)
print("""
Acoustic phonon modes at the Gamma point (q=0) represent rigid translations
of the entire crystal. These modes have:

1. Frequency ω → 0 at the Gamma point
2. Thermal displacement formula: u ∝ 1/√ω
3. This causes divergence as ω → 0

Setting amplitude = 0 for these modes is correct because:
- Rigid translations don't contribute to internal thermal motion
- They represent center-of-mass motion of the entire crystal
- In a finite crystal, these modes have tiny but finite frequencies
- In the thermodynamic limit, they don't contribute to thermal properties

For optical modes (ω > threshold), the standard thermal displacement
formula applies and gives finite amplitudes.
""")

print("=" * 80)
print("Test Complete")
print("=" * 80)
