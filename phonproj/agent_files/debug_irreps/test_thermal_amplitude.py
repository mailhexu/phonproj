"""
Test thermal amplitude dependence on frequency.

Purpose:
    Verify that thermal displacement amplitude shows the expected strong
    frequency dependence: low-frequency modes should have MUCH larger
    amplitudes than high-frequency modes.

How to run:
    uv run python agent_files/debug_irreps/test_thermal_amplitude.py

Expected behavior:
    At T=200K, comparing modes at different frequencies:
    - Low freq (50 cm⁻¹, ~1.5 THz): n(T) ~ 10-20, amplitude ~ 0.1-0.3 Å
    - High freq (500 cm⁻¹, ~15 THz): n(T) ~ 0.1, amplitude ~ 0.01-0.03 Å

    Amplitude ratio should be ~10x or more.
"""

import numpy as np
from scipy import constants


def convert_wavenumber_to_angular_frequency(wavenumber):
    """Convert wavenumber (cm^-1) to angular frequency (rad/s)"""
    frequency = wavenumber * constants.c * 100
    return 2 * np.pi * frequency


def bose_einstein_distribution(wavenumber, T):
    """Calculate Bose-Einstein occupation number"""
    if T == 0:
        return 0
    omega = convert_wavenumber_to_angular_frequency(wavenumber)
    kB = constants.Boltzmann
    hbar = constants.hbar
    x = hbar * omega / (kB * T)

    # Avoid overflow for very large x
    if x > 100:
        return 0.0

    return 1 / (np.exp(x) - 1)


def calculate_amplitude(mass, wavenumber, temperature):
    """
    Calculate thermal displacement amplitude for given frequency and temperature.

    Formula: u = sqrt(ℏ/(2mω)) * sqrt(1 + 2n(T))

    where n(T) = 1/(exp(ℏω/kT) - 1) is Bose-Einstein occupation
    """
    mass_kg = mass * constants.atomic_mass
    hbar = constants.hbar
    omega = convert_wavenumber_to_angular_frequency(wavenumber)

    # Calculate occupation number
    n = bose_einstein_distribution(wavenumber, temperature)

    # Calculate prefactor: sqrt(ℏ/(2mω))
    prefactor = np.sqrt(hbar / (2 * mass_kg * omega))

    # Calculate (1 + 2n) factor
    occupation_factor = 1 + 2 * n

    # Total amplitude (in meters)
    amplitude = prefactor * np.sqrt(occupation_factor)

    # Convert to Angstroms
    return amplitude * 1e10, n


print("=" * 80)
print("Thermal Displacement Amplitude vs Frequency")
print("=" * 80)

# Test parameters
temperature = 200.0  # K
mass = 50.0  # amu (typical atom)

# Test frequencies (in cm^-1)
test_wavenumbers = [
    10.0,  # Very low (acoustic-like)
    50.0,  # Low optical
    100.0,  # Medium-low
    200.0,  # Medium
    300.0,  # Medium-high
    400.0,  # High
    500.0,  # Very high
    1000.0,  # Extremely high
]

print(f"\nTemperature: {temperature} K")
print(f"Mass: {mass} amu")
print("\n" + "-" * 80)
print(
    f"{'Wavenumber':>12} {'Freq (THz)':>12} {'n(T)':>12} {'Amplitude':>15} {'√(1+2n)':>12}"
)
print(f"{'(cm⁻¹)':>12} {'':>12} {'':>12} {'(Å)':>15} {'':>12}")
print("-" * 80)

amplitudes = []
frequencies_thz = []

for wn in test_wavenumbers:
    freq_thz = wn / 33.35641  # Convert cm^-1 to THz
    amplitude, n = calculate_amplitude(mass, wn, temperature)
    occupation_contribution = np.sqrt(1 + 2 * n)

    amplitudes.append(amplitude)
    frequencies_thz.append(freq_thz)

    print(
        f"{wn:>12.1f} {freq_thz:>12.3f} {n:>12.2f} {amplitude:>15.6f} {occupation_contribution:>12.2f}"
    )

print("-" * 80)

# Analyze the ratios
print("\n" + "=" * 80)
print("Analysis: Amplitude Ratios")
print("=" * 80)

lowest_amp = amplitudes[0]
print(
    f"\nLowest frequency ({test_wavenumbers[0]:.0f} cm⁻¹): amplitude = {lowest_amp:.6f} Å"
)

for i in range(1, len(amplitudes)):
    ratio = lowest_amp / amplitudes[i]
    print(f"  Ratio to {test_wavenumbers[i]:>6.0f} cm⁻¹: {ratio:>6.2f}x larger")

print("\n" + "=" * 80)
print("Expected Behavior")
print("=" * 80)
print("""
For T=200K:
1. Low frequency modes (10-50 cm⁻¹) have high occupation n(T) ~ 10-50
   - Large prefactor 1/√ω
   - Large occupation factor √(1+2n) ~ 3-10
   - Result: amplitude ~ 0.1-0.5 Å

2. High frequency modes (500-1000 cm⁻¹) have low occupation n(T) ~ 0.01-0.1
   - Small prefactor 1/√ω
   - Small occupation factor √(1+2n) ~ 1.1-1.4
   - Result: amplitude ~ 0.01-0.03 Å

Expected ratio: 10-50x difference between low and high frequency modes.

If you're seeing similar amplitudes for all modes, there may be an issue with:
- Eigenvector normalization
- Temperature units
- Frequency units in the calculation
""")

# Plot if matplotlib available
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Amplitude vs frequency
    ax1.plot(frequencies_thz, amplitudes, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Amplitude (Å)", fontsize=12)
    ax1.set_title(f"Thermal Displacement at T={temperature}K", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Occupation number
    occupations = [
        bose_einstein_distribution(wn, temperature) for wn in test_wavenumbers
    ]
    ax2.plot(frequencies_thz, occupations, "s-", linewidth=2, markersize=8, color="red")
    ax2.set_xlabel("Frequency (THz)", fontsize=12)
    ax2.set_ylabel("Occupation n(T)", fontsize=12)
    ax2.set_title(f"Bose-Einstein Occupation at T={temperature}K", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("agent_files/debug_irreps/thermal_amplitude_analysis.png", dpi=150)
    print(f"\n✓ Plot saved to: agent_files/debug_irreps/thermal_amplitude_analysis.png")

except ImportError:
    print("\n(matplotlib not available - skipping plot)")
