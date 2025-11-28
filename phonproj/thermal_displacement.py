import numpy as np
from scipy import constants


def convert_wavenumber_to_angular_frequency(wavenumber):
    """
    Convert wavenumber (cm^-1) to angular frequency (rad/s)
    """
    # Convert from cm^-1 to Hz (multiply by c in cm/s)
    frequency = wavenumber * constants.c * 100
    # Convert to angular frequency
    return 2 * np.pi * frequency


def bose_einstein_distribution(wavenumber, T, verbose=True):
    """
    Calculate Bose-Einstein distribution

    Args:
        omega: phonon frequency (in THz)
        T: temperature (in K)

    Returns:
        n: occupation number
    """
    if T == 0:
        return 0

    # Convert wavenumber to angular frequency (rad/s)
    omega = convert_wavenumber_to_angular_frequency(wavenumber)

    # k_B in J/K and ℏ in J⋅s
    kB = constants.Boltzmann
    hbar = constants.hbar

    # Calculate ℏω/kT
    x = hbar * omega / (kB * T)

    if verbose:
        print(f"ℏω/kT calculation:")
        print(f"  ω = {omega:.2e} rad/s")
        print(f"  T = {T} K")
        print(f"  ℏω/kT = {x:.2e}")
        print(f"  n(T) = {1 / (np.exp(x) - 1):.2e}")

    return 1 / (np.exp(x) - 1)


def calculate_displacement(mass, wavenumber, eigenvector, temperature, N=1, factor=1):
    """
    Calculate thermal displacement (not squared) for an atom

    Args:
        mass: atomic mass (in atomic mass units)
        wavenumber: phonon frequency (in cm^-1)
        eigenvector: complex polarization vector (3D array)
        temperature: temperature (in K)
        N: number of unit cells

    Returns:
        u: displacement for each cartesian direction in Angstroms
    """
    # Convert mass from amu to kg
    mass_kg = mass * constants.atomic_mass

    # ℏ in J⋅s = kg⋅m²/s
    hbar = constants.hbar

    # Calculate occupation number
    n = bose_einstein_distribution(wavenumber, temperature, verbose=False)

    # Convert wavenumber to angular frequency (rad/s)
    omega = convert_wavenumber_to_angular_frequency(wavenumber)

    # Calculate prefactor: ℏ/(2mω)
    # ℏ in kg⋅m²/s
    # mass_kg in kg
    # omega in rad/s = 1/s
    # Result will be in m²
    prefactor = np.sqrt(hbar / (2 * N * mass_kg * omega))

    # Calculate (1 + 2n) factor
    occupation_factor = 1 + 2 * n

    # Calculate displacement
    # The prefactor is already sqrt(ℏ/(2mω))
    # Apply sqrt(1+2n) to get the amplitude
    amplitude = prefactor * np.sqrt(occupation_factor)

    # Multiply by normalized eigenvector to get displacement in each direction
    # Apply the factor (1 or -1) to control displacement direction
    # Result is in meters, convert to Angstroms
    # Take the real part of the displacement
    u = np.real(amplitude * eigenvector) * 1e10 * factor

    return u


def calculate_thermal_displacement_squared(
    mass, wavenumber, eigenvector, temperature, N=1
):
    """
    Calculate mean square thermal displacement for an atom

    Args:
        mass: atomic mass (in atomic mass units)
        wavenumber: phonon frequency (in cm^-1)
        eigenvector: complex polarization vector (3D array)
        temperature: temperature (in K)
        N: number of unit cells

    Returns:
        u_squared: mean square displacement for each cartesian direction in Angstroms^2
    """
    u = calculate_displacement(mass, wavenumber, eigenvector, temperature, N)
    return u * u


def calculate_total_displacement(masses, wavenumbers, eigenvectors, temperature, N=1):
    """
    Calculate total thermal displacement for all atoms and modes

    Args:
        masses: array of atomic masses (in amu)
        frequencies: array of frequencies for each mode (in THz)
        eigenvectors: array of eigenvectors (shape: [n_modes, n_atoms, 3])
        temperature: temperature (in K)
        N: number of unit cells

    Returns:
        total_u_squared: total mean square displacement for each atom and direction
    """
    n_atoms = len(masses)
    # Initialize with real numbers
    total_u_squared = np.zeros((n_atoms, 3), dtype=float)

    for mode_idx, wavenumber in enumerate(wavenumbers):
        for atom_idx in range(n_atoms):
            u_squared = calculate_thermal_displacement_squared(
                masses[atom_idx],
                wavenumber,
                eigenvectors[mode_idx, atom_idx],
                temperature,
                N,
            )
            total_u_squared[atom_idx] += u_squared

    return total_u_squared


if __name__ == "__main__":
    # Example usage:
    # Define example inputs
    masses = np.array([150.36, 55.845, 15.999])  # Sm, Fe, O masses in amu
    wavenumbers = np.array([100.0, 200.0, 300.0])  # Example frequencies in cm^-1
    # Generate random complex eigenvectors
    eigenvectors = (
        np.random.random((3, 3, 3)) + 1j * np.random.random((3, 3, 3))
    ) / np.sqrt(2)
    temperature = 300  # Room temperature in K

    # Calculate displacements
    print("\nExample calculation at room temperature:")
    # Calculate mean square displacements
    displacements_squared = calculate_total_displacement(
        masses, wavenumbers, eigenvectors, temperature
    )

    # Calculate single phonon displacements
    single_phonon_displacements = np.zeros_like(displacements_squared, dtype=float)
    for mode_idx, wavenumber in enumerate(wavenumbers):
        for atom_idx in range(len(masses)):
            u = calculate_displacement(
                masses[atom_idx],
                wavenumber,
                eigenvectors[mode_idx, atom_idx],
                temperature,
                N=1,
            )
            single_phonon_displacements[atom_idx] += u

    print("Mean square displacements (Å²):")
    for i, atom in enumerate(["Sm", "Fe", "O"]):
        print(f"{atom}: {displacements_squared[i]}")

    print("\nSingle phonon displacements (Å):")
    for i, atom in enumerate(["Sm", "Fe", "O"]):
        print(f"{atom}: {single_phonon_displacements[i]}")
