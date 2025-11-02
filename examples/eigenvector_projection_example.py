import numpy as np
from phonproj.modes import PhononModes
from ase import Atoms

# Dummy PhononModes setup
primitive_cell = Atoms(
    symbols=["H", "O"], positions=[[0, 0, 0], [0, 0, 1]], cell=[1, 1, 2]
)
qpoints = np.array([[0.0, 0.0, 0.0]])
frequencies = np.array([[1.0, 2.0]])
evecs = np.zeros((1, 2, 6), dtype=complex)
evecs[0, 0] = np.array([1, 0, 0, 0, 1, 0]) / np.sqrt(2)
evecs[0, 1] = np.array([0, 1, 0, 1, 0, 0]) / np.sqrt(2)
atomic_masses = np.array([1.0, 16.0])
modes = PhononModes(primitive_cell, qpoints, frequencies, evecs, atomic_masses)

q_index = 0
unit_vector = np.array([1, 0, 0, 0, 1, 0]) / np.linalg.norm([1, 0, 0, 0, 1, 0])

# Project eigenvectors
projections = modes.project_eigenvectors(q_index, unit_vector)
print("Projections onto unit vector:", projections)

# Verify completeness
is_complete, sum_sq = modes.verify_completeness(q_index, unit_vector)
print(f"Completeness: {is_complete}, Sum of squares: {sum_sq:.6f}")
