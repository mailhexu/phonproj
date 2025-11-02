import numpy as np


def auto_kpath(cell, path="GMXMG", npoints=50):
    # Minimal stub: returns dummy k-path data
    # Replace with actual k-path logic as needed
    xlist = np.linspace(0, 1, npoints)
    kptlist = [np.zeros((npoints, 3))]
    Xs = None
    knames = ["G", "M", "X"]
    spk = {"G": np.zeros(3), "M": np.ones(3), "X": np.ones(3) * 0.5}
    return xlist, kptlist, Xs, knames, spk


def validate_kpath(path):
    return True


def suggest_path(cell):
    return "GMXMG"
