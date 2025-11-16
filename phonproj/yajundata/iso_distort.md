In the isodistort output, there are such blocks:

# Parent structure:
```
Parent structure (1 P1)
a=5.59234, b=5.59234, c=3.89428, alpha=90.00000, beta=90.00000, gamma=90.00000
atom site    x         y         z         occ
Pb1  1a     0.00000   0.00000   0.00000   1.00000
Pb2  1a     0.50000   0.50000   0.00000   1.00000
Ti1  1a     0.50000   0.00000   0.50000   1.00000
Ti2  1a     0.00000   0.50000   0.50000   1.00000
O1   1a     0.25000   0.25000   0.50000   1.00000
O2   1a    -0.25000  -0.25000   0.50000   1.00000
O3   1a     0.25000  -0.25000   0.50000   1.00000
O4   1a    -0.25000   0.25000   0.50000   1.00000
O5   1a     0.50000   0.00000   0.00000   1.00000
O6   1a     0.00000   0.50000   0.00000   1.00000
```

The a, b, c, alpha, beta, gamma gives the lattice parameters of the parent structure.
Pb1 is the 1st Pb atom in the parent structure, Pb2 is the 2nd Pb atom, and so on.
1a indicates the Wyckoff position.
x, y, z are the fractional coordinates of the atom in the unit cell.
occ is the occupancy of the atom.



# undistorted superstructure:
```
Undistorted superstructure
a=89.47749, b=5.59234, c=3.89428, alpha=90.00000, beta=90.00000, gamma=90.00000
atom site    x         y         z         occ      displ
Pb7  1a     0.18701   0.00000   0.01301   1.00000   0.00000
Pb9  1a     0.24951   0.00000   0.01301   1.00000   0.00000
....

```
The format is the same as the parent structure, but now there is an additional column "displ" at the end.

# distorted superstructure:
```
Distorted superstructure
a=89.47749, b=5.59234, c=3.91655, alpha=90.00000, beta=90.00000, gamma=90.00000
atom site    x         y         z         occ      displ
Pb7  1a     0.18703  -0.00372   0.01301   1.00000   0.02087
Pb9  1a     0.24959   0.03960   0.01301   1.00000   0.22157
....

```
The format is the same as the undistorted superstructure.



