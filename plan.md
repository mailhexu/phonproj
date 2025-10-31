# Project Implmentation rules
1. If there is something already implemented in the project, do not re-implement it.
2. Use phonopy API wherever possible, instead of re-inventing the wheel. 
3. Do real-world testing wherever possible, instead of mock testing. 
  There are two test cases
    - BaTiO3, in which the phonopy object can be loaded from, /Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml
    - PbTiO3, of which the phonopy object can be loaded from /Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO directory. 
4. The test should be in the tests directory, where each test file should be in one sub-directory.
5. The formulation should be in the docs/formulation.md file, with proper references to the equations in literature.
6. For each function or method, add an example in the examples directory. It should be minimal working examples, with proper docstring and comments.
7. the files in refs directory are reference implementations, which can be copied and modified as needed, they are not part of the project codebase.



# Steps:
0. Setup the project structure, with directories for src, tests, docs, examples,and python package initialization.
1. Implement loading of the phonopy files to get the phonopy object (with phonopy.load function).
  You can copy the file from refs/phonon_utils.py. Two versions, with either reading the yaml file, or the directory will be implemented, and tested.

2. Implement the calculation of the eigen vectors and eigen displacement for a list of q-points. Also implement the function to plot the band structure, with reference to refs/band_structure.py
Test:  plot the band structure of BaTiO3 and PbTiO3. 

3. Implement the eigenvectors for one q, and the function to test if the eigenvectors are orthonormal. 
Test: for a given q-point, check if the eigenvectors are orthonormal.

4. Implement the projection of the eigenvectors of one q-point with a arbitrary unit vector (with same size as the eigenvector). 
And test if the sum of the projections squared is 1. That is to test the completeness of the eigenvectors.

5. Implement the calculation of the eigen displacement with the phonopy get modulation and supercell method. Then
 Implement the function to compute the norm of eigendisplacement, by <u|M|u> = 1, where M is the masses for the atoms. 

6. Implement the function to compute the projections between two. Then with this, implement the method to check if the eigendisplacement is orthonormal with the mass-weighted inner product for a Gamma point. Check if the eigendisplacement is orthonormal with the mass-weighted inner product for the examples. 

7. In modes.py, the _calculate_supercell_displacements should be replaced by calling the phonopy API to get the modulation and the supercell. Implement this and test it. 
  Implement the function to generate the displacement of all the supercell displacements for a given q-point and all the modes.
  Implement a function to get all the commensurate q-points for a given supercell.
  Then implement the function to generate the list of displacement for all the commensurate q-points of a given supercell.
 List of tests:
    - Test for the Gamma q-point and the supercell of 1x1x1, if the displacement are orthonormal with mass-weighted inner product.
    - Test for the completeness of the eigendisplacement for Gamma q-point and supercell of 1x1x1, by first normalize the random unit displacement with the mass-weighted norm 1, then project it to all the eigendisplacement, and check if the sum of the projections squared is 1.
    - Test the implementation of generating the dispalcement for all commensurate q-points of a given supercell.
     - Test for a non-Gamma q-point and supercell of 2x2x2, if the displacement are orthogonal with mass-weighted inner product, and the mass-weighted norm of each displacement is 1.
    - Test the completeness of the eigendisplacement for all the commensurate q-point of 2x2x2 supercell, by first normalize the random unit displacement with the mass-weighted norm 1, then project it to all the eigendisplacement, and check if the sum of the projections squared is 1. 

