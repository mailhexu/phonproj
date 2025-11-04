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

8. Implement a projection of two displacements in two supercells. The supercells are with similar structure but not necessarily the same atom order, and there could be difference in the positions of atoms due to periodic boundary conditions. The function  should find the mapping between the two supercells, then do the projection with mass-weighted inner product of the corresponding displacements. An option is to mass-weighted normalize the displacements before doing the projection. You can refer to refs/structure_analysis.py. The projection can be done by find the mapping and map the displacements accordingly, then do the projection with mass-weighted inner product.
  The test should include:
    - two identical supercells and displacements. Test both normalized and unnormalized projection.
    - one supercell and displacement is a translated version of the other.
    - then one supercell with atoms shuffled, and the corresponding displacement shuffled accordingly.
    - finally, a combination of translation and shuffling.

9. Based on step 8, implement the projection of a displacment in a supercell projected to all the displacement due to phonon modes in all the commensurate q-points. 
   print the table of all the projections and its squared value.
   Tests:
    - test it with BaTiO3, with a random displacement in the supercell. 
    - random displacement + shuffling of atoms and displacements in the supercell .
    - test the sum of the squared projections is 1, for normalized random displacement + shuffling.

10. Add an example based on Step 8.  First generate the supercell from the /Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO directory, get all the commensurart q-point of a (16,1,1) supercell, and the corresponding displacment, and the supercell without displacement. Compute the displace from the displaced structure is in the file /Users/hexu/projects/phonproj/data/yajundata/CONTCAR-a1a2-GS, and the supercell before displacement is a 16x1x1. Note that the displaced structure is not necessarily in the same order as the supercell generated from phonopy, so shuffling is needed, and there could be atoms crossing the periodic boundary. Implement a function to handel the case ( a function to find the mapping, reorder the atoms to the reference supercell, and apply periodic boundary condition to make sure the positions are close to the reference supercell). Then do the projection to all the eigendisplacement of all the commensurate q-points, and print the table of projections and squared projections. 
  

11. in @cli.py, allow passing a isodistort file as a input file instead of the displaced structure.
  - parse it to get the undistorted structure and the distorted structure with the isodistort parser.
  - then compute the displacement using the two structures.
  - then compute the projection with the displacement. It stll has to be mapped to the supercell generated from phonopy, but here the mapping should be using the undistorted structure to the phonopy supercell. 
  - test it like in step 10, but using an isodistort file P4mmm-ref.txt instead of the displaced structure file.


12. implement a function to compute the distance of two positions within a cell considering the pbc, and minimize the distance by pbc. You can use the function in ASE. 2. find the atom closest to the origin (0,0,0), and shift both structures so that that atom is at the origin. 3. Find the mapping of the two structures and shift, so that the positions of disp[mapping] + shift[mapping] can make the distance (without considering pbc) are closest. Note that the mapping should not make a map between two different species. 4. output the details in the procedure of the mapping into txt file saved in data/mapping directory. The details should include the mapping index, the shift vector for each atom, and the final distance after mapping and shifting, output it as a table. 
Tests (data in data/yajundata/):
 - the ref.vasp, ref.vasp + random shuffle of atoms
 - the ref.vasp, ref.vasp + random shuffle + random translation of atoms by 1 in scaled_positions.
 - the ref.vasp, ref.vasp + random shuffle + random translation of atoms by 1 in scaled_positions + uniform displacement of all atoms by a small random vector (0.1 A)
 - the ref.vasp, SM2.vasp
 - the ref.vasp, supercell_undistorted.vasp


