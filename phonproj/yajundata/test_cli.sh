#phonproj-decompose -p ./0.02-P4mmm-PTO -s 16x1x1 -d ./supercell_undistorted.vasp  -i ./supercell_undistorted.vasp --no-sort --normalize  > result_undistorted.txt 
#phonproj-decompose -p ./0.02-P4mmm-PTO -s 16x1x1 -d ./newSM2.vasp  -i ./supercell_undistorted.vasp --no-sort --normalize  > result_sm2.txt 
#phonproj-decompose -p ./0.02-P4mmm-PTO -s 16x1x1 -d supercell_distorted.vasp -i ./supercell_undistorted.vasp   --no-sort  > result_full.txt 
phonproj-decompose -p ./0.02-P4mmm-PTO -s 16x1x1 -d  CONTCAR-a1a2-GS    --remove-com --normalize > result_full_sorted.txt 

