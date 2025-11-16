#uv run python -m phonproj.map_structure  --help
uv run python -m phonproj.map_structure ref.vasp ref.vasp --optimize-shift --align-origin
uv run python -m phonproj.map_structure ref.vasp ref_shuffle.vasp --optimize-shift --align-origin
uv run python -m phonproj.map_structure ref.vasp supercell_undistorted.vasp --optimize-shift --align-origin
uv run python -m phonproj.map_structure ref.vasp SM2.vasp --optimize-shift --align-origin
