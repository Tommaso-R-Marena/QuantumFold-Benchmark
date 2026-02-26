import pytest
import numpy as np
import os
from qf_bench.metrics.scoring import calculate_rmsd, calculate_tm_score, calculate_plddt

def create_dummy_pdb(path, coords, bfactors=None):
    with open(path, "w") as f:
        for i, coord in enumerate(coords):
            bfactor = bfactors[i] if bfactors else 100.0
            f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{bfactor:6.2f}           C\n")
        f.write("END\n")

def test_rmsd_calculation():
    path1 = "tests/m1.pdb"
    path2 = "tests/m2.pdb"

    # 0 RMSD case
    coords = [[0, 0, 0], [3.8, 0, 0]]
    create_dummy_pdb(path1, coords)
    create_dummy_pdb(path2, coords)
    assert calculate_rmsd(path1, path2) == pytest.approx(0.0)

    # 1.0 RMSD case (shifted)
    coords2 = [[1, 0, 0], [4.8, 0, 0]]
    create_dummy_pdb(path2, coords2)
    # Since it's a pure translation, superimposer should make it 0
    assert calculate_rmsd(path1, path2) == pytest.approx(0.0, abs=1e-6)

    # Actual difference
    coords3 = [[0, 1, 0], [3.8, 0, 0]]
    create_dummy_pdb(path2, coords3)
    rmsd = calculate_rmsd(path1, path2)
    assert rmsd > 0

    os.remove(path1)
    os.remove(path2)

def test_tm_score_calculation():
    path1 = "tests/tm1.pdb"
    path2 = "tests/tm2.pdb"
    coords = [[float(i)*3.8, 0, 0] for i in range(20)]
    create_dummy_pdb(path1, coords)
    create_dummy_pdb(path2, coords)

    tm = calculate_tm_score(path1, path2)
    assert tm == pytest.approx(1.0)

    os.remove(path1)
    os.remove(path2)

def test_plddt_extraction():
    path = "tests/plddt.pdb"
    bfactors = [90.0, 80.0]
    coords = [[0,0,0], [3.8,0,0]]
    create_dummy_pdb(path, coords, bfactors)

    plddt = calculate_plddt(path)
    assert plddt == pytest.approx(85.0)
    os.remove(path)
