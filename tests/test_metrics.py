import numpy as np
import pytest
from qf_bench.metrics.scoring import (
    calculate_gdt_ts,
    calculate_plddt,
    calculate_rmsd,
    calculate_tm_score,
    get_common_ca_atoms,
)


def create_dummy_pdb(path, coords, bfactors=None, res_ids=None, chain_ids=None):
    with open(path, "w") as f:
        for i, coord in enumerate(coords):
            bfactor = bfactors[i] if bfactors else 100.0
            res_id = res_ids[i] if res_ids else i + 1
            chain_id = chain_ids[i] if chain_ids else "A"
            f.write(
                f"ATOM  {i+1:5d}  CA  ALA {chain_id}{res_id:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{bfactor:6.2f}           C\n"
            )
        f.write("END\n")


def test_rmsd_calculation(tmp_path):
    path1 = str(tmp_path / "m1.pdb")
    path2 = str(tmp_path / "m2.pdb")

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


def test_tm_score_calculation(tmp_path):
    path1 = str(tmp_path / "tm1.pdb")
    path2 = str(tmp_path / "tm2.pdb")
    coords = [[float(i) * 3.8, 0, 0] for i in range(20)]
    create_dummy_pdb(path1, coords)
    create_dummy_pdb(path2, coords)

    tm = calculate_tm_score(path1, path2)
    assert tm == pytest.approx(1.0)


def test_plddt_extraction(tmp_path):
    path = str(tmp_path / "plddt.pdb")
    bfactors = [90.0, 80.0]
    coords = [[0, 0, 0], [3.8, 0, 0]]
    create_dummy_pdb(path, coords, bfactors)

    plddt = calculate_plddt(path)
    assert plddt == pytest.approx(85.0)


def test_residue_matching(tmp_path):
    path1 = str(tmp_path / "match1.pdb")
    path2 = str(tmp_path / "match2.pdb")

    # path1 has residues 1, 2, 3
    coords1 = [[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]]
    create_dummy_pdb(path1, coords1, res_ids=[1, 2, 3])

    # path2 has residues 1, 3, 4 (missing 2, has extra 4)
    # but 1 and 3 are in the same relative position
    coords2 = [[0, 0, 0], [7.6, 0, 0], [11.4, 0, 0]]
    create_dummy_pdb(path2, coords2, res_ids=[1, 3, 4])

    ref_atoms, target_atoms = get_common_ca_atoms(path1, path2)

    assert len(ref_atoms) == 2
    assert len(target_atoms) == 2
    assert ref_atoms[0].get_parent().get_id()[1] == 1
    assert ref_atoms[1].get_parent().get_id()[1] == 3
    assert target_atoms[0].get_parent().get_id()[1] == 1
    assert target_atoms[1].get_parent().get_id()[1] == 3

    # RMSD should be 0 for residues 1 and 3
    assert calculate_rmsd(path1, path2) == pytest.approx(0.0)


def test_gdt_ts_calculation(tmp_path):
    path1 = str(tmp_path / "gdt1.pdb")
    path2 = str(tmp_path / "gdt2.pdb")

    # Identical structures
    coords = [[float(i) * 3.8, 0, 0] for i in range(10)]
    create_dummy_pdb(path1, coords)
    create_dummy_pdb(path2, coords)
    assert calculate_gdt_ts(path1, path2) == pytest.approx(1.0)

    # All atoms shifted by 5.0A
    # GDT-TS should have 100% for 8A, but 0% for 1A, 2A, 4A.
    # (GDT_1=0 + GDT_2=0 + GDT_4=0 + GDT_8=1) / 4 = 0.25
    coords_shifted = [[float(i) * 3.8 + 5.0, 0, 0] for i in range(10)]
    create_dummy_pdb(path2, coords_shifted)
    # After superimposition, it should still be 1.0 because it's a pure translation
    assert calculate_gdt_ts(path1, path2) == pytest.approx(1.0)

    # Add jitter
    rng = np.random.default_rng(42)
    coords_jitter = np.array(coords) + rng.uniform(-2, 2, size=(10, 3))
    create_dummy_pdb(path2, coords_jitter)
    gdt = calculate_gdt_ts(path1, path2)
    assert 0 < gdt < 1.0
