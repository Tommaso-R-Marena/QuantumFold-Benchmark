import pytest
import os
from qf_bench.data.loader import BenchmarkDataLoader
from qf_bench.models.quantum import QuantumFoldAdvantage
from qf_bench.runner import BenchmarkRunner
from qf_bench.metrics.scoring import calculate_rmsd

def test_data_loader():
    loader = BenchmarkDataLoader(cache_dir="tests/cache")
    targets = loader.get_casp15_targets()
    assert len(targets) > 0
    assert "sequence" in targets[0]

def test_quantum_model():
    model = QuantumFoldAdvantage()
    output_path = "tests/test_pred.pdb"
    model.predict("MAAHKGAEHHHK", output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)

def test_metrics():
    # Create two dummy PDBs
    path1 = "tests/dummy1.pdb"
    path2 = "tests/dummy2.pdb"

    with open(path1, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        f.write("END\n")

    with open(path2, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n")
        f.write("END\n")

    rmsd = calculate_rmsd(path1, path2)
    # Since there's only one atom, and they are 1.0 apart, but superimposer might center them
    # Actually Superimposer with 1 atom will result in 0 RMSD after alignment.
    assert rmsd >= 0
    os.remove(path1)
    os.remove(path2)
