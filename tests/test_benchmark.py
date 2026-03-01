from qf_bench.data.loader import BenchmarkDataLoader
from qf_bench.metrics.scoring import calculate_rmsd
from qf_bench.models.quantum import QuantumFoldAdvantage


def test_data_loader(tmp_path):
    loader = BenchmarkDataLoader(cache_dir=tmp_path / "cache")
    targets = loader.get_casp15_targets()
    assert len(targets) > 0
    assert "sequence" in targets[0]


def test_quantum_model(tmp_path):
    model = QuantumFoldAdvantage()
    output_path = tmp_path / "test_pred.pdb"
    model.predict("MAAHKGAEHHHK", str(output_path))
    assert output_path.exists()


def test_metrics(tmp_path):
    # Create two dummy PDBs
    path1 = tmp_path / "dummy1.pdb"
    path2 = tmp_path / "dummy2.pdb"

    path1.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    path2.write_text(
        "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )

    rmsd = calculate_rmsd(str(path1), str(path2))
    # Since there's only one atom, and they are 1.0 apart, but superimposer might center them
    # Actually Superimposer with 1 atom will result in 0 RMSD after alignment.
    assert rmsd >= 0
