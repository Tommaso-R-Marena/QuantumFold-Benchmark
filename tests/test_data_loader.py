from qf_bench.data.loader import BenchmarkDataLoader


def test_data_loader_targets():
    loader = BenchmarkDataLoader()
    casp = loader.get_casp15_targets()
    mini = loader.get_miniproteins()
    idr = loader.get_idrs()

    assert len(casp) == 3
    assert len(mini) == 3
    assert len(idr) == 2

    for t in casp + mini + idr:
        assert "id" in t
        assert "sequence" in t


def test_download_pdb(tmp_path):
    # Use tmp_path fixture for cache
    cache_dir = tmp_path / "cache"
    loader = BenchmarkDataLoader(cache_dir=cache_dir)

    # Test downloading a real PDB (small one)
    path = loader.download_pdb("1A2P")
    assert (tmp_path / "cache" / "1A2P.pdb").exists()
    assert (tmp_path / "cache" / "1A2P.pdb").stat().st_size > 0


def test_dummy_pdb_creation(tmp_path):
    cache_dir = tmp_path / "cache"
    loader = BenchmarkDataLoader(cache_dir=cache_dir)

    # Test creating a dummy for non-existent PDB
    path = loader.download_pdb("XXXX")
    assert (tmp_path / "cache" / "XXXX.pdb").exists()
    with open(path, "r") as f:
        content = f.read()
        assert "ATOM" in content
