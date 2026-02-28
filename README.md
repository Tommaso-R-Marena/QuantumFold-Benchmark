# QuantumFold-Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Benchmark/blob/main/notebooks/QuantumFold_Benchmarking.ipynb)

Production-ready benchmarking suite for **QuantumFold-Advantage**: Rigorous comparison of hybrid quantum-classical protein folding against **AlphaFold3**, **Boltz2**, and other classical baselines.

## Overview

QuantumFold-Advantage leverages hybrid quantum-classical algorithms (e.g., VQE) to sample protein conformations more efficiently than traditional classical methods. This benchmarking suite provides a standardized way to evaluate its performance against state-of-the-art models on key biological datasets.

### Key Features
- **Datasets**: Automatic fetching and processing of CASP15 targets, curated miniproteins, and Intrinsically Disordered Regions (IDRs).
- **Models**:
  - `QuantumFold-Advantage`: Hybrid quantum-classical solver (simulated via Qiskit).
  - `AlphaFold3`: Proprietary industry leader.
  - `Boltz2`: Open-source high-performance alternative.
- **Metrics**: Production-grade structural metrics including RMSD, TM-score, GDT-TS, and pLDDT.
- **Reporting**: Automated generation of markdown reports and Seaborn-based visualizations.
- **Colab Ready**: Designed to run seamlessly in Google Colab for democratization of quantum research.

## Installation

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## Usage

Run the full benchmark suite:

```bash
python scripts/run_benchmarks.py
```

Results will be saved in the `results/` directory, including:
- `benchmark_results.csv`: Raw data.
- `report.md`: Summary report.
- `rmsd_comparison.png`, `tm_score_comparison.png`, `gdt_ts_comparison.png`, `plddt_comparison.png`, `plddt_distribution.png`: Visualizations.

## Advanced Usage

### Adding Custom Models
To add a new model, create a new class in `src/qf_bench/models/` that inherits from `FoldingModel` and implements the `predict` method and `name` property.

```python
from qf_bench.models.base import FoldingModel

class MyNewModel(FoldingModel):
    @property
    def name(self) -> str:
        return "MyNewModel"

    def predict(self, sequence: str, output_path: str) -> str:
        # Implementation here
        return output_path
```

### Adding Custom Datasets
Update `src/qf_bench/data/targets.json` with your new dataset name and a list of target IDs and sequences.

## Project Structure

- `src/qf_bench/`: Core library.
  - `data/`: Data loading and PDB management.
  - `models/`: Model wrappers (Quantum, AF3, Boltz2).
  - `metrics/`: Structural scoring logic.
  - `utils/`: Visualization and reporting tools.
- `scripts/`: CLI entry points.
- `tests/`: Comprehensive test suite.
- `notebooks/`: Colab-compatible demonstration notebooks.

## Testing

Run tests with pytest:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m pytest tests/
```

## License
Apache 2.0
