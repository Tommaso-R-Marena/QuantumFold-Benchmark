import pandas as pd

from qf_bench.metrics.statistics import compare_models


def test_compare_models_outputs_statistics():
    df = pd.DataFrame(
        [
            {"target_id": "T1", "model": "QuantumFold-Advantage", "rmsd": 2.0, "tm_score": 0.7, "plddt": 82.0},
            {"target_id": "T1", "model": "AlphaFold3", "rmsd": 1.6, "tm_score": 0.75, "plddt": 86.0},
            {"target_id": "T2", "model": "QuantumFold-Advantage", "rmsd": 3.0, "tm_score": 0.55, "plddt": 71.0},
            {"target_id": "T2", "model": "AlphaFold3", "rmsd": 2.8, "tm_score": 0.58, "plddt": 73.0},
            {"target_id": "T3", "model": "QuantumFold-Advantage", "rmsd": 1.8, "tm_score": 0.8, "plddt": 90.0},
            {"target_id": "T3", "model": "AlphaFold3", "rmsd": 1.5, "tm_score": 0.82, "plddt": 91.0},
        ]
    )

    stats_df = compare_models(df, baseline_model="QuantumFold-Advantage")

    assert not stats_df.empty
    assert set(["metric", "p_value_ttest", "p_value_wilcoxon", "cohens_d", "p_value_corrected"]).issubset(stats_df.columns)
    assert set(stats_df["challenger_model"].unique()) == {"AlphaFold3"}
