from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


@dataclass
class PairedComparison:
    baseline_model: str
    challenger_model: str
    metric: str
    n_pairs: int
    mean_difference: float
    ci95_low: float
    ci95_high: float
    p_value_ttest: float
    p_value_wilcoxon: float
    cohens_d: float


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    samples = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _cohens_d(values: np.ndarray) -> float:
    std = np.std(values, ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(values) / std)


def compare_models(
    df: pd.DataFrame,
    baseline_model: str,
    metrics: List[str] | None = None,
    correction: str = "benjamini-hochberg",
) -> pd.DataFrame:
    """Run paired significance tests and effect size estimates vs baseline model."""
    if metrics is None:
        metrics = ["rmsd", "tm_score", "plddt"]

    results: List[PairedComparison] = []
    models = sorted(m for m in df["model"].unique() if m != baseline_model)

    for challenger in models:
        merged = df[df["model"].isin([baseline_model, challenger])]
        pivot = merged.pivot_table(index="target_id", columns="model", values=metrics)
        for metric in metrics:
            if metric not in pivot:
                continue
            paired = pivot[metric].dropna()
            if baseline_model not in paired.columns or challenger not in paired.columns:
                continue
            diffs = (paired[challenger] - paired[baseline_model]).to_numpy(dtype=float)
            if len(diffs) < 2:
                continue

            t_p = float(ttest_rel(paired[challenger], paired[baseline_model], nan_policy="omit").pvalue)
            try:
                w_p = float(wilcoxon(diffs).pvalue)
            except ValueError:
                w_p = 1.0
            ci_low, ci_high = _bootstrap_ci(diffs)

            results.append(
                PairedComparison(
                    baseline_model=baseline_model,
                    challenger_model=challenger,
                    metric=metric,
                    n_pairs=int(len(diffs)),
                    mean_difference=float(np.mean(diffs)),
                    ci95_low=ci_low,
                    ci95_high=ci_high,
                    p_value_ttest=t_p,
                    p_value_wilcoxon=w_p,
                    cohens_d=_cohens_d(diffs),
                )
            )

    out = pd.DataFrame([r.__dict__ for r in results])
    if out.empty:
        return out

    if correction.lower() == "bonferroni":
        m = len(out)
        out["p_value_corrected"] = np.minimum(out["p_value_wilcoxon"] * m, 1.0)
    else:
        order = np.argsort(out["p_value_wilcoxon"].to_numpy())
        ranked = out.iloc[order].copy()
        m = len(ranked)
        ranked["p_value_corrected"] = ranked["p_value_wilcoxon"] * m / (np.arange(m) + 1)
        ranked["p_value_corrected"] = np.minimum.accumulate(ranked["p_value_corrected"][::-1])[::-1]
        ranked["p_value_corrected"] = np.clip(ranked["p_value_corrected"], 0, 1)
        out = ranked.sort_index()

    return out
