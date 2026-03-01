import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_benchmark_results(df: pd.DataFrame, output_dir: str | Path):
    """
    Generate plots for benchmark results.

    Args:
        df: DataFrame containing benchmark results.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. RMSD Boxplot with Swarm
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="dataset", y="rmsd", hue="model")
    sns.stripplot(
        data=df, x="dataset", y="rmsd", hue="model", dodge=True, alpha=0.5, palette="dark:black"
    )
    plt.title("RMSD Comparison across Datasets")
    plt.ylabel("RMSD (Å)")
    plt.xlabel("Dataset")
    plt.savefig(output_dir / "rmsd_comparison.png")
    plt.close()

    # 2. TM-score Boxplot with Swarm
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="dataset", y="tm_score", hue="model")
    sns.stripplot(
        data=df, x="dataset", y="tm_score", hue="model", dodge=True, alpha=0.5, palette="dark:black"
    )
    plt.title("TM-score Comparison across Datasets")
    plt.ylabel("TM-score")
    plt.xlabel("Dataset")
    plt.savefig(output_dir / "tm_score_comparison.png")
    plt.close()

    # 3. GDT-TS Comparison
    if "gdt_ts" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="dataset", y="gdt_ts", hue="model")
        plt.title("GDT-TS Comparison across Datasets")
        plt.ylabel("GDT-TS")
        plt.xlabel("Dataset")
        plt.savefig(output_dir / "gdt_ts_comparison.png")
        plt.close()

    # 4. pLDDT Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="dataset", y="plddt", hue="model", errorbar="sd", palette="viridis")
    plt.title("Average pLDDT Confidence by Model (with Std Dev)")
    plt.ylabel("pLDDT")
    plt.xlabel("Dataset")
    plt.savefig(output_dir / "plddt_comparison.png", dpi=300)
    plt.close()

    # 6. Execution Time Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="dataset", y="execution_time", hue="model", palette="magma")
    plt.title("Average Execution Time by Model")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Dataset")
    plt.savefig(output_dir / "execution_time_comparison.png", dpi=300)
    plt.close()

    # 5. pLDDT Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="plddt", hue="model", kde=True, element="step")
    plt.title("pLDDT Score Distribution")
    plt.xlabel("pLDDT")
    plt.savefig(output_dir / "plddt_distribution.png")
    plt.close()


def generate_markdown_report(
    df: pd.DataFrame,
    output_path: str | Path,
    statistical_report: Optional[pd.DataFrame] = None,
):
    """
    Generate a markdown report from benchmark results.

    Args:
        df: DataFrame containing benchmark results.
        output_path: Path to save the markdown report.
        statistical_report: Optional DataFrame containing paired statistical tests.
    """
    metrics = ["rmsd", "tm_score", "plddt"]
    if "gdt_ts" in df.columns:
        metrics.append("gdt_ts")

    summary = (
        df.groupby(["dataset", "model"])[metrics + ["execution_time"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Flatten multi-index columns
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write("# QuantumFold-Advantage Benchmark Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n")
        f.write(
            "This report presents a rigorous comparison of the hybrid quantum-classical folding model, **QuantumFold-Advantage**, against industry standards **AlphaFold3** and **Boltz2**.\n\n"
        )

        f.write("## Mean Performance Metrics\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")

        if statistical_report is not None and not statistical_report.empty:
            f.write("## Statistical Comparison vs Baseline\n\n")
            f.write(
                "Paired statistical tests (Wilcoxon) and effect size (Cohen's d) estimation.\n\n"
            )
            f.write(statistical_report.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Detailed Results per Target\n\n")
        display_cols = ["dataset", "target_id", "model"] + metrics + ["execution_time"]
        detailed = df[display_cols].sort_values(["dataset", "target_id"])
        f.write(detailed.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Visualizations\n")
        f.write("### RMSD Comparison\n")
        f.write("![RMSD Comparison](rmsd_comparison.png)\n\n")
        f.write("### TM-score Comparison\n")
        f.write("![TM-score Comparison](tm_score_comparison.png)\n\n")

        if "gdt_ts" in df.columns:
            f.write("### GDT-TS Comparison\n")
            f.write("![GDT-TS Comparison](gdt_ts_comparison.png)\n\n")

        f.write("### pLDDT Confidence\n")
        f.write("![pLDDT Comparison](plddt_comparison.png)\n\n")
        f.write("### pLDDT Distribution\n")
        f.write("![pLDDT Distribution](plddt_distribution.png)\n\n")

        f.write("## Methodology\n")
        f.write("- **Datasets**: CASP15, Miniproteins, and IDRs.\n")
        f.write("- **Metrics**: RMSD, TM-score, GDT-TS, and pLDDT.\n")
        f.write("- **Quantum Engine**: Simulated VQE using Qiskit Aer.\n")
