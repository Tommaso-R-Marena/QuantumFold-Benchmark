import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_benchmark_results(df: pd.DataFrame, output_dir: str):
    """
    Generate plots for benchmark results.

    Args:
        df: DataFrame containing benchmark results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. RMSD Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='dataset', y='rmsd', hue='model')
    plt.title('RMSD Comparison across Datasets')
    plt.ylabel('RMSD (Å)')
    plt.xlabel('Dataset')
    plt.savefig(os.path.join(output_dir, 'rmsd_comparison.png'))
    plt.close()

    # 2. TM-score Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='dataset', y='tm_score', hue='model')
    plt.title('TM-score Comparison across Datasets')
    plt.ylabel('TM-score')
    plt.xlabel('Dataset')
    plt.savefig(os.path.join(output_dir, 'tm_score_comparison.png'))
    plt.close()

    # 3. pLDDT Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='dataset', y='plddt', hue='model')
    plt.title('Average pLDDT Confidence by Model')
    plt.ylabel('pLDDT')
    plt.xlabel('Dataset')
    plt.savefig(os.path.join(output_dir, 'plddt_comparison.png'))
    plt.close()

def generate_markdown_report(df: pd.DataFrame, output_path: str):
    """
    Generate a markdown report from benchmark results.

    Args:
        df: DataFrame containing benchmark results.
        output_path: Path to save the markdown report.
    """
    summary = df.groupby(['dataset', 'model'])[['rmsd', 'tm_score', 'plddt']].mean().reset_index()

    with open(output_path, 'w') as f:
        f.write("# QuantumFold-Advantage Benchmark Report\n\n")
        f.write("## Executive Summary\n")
        f.write("This report presents a rigorous comparison of the hybrid quantum-classical folding model, **QuantumFold-Advantage**, against industry standards **AlphaFold3** and **Boltz2**.\n\n")

        f.write("## Mean Performance Metrics\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Visualizations\n")
        f.write("### RMSD Comparison\n")
        f.write("![RMSD Comparison](rmsd_comparison.png)\n\n")
        f.write("### TM-score Comparison\n")
        f.write("![TM-score Comparison](tm_score_comparison.png)\n\n")
        f.write("### pLDDT Confidence\n")
        f.write("![pLDDT Comparison](plddt_comparison.png)\n\n")

        f.write("## Methodology\n")
        f.write("- **Datasets**: CASP15, Miniproteins, and IDRs.\n")
        f.write("- **Metrics**: RMSD, TM-score, and pLDDT.\n")
        f.write("- **Quantum Engine**: Simulated VQE using Qiskit Aer.\n")
