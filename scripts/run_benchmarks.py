import os
import pandas as pd
from qf_bench.data.loader import BenchmarkDataLoader
from qf_bench.models.quantum import QuantumFoldAdvantage
from qf_bench.models.classical import AlphaFold3Wrapper, Boltz2Wrapper
from qf_bench.runner import BenchmarkRunner
from qf_bench.utils.visualizer import plot_benchmark_results, generate_markdown_report

def main():
    loader = BenchmarkDataLoader()

    # Initialize models
    # Note: In a real production scenario, api_keys would be loaded from env vars
    models = [
        QuantumFoldAdvantage(),
        AlphaFold3Wrapper(api_token=os.getenv("AF3_TOKEN")),
        Boltz2Wrapper()
    ]

    output_dir = "results"
    runner = BenchmarkRunner(models, loader, output_dir=output_dir)

    all_results = []

    # CASP15
    print("\n--- CASP15 Benchmark ---")
    casp15_targets = loader.get_casp15_targets()
    res_casp = runner.run_benchmark("CASP15", casp15_targets)
    all_results.append(res_casp)

    # Miniproteins
    print("\n--- Miniproteins Benchmark ---")
    mini_targets = loader.get_miniproteins()
    res_mini = runner.run_benchmark("Miniproteins", mini_targets)
    all_results.append(res_mini)

    # IDRs
    print("\n--- IDR Benchmark ---")
    idr_targets = loader.get_idrs()
    res_idr = runner.run_benchmark("IDRs", idr_targets)
    all_results.append(res_idr)

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)
    runner.save_results(final_df, "benchmark_results.csv")

    # Generate Visualizations
    print("\nGenerating visualizations...")
    plot_benchmark_results(final_df, output_dir)

    # Generate Report
    print("Generating markdown report...")
    generate_markdown_report(final_df, os.path.join(output_dir, "report.md"))

    print("\nBenchmark Suite execution complete. See results/ directory for details.")

if __name__ == "__main__":
    main()
