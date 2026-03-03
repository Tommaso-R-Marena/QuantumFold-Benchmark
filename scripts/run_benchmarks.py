import os
import pandas as pd
import argparse
import logging
from qf_bench.data.loader import BenchmarkDataLoader
from qf_bench.models.quantum import QuantumFoldAdvantage
from qf_bench.models.classical import AlphaFold3Wrapper, Boltz2Wrapper
from qf_bench.runner import BenchmarkRunner
from qf_bench.utils.visualizer import plot_benchmark_results, generate_markdown_report

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="QuantumFold-Advantage Benchmarking Suite")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--datasets", nargs="+", choices=["casp15", "miniproteins", "idrs"],
                        default=["casp15", "miniproteins", "idrs"], help="Datasets to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--baseline", type=str, default=None, help="Model name to use as baseline for stats")

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    loader = BenchmarkDataLoader()

    # Initialize models
    models = [
        QuantumFoldAdvantage(seed=args.seed),
        AlphaFold3Wrapper(api_token=os.getenv("AF3_TOKEN")),
        Boltz2Wrapper()
    ]

    runner = BenchmarkRunner(models, loader, output_dir=args.output_dir)

    all_results = []

    dataset_map = {
        "casp15": ("CASP15", loader.get_casp15_targets),
        "miniproteins": ("Miniproteins", loader.get_miniproteins),
        "idrs": ("IDRs", loader.get_idrs),
    }

    for ds_key in args.datasets:
        name, fetcher = dataset_map[ds_key]
        logger.info(f"Running {name} Benchmark...")
        targets = fetcher()
        if targets:
            res = runner.run_benchmark(
                name, targets, max_workers=args.max_workers, baseline_model=args.baseline
            )
            all_results.append(res)

    if not all_results:
        logger.warning("No datasets selected to run.")
        return

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)
    runner.save_results(final_df, "benchmark_results.csv")

    # Generate Visualizations
    logger.info("Generating visualizations...")
    plot_benchmark_results(final_df, args.output_dir)

    # Generate Report
    logger.info("Generating markdown report...")
    generate_markdown_report(
        final_df,
        os.path.join(args.output_dir, "report.md"),
        statistical_report=runner.last_statistical_report,
    )

    logger.info(f"Benchmark Suite execution complete. See {args.output_dir}/ directory for details.")

if __name__ == "__main__":
    main()
