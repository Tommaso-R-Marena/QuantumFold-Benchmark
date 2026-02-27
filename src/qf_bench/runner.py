import os
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models.base import FoldingModel
from .data.loader import BenchmarkDataLoader
from .metrics.scoring import calculate_metrics
from .metrics.statistics import compare_models

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, models: List[FoldingModel], data_loader: BenchmarkDataLoader, output_dir: str = "results"):
        self.models = models
        self.data_loader = data_loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.last_statistical_report = pd.DataFrame()

    def run_benchmark(self, dataset_name: str, targets: List[Dict], max_workers: int = 4) -> pd.DataFrame:
        """
        Runs the benchmark suite on a specific dataset using parallel execution.

        Args:
            dataset_name (str): Name of the dataset (e.g., "CASP15").
            targets (List[Dict]): List of target dictionaries with 'id' and 'sequence'.
            max_workers (int): Maximum number of threads for parallel execution.

        Returns:
            pd.DataFrame: DataFrame containing the benchmarking results.
        """
        results = []
        logger.info(f"Starting benchmark for dataset: {dataset_name} ({len(targets)} targets)")

        def process_target_model(target, model):
            target_id = target["id"]
            sequence = target["sequence"]

            # Fetch or generate ground truth PDB (locking not strictly necessary if download_pdb handles it or if they are pre-downloaded)
            try:
                ground_truth_path = self.data_loader.download_pdb(target_id, sequence=sequence)
            except Exception as e:
                return f"Failed to obtain ground truth for {target_id}: {e}"

            pred_path = os.path.join(self.output_dir, f"{target_id}_{model.name}.pdb")
            try:
                model.predict(sequence, pred_path)
                metrics = calculate_metrics(ground_truth_path, pred_path)

                return {
                    "target_id": target_id,
                    "dataset": dataset_name,
                    "model": model.name,
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                }
            except Exception as e:
                return f"Error running {model.name} on {target_id}: {e}"

        tasks = []
        for target in targets:
            for model in self.models:
                tasks.append((target, model))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_target_model, t, m): (t, m) for t, m in tasks}

            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Dataset: {dataset_name}"):
                res = future.result()
                if isinstance(res, dict):
                    results.append(res)
                else:
                    logger.error(res)

        if not results:
            logger.warning(f"No results generated for dataset: {dataset_name}")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Display summary in console
        summary = df.groupby("model")[["rmsd", "tm_score", "plddt"]].mean().reset_index()
        logger.info(f"\nSummary for {dataset_name}:\n{tabulate(summary, headers='keys', tablefmt='pretty', showindex=False)}")

        baseline = self.models[0].name if self.models else None
        if baseline and len(df["model"].unique()) > 1:
            self.last_statistical_report = compare_models(df, baseline_model=baseline)
            if not self.last_statistical_report.empty:
                logger.info(
                    "\nPaired statistical comparison vs %s:\n%s",
                    baseline,
                    tabulate(self.last_statistical_report, headers='keys', tablefmt='pretty', showindex=False),
                )

        return df

    def save_results(self, df: pd.DataFrame, filename: str = "benchmark_results.csv"):
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")
