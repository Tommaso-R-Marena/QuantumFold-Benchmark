import os
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from tabulate import tabulate
from .models.base import FoldingModel
from .data.loader import BenchmarkDataLoader
from .metrics.scoring import calculate_metrics

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, models: List[FoldingModel], data_loader: BenchmarkDataLoader, output_dir: str = "results"):
        self.models = models
        self.data_loader = data_loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark(self, dataset_name: str, targets: List[Dict]) -> pd.DataFrame:
        """
        Runs the benchmark suite on a specific dataset.

        Args:
            dataset_name (str): Name of the dataset (e.g., "CASP15").
            targets (List[Dict]): List of target dictionaries with 'id' and 'sequence'.

        Returns:
            pd.DataFrame: DataFrame containing the benchmarking results.
        """
        results = []
        logger.info(f"Starting benchmark for dataset: {dataset_name} ({len(targets)} targets)")

        for target in tqdm(targets, desc=f"Dataset: {dataset_name}"):
            target_id = target["id"]
            sequence = target["sequence"]
            logger.info(f"Benchmarking target {target_id} (length {len(sequence)})...")

            # Fetch or generate ground truth PDB
            try:
                ground_truth_path = self.data_loader.download_pdb(target_id, sequence=sequence)
            except Exception as e:
                logger.error(f"Failed to obtain ground truth for {target_id}: {e}")
                continue

            for model in self.models:
                logger.debug(f"  Running model {model.name}...")
                pred_path = os.path.join(self.output_dir, f"{target_id}_{model.name}.pdb")
                try:
                    model.predict(sequence, pred_path)
                    metrics = calculate_metrics(ground_truth_path, pred_path)

                    res = {
                        "target_id": target_id,
                        "dataset": dataset_name,
                        "model": model.name,
                        "timestamp": datetime.now().isoformat(),
                        **metrics
                    }
                    results.append(res)
                except Exception as e:
                    logger.error(f"Error running {model.name} on {target_id}: {e}")

        if not results:
            logger.warning(f"No results generated for dataset: {dataset_name}")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Display summary in console
        summary = df.groupby("model")[["rmsd", "tm_score", "plddt"]].mean().reset_index()
        logger.info(f"\nSummary for {dataset_name}:\n{tabulate(summary, headers='keys', tablefmt='pretty', showindex=False)}")

        return df

    def save_results(self, df: pd.DataFrame, filename: str = "benchmark_results.csv"):
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")
