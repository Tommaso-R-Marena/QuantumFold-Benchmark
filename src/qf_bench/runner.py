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
        results = []
        logger.info(f"Starting benchmark for dataset: {dataset_name} ({len(targets)} targets)")

        for target in tqdm(targets, desc=f"Dataset: {dataset_name}"):
            target_id = target["id"]
            sequence = target["sequence"]
            logger.debug(f"Benchmarking target {target_id}...")

            # In a real benchmark, we'd have a ground truth PDB.
            # Here we either download it or use a placeholder.
            ground_truth_path = self.data_loader.download_pdb(target_id, sequence=sequence)

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

        df = pd.DataFrame(results)
        if not df.empty:
            summary = df.groupby("model")[["rmsd", "tm_score", "plddt"]].mean().reset_index()
            print(f"\nSummary for {dataset_name}:")
            print(tabulate(summary, headers="keys", tablefmt="pretty", showindex=False))

        return df

    def save_results(self, df: pd.DataFrame, filename: str = "benchmark_results.csv"):
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")
