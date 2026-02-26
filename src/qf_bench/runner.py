import os
import pandas as pd
from datetime import datetime
from typing import List, Dict
from .models.base import FoldingModel
from .data.loader import BenchmarkDataLoader
from .metrics.scoring import calculate_metrics

class BenchmarkRunner:
    def __init__(self, models: List[FoldingModel], data_loader: BenchmarkDataLoader, output_dir: str = "results"):
        self.models = models
        self.data_loader = data_loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark(self, dataset_name: str, targets: List[Dict]) -> pd.DataFrame:
        results = []
        for target in targets:
            target_id = target["id"]
            sequence = target["sequence"]
            print(f"Benchmarking target {target_id}...")

            # In a real benchmark, we'd have a ground truth PDB.
            # Here we either download it or use a placeholder.
            ground_truth_path = self.data_loader.download_pdb(target_id)

            for model in self.models:
                print(f"  Running model {model.name}...")
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
                    print(f"    Error running {model.name} on {target_id}: {e}")

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, filename: str = "benchmark_results.csv"):
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")
