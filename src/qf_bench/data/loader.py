import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from ..utils.pdb_utils import save_to_pdb

logger = logging.getLogger(__name__)


class BenchmarkDataLoader:
    """
    Data loader for protein folding benchmarks.
    Handles fetching of CASP15, miniproteins, and IDRs.
    """

    def __init__(self, cache_dir: str | Path = "data/cache"):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to cache downloaded PDB files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.targets_file = Path(__file__).parent / "targets.json"
        self._download_lock = Lock()
        self.all_targets: Dict[str, List[Dict[str, str]]] = {}
        self._load_targets()

    def _load_targets(self) -> None:
        try:
            with open(self.targets_file, "r") as f:
                self.all_targets = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load targets from {self.targets_file}: {e}")
            self.all_targets = {}

    def get_casp15_targets(self) -> List[Dict[str, str]]:
        """
        Fetch representative CASP15 target metadata.

        Returns:
            List[Dict]: List of targets with 'id' and 'sequence'.
        """
        return self.all_targets.get("casp15", [])

    def get_miniproteins(self) -> List[Dict[str, str]]:
        """
        Fetch representative miniprotein targets.

        Returns:
            List[Dict]: List of targets with 'id' and 'sequence'.
        """
        return self.all_targets.get("miniproteins", [])

    def get_idrs(self) -> List[Dict[str, str]]:
        """
        Fetch targets with Intrinsically Disordered Regions.

        Returns:
            List[Dict]: List of targets with 'id' and 'sequence'.
        """
        return self.all_targets.get("idrs", [])

    def download_pdb(
        self, pdb_id: str, sequence: Optional[str] = None, max_retries: int = 3
    ) -> str:
        """
        Download a PDB file from RCSB or create a dummy if not available.
        Thread-safe implementation with exponential backoff.

        Args:
            pdb_id: The 4-character PDB ID.
            sequence: Optional amino acid sequence to use for the dummy if download fails.
            max_retries: Number of retry attempts for network errors.

        Returns:
            str: Path to the downloaded/generated PDB file.
        """
        path = self.cache_dir / f"{pdb_id}.pdb"

        with self._download_lock:
            if not path.exists():
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                success = False
                for attempt in range(max_retries):
                    try:
                        logger.info(
                            f"Attempt {attempt + 1}/{max_retries} to download PDB {pdb_id} from RCSB..."
                        )
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            with open(path, "w") as f:
                                f.write(resp.text)
                            logger.info(f"Successfully downloaded PDB {pdb_id}")
                            success = True
                            break
                        elif resp.status_code == 404:
                            logger.warning(f"PDB {pdb_id} not found in RCSB (404).")
                            break
                        else:
                            logger.warning(
                                f"Unexpected status {resp.status_code} for {pdb_id}"
                            )
                    except Exception as e:
                        logger.error(f"Error downloading PDB {pdb_id} (Attempt {attempt + 1}): {e}")

                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff

                if not success and not path.exists():
                    logger.info(f"Falling back to dummy PDB for {pdb_id}")
                    self._create_robust_dummy_pdb(pdb_id, path, sequence)

        return str(path)

    def _create_robust_dummy_pdb(
        self, pdb_id: str, path: Path, sequence: Optional[str] = None
    ) -> None:
        """
        Creates a dummy PDB file with a realistic-ish helical structure.
        Used when the real PDB is not available.
        """
        # Use provided sequence or fallback to a default
        if not sequence:
            sequence = "G" * 50
            logger.info(f"No sequence provided for dummy {pdb_id}, using 50x Glycine.")

        coords = []
        # Alpha-helical parameters:
        # Rise per residue: 1.5A
        # Rotation per residue: 100 degrees (100 * pi / 180 radians)
        # Radius: ~2.3A
        rise = 1.5
        rotation_per_res = 100 * np.pi / 180
        radius = 2.3

        for i in range(len(sequence)):
            angle = i * rotation_per_res
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = i * rise
            coords.append(np.array([x, y, z], dtype="f"))

        save_to_pdb(sequence, np.array(coords), str(path), pdb_id=pdb_id)
        logger.info(f"Created robust dummy helical PDB for {pdb_id} at {path}")
