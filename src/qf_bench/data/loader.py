import json
import logging
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

    def download_pdb(self, pdb_id: str, sequence: Optional[str] = None) -> str:
        """
        Download a PDB file from RCSB or create a dummy if not available.
        Thread-safe implementation.

        Args:
            pdb_id: The 4-character PDB ID.
            sequence: Optional amino acid sequence to use for the dummy if download fails.

        Returns:
            str: Path to the downloaded/generated PDB file.
        """
        path = self.cache_dir / f"{pdb_id}.pdb"

        with self._download_lock:
            if not path.exists():
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                try:
                    logger.info(f"Attempting to download PDB {pdb_id} from RCSB...")
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        with open(path, "w") as f:
                            f.write(resp.text)
                        logger.info(f"Successfully downloaded PDB {pdb_id}")
                    else:
                        logger.warning(
                            f"PDB {pdb_id} not found in RCSB (Status: {resp.status_code}). Creating dummy."
                        )
                        self._create_robust_dummy_pdb(pdb_id, path, sequence)
                except Exception as e:
                    logger.error(f"Error downloading PDB {pdb_id}: {e}. Creating dummy.")
                    self._create_robust_dummy_pdb(pdb_id, path, sequence)

        return str(path)

    def _create_robust_dummy_pdb(
        self, pdb_id: str, path: Path, sequence: Optional[str] = None
    ) -> None:
        """
        Creates a dummy PDB file with a realistic-ish linear structure.
        Used when the real PDB is not available.
        """
        # Use provided sequence or fallback to a default
        if not sequence:
            sequence = "G" * 50
            logger.info(f"No sequence provided for dummy {pdb_id}, using 50x Glycine.")

        coords = []
        for i in range(len(sequence)):
            # Linear arrangement with 3.8A between C-alpha atoms
            coord = np.array([float(i) * 3.8, 0.0, 0.0], dtype="f")
            coords.append(coord)

        save_to_pdb(sequence, np.array(coords), str(path), pdb_id=pdb_id)
        logger.info(f"Created robust dummy PDB for {pdb_id} at {path}")
