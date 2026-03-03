from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..utils.pdb_utils import save_to_pdb


class FoldingModel(ABC):
    """
    Abstract base class for protein folding models.
    All model wrappers should inherit from this class to ensure a consistent
    interface across different folding engines (quantum, classical, or API-based).
    """

    @abstractmethod
    def predict(self, sequence: str, output_path: str) -> str:
        """
        Predicts the 3D structure for a given amino acid sequence and saves it to a PDB file.

        Args:
            sequence (str): The amino acid sequence of the protein (e.g., "MKV...").
            output_path (str): The filesystem path where the resulting PDB file should be saved.

        Returns:
            str: The absolute or relative path to the saved PDB file.

        Raises:
            Exception: If prediction fails or output cannot be written.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the human-readable name of the model.

        Returns:
            str: The model name (e.g., "AlphaFold3", "QuantumFold-Advantage").
        """
        pass


class SimulationMixin:
    """
    Mixin to provide common simulation capabilities for folding models.
    Useful for creating realistic-looking simulated outputs for testing or
    benchmarking when actual inference is not available.
    """

    def _simulate_fold(
        self,
        sequence: str,
        output_path: str,
        pdb_id: str,
        rng: np.random.Generator,
        persistence: float = 0.8,
        jitter_scale: float = 0.1,
        plddt_base: float = 90.0,
        plddt_range: tuple = (-5, 5),
        bias_direction: Optional[np.ndarray] = None,
    ) -> str:
        """
        Simulates a protein fold using a persistent random walk.

        Args:
            sequence: The amino acid sequence.
            output_path: Path to save the PDB.
            pdb_id: ID for the PDB structure.
            rng: NumPy random generator.
            persistence: How much of the previous direction to keep (0 to 1).
            jitter_scale: Scale of random noise added to each step.
            plddt_base: Base pLDDT value.
            plddt_range: Range of random variation for pLDDT.
            bias_direction: Optional direction vector to bias the walk.

        Returns:
            str: Path to the saved PDB file.
        """
        coords = []
        plddts = []
        pos = np.array([0.0, 0.0, 0.0])

        if bias_direction is not None:
            direction = bias_direction / np.linalg.norm(bias_direction)
        else:
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)

        for i in range(len(sequence)):
            # Update direction
            new_dir = rng.normal(0, 1, 3)
            new_dir /= np.linalg.norm(new_dir)

            direction = (persistence * direction) + ((1 - persistence) * new_dir)
            direction /= np.linalg.norm(direction)

            # Add jitter
            jitter = rng.normal(0, jitter_scale, 3)
            pos = pos + (direction * 3.8) + jitter
            coords.append(pos.copy())

            # Mock pLDDT
            plddt = plddt_base + rng.uniform(*plddt_range)
            plddts.append(np.clip(plddt, 0, 100))

        save_to_pdb(
            sequence,
            np.array(coords),
            output_path,
            pdb_id=pdb_id,
            plddts=np.array(plddts),
        )
        return output_path
