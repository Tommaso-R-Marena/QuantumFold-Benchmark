from abc import ABC, abstractmethod
from typing import Optional

class FoldingModel(ABC):
    """
    Abstract base class for protein folding models.
    All model wrappers should inherit from this class.
    """

    @abstractmethod
    def predict(self, sequence: str, output_path: str) -> str:
        """
        Predicts the 3D structure for a given amino acid sequence.

        Args:
            sequence (str): The amino acid sequence of the protein.
            output_path (str): The path where the resulting PDB file should be saved.

        Returns:
            str: The path to the saved PDB file.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the model.

        Returns:
            str: Model name.
        """
        pass
