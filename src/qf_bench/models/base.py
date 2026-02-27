from abc import ABC, abstractmethod

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
