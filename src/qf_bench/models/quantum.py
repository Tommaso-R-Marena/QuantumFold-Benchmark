import logging
from typing import Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from ..utils.pdb_utils import save_to_pdb
from .base import FoldingModel

logger = logging.getLogger(__name__)

HYDROPHOBIC = set("AILMFWVPG")


class QuantumFoldAdvantage(FoldingModel):
    """Hybrid quantum-classical folding model with a toy but iterative VQE loop."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        optimizer: str = "COBYLA",
        maxiter: int = 40,
        shots: int = 512,
    ):
        """
        Initialize the QuantumFold-Advantage model.

        Args:
            api_key: Optional API key for cloud execution (simulated here).
            seed: Random seed for reproducibility.
            optimizer: Classical optimizer to use (e.g., COBYLA, SLSQP).
            maxiter: Maximum number of optimizer iterations.
            shots: Number of shots for quantum circuit execution.
        """
        self.api_key = api_key
        self.simulator = AerSimulator()
        self.seed = seed
        self.optimizer = optimizer.upper()
        self.maxiter = maxiter
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.last_run_stats: Dict[str, float] = {}
        self._last_optimizer_iterations = 0

    @property
    def name(self) -> str:
        return "QuantumFold-Advantage"

    def predict(self, sequence: str, output_path: str) -> str:
        """
        Predicts protein structure using a hybrid VQE approach.

        Args:
            sequence: Amino acid sequence.
            output_path: Path to save the resulting PDB file.

        Returns:
            str: Path to the saved PDB file.
        """
        logger.info(
            "Initializing QuantumFold-Advantage for sequence of length %s...",
            len(sequence),
        )
        n_qubits = min(max(len(sequence), 2), 16)
        hamiltonian = self._build_ising_hamiltonian(sequence, n_qubits)

        best_params, best_energy = self._optimize_vqe(hamiltonian, n_qubits)
        bitstring, resources = self._sample_best_bitstring(best_params, n_qubits)

        self.last_run_stats = {
            "energy": float(best_energy),
            "n_qubits": n_qubits,
            "circuit_depth": resources["depth"],
            "gate_count": resources["gate_count"],
            "nonlocal_gate_count": resources["nonlocal_gate_count"],
            "optimizer_iterations": resources["iterations"],
            "shots": self.shots,
        }

        self._create_vqe_pdb(sequence, bitstring, output_path)
        logger.info(
            "QuantumFold-Advantage prediction complete for %s | stats=%s",
            output_path,
            self.last_run_stats,
        )
        return output_path

    def _build_ansatz(self, n_qubits: int, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        layer_size = 2 * n_qubits
        n_layers = max(1, len(params) // layer_size)
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qc.ry(params[idx], q)
                idx += 1
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(n_qubits):
                qc.rz(params[idx], q)
                idx += 1
        return qc

    def _build_ising_hamiltonian(
        self, sequence: str, n_qubits: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct a simple HP-inspired Ising model H = sum h_i z_i + sum J_ij z_i z_j."""
        h = np.zeros(n_qubits)
        J = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            aa = sequence[i % len(sequence)]
            h[i] = -0.4 if aa in HYDROPHOBIC else 0.2
            if i < n_qubits - 1:
                J[i, i + 1] = -0.15  # compactness preference
            if i + 2 < n_qubits:
                J[i, i + 2] = 0.05  # discourage tight clashes
        return h, J

    def _bitstring_energy(
        self, bitstring: str, hamiltonian: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        h, J = hamiltonian
        z = np.array([1 if b == "0" else -1 for b in bitstring])
        energy = float(np.dot(h, z))
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                if J[i, j] != 0:
                    energy += float(J[i, j] * z[i] * z[j])
        return energy

    def _expected_energy(
        self, params: np.ndarray, hamiltonian: Tuple[np.ndarray, np.ndarray], n_qubits: int
    ) -> float:
        qc = self._build_ansatz(n_qubits, params)
        qc.measure_all()
        result = self.simulator.run(
            qc, shots=self.shots, seed_simulator=self.seed
        ).result()
        counts = result.get_counts()
        total = sum(counts.values())
        exp_energy = 0.0
        for bitstring, count in counts.items():
            exp_energy += self._bitstring_energy(bitstring, hamiltonian) * (
                count / total
            )
        return exp_energy

    def _optimize_vqe(
        self, hamiltonian: Tuple[np.ndarray, np.ndarray], n_qubits: int
    ) -> Tuple[np.ndarray, float]:
        n_layers = 2
        n_params = n_layers * 2 * n_qubits
        initial = self.rng.uniform(0, 2 * np.pi, n_params)

        def objective(params: np.ndarray) -> float:
            return self._expected_energy(params, hamiltonian, n_qubits)

        method = (
            "COBYLA"
            if self.optimizer not in {"COBYLA", "POWELL", "NELDER-MEAD"}
            else self.optimizer
        )
        opt_result = minimize(
            objective,
            initial,
            method=method,
            options={"maxiter": self.maxiter, "disp": False},
        )
        if not opt_result.success:
            logger.warning("VQE optimizer did not fully converge: %s", opt_result.message)
        self._last_optimizer_iterations = int(getattr(opt_result, "nit", self.maxiter))
        return opt_result.x, float(opt_result.fun)

    def _sample_best_bitstring(
        self, params: np.ndarray, n_qubits: int
    ) -> Tuple[str, Dict[str, int]]:
        qc = self._build_ansatz(n_qubits, params)
        qc.measure_all()
        result = self.simulator.run(
            qc, shots=self.shots, seed_simulator=self.seed
        ).result()
        counts = result.get_counts()
        bitstring = max(counts, key=counts.get)
        resources = {
            "depth": int(qc.depth()),
            "gate_count": int(sum(qc.count_ops().values())),
            "nonlocal_gate_count": int(qc.num_nonlocal_gates()),
            "iterations": int(getattr(self, "_last_optimizer_iterations", self.maxiter)),
        }
        return bitstring, resources

    def _create_vqe_pdb(self, sequence: str, bitstring: str, output_path: str) -> None:
        """
        Translates the quantum bitstring into a 3D structure.
        Uses a random walk with bias based on the bitstring.
        """
        pos = np.array([0.0, 0.0, 0.0], dtype="f")
        # Standard bond length ~3.8A
        bond_length = 3.8

        # Directions for a simple cubic lattice walk
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ]

        coords = []
        plddts = []

        # We'll use a slightly more sophisticated walk to avoid trivial linear structures
        # even if it's still very much a toy.
        for i in range(len(sequence)):
            # Use bitstring to influence direction
            bit_idx = (i * 3) % len(bitstring)
            chunk = bitstring[bit_idx : bit_idx + 3]
            if len(chunk) < 3:
                chunk = (chunk + bitstring)[:3]

            dir_idx = int(chunk, 2) % len(directions)
            base_dir = directions[dir_idx]

            # Add some persistence to the walk
            if i > 0:
                prev_dir = (coords[-1] - coords[-2]) if i > 1 else np.array([1.0, 0, 0])
                prev_dir /= np.linalg.norm(prev_dir)
                # Blend previous direction with new chosen direction
                step_dir = 0.7 * prev_dir + 0.3 * base_dir
                step_dir /= np.linalg.norm(step_dir)
            else:
                step_dir = base_dir

            # Jitter to avoid exact lattice
            jitter = self.rng.normal(0, 0.1, 3)
            pos = pos + (step_dir * bond_length) + jitter

            coords.append(pos.copy())

            # Mock pLDDT: decays slightly with length, some noise
            base_plddt = max(40.0, 85.0 - len(sequence) * 0.05)
            plddts.append(base_plddt + self.rng.uniform(-10, 5))

        save_to_pdb(sequence, np.array(coords), output_path, plddts=np.array(plddts))
