import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure

logger = logging.getLogger(__name__)


def _get_ca_atoms_dict(structure: Structure) -> Dict[Tuple[str, int], Atom]:
    """
    Helper to extract CA atoms with (chain_id, res_id) as keys.
    Handles altloc by picking the primary one.
    Only considers the first model if multiple models are present.
    """
    ca_atoms = {}
    # Use only the first model
    try:
        model = structure[0]
    except (IndexError, KeyError):
        return {}

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            # Check if it's a standard amino acid residue
            if residue.get_id()[0] != " ":  # Skip hetero-atoms
                continue
            if "CA" in residue:
                ca_atom = residue["CA"]
                # If multiple locations exist, pick the first one
                if ca_atom.is_disordered():
                    try:
                        ca_atom = ca_atom.disordered_get()
                    except Exception:
                        # Fallback to choosing one from the disordered atom
                        ca_atom = list(ca_atom.child_dict.values())[0]
                res_id = residue.get_id()[1]
                ca_atoms[(chain_id, res_id)] = ca_atom
    return ca_atoms


def get_common_ca_atoms(
    ref_pdb: str, target_pdb: str
) -> Tuple[List[Atom], List[Atom]]:
    """
    Parses two PDB files and returns lists of CA atoms common to both,
    matched by chain and residue ID.
    """
    parser = PDBParser(QUIET=True)
    try:
        if not os.path.exists(ref_pdb):
            logger.error(f"Reference PDB file not found: {ref_pdb}")
            return [], []
        if not os.path.exists(target_pdb):
            logger.error(f"Target PDB file not found: {target_pdb}")
            return [], []

        ref_struct = parser.get_structure("ref", ref_pdb)
        target_struct = parser.get_structure("target", target_pdb)
    except Exception as e:
        logger.error(f"Error parsing PDB files for atom matching: {e}")
        return [], []

    ref_ca = _get_ca_atoms_dict(ref_struct)
    target_ca = _get_ca_atoms_dict(target_struct)

    common_keys = sorted(set(ref_ca.keys()) & set(target_ca.keys()))

    if not common_keys:
        logger.warning(
            f"No common CA atoms found between {ref_pdb} and {target_pdb}. "
            f"Ref keys: {len(ref_ca)}, Target keys: {len(target_ca)}"
        )
        return [], []

    if len(common_keys) != len(ref_ca) or len(common_keys) != len(target_ca):
        logger.info(
            f"Matched {len(common_keys)} CA atoms (Ref total: {len(ref_ca)}, Target total: {len(target_ca)})"
        )

    ref_list = [ref_ca[k] for k in common_keys]
    target_list = [target_ca[k] for k in common_keys]

    return ref_list, target_list


def calculate_rmsd(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates Root Mean Square Deviation (RMSD) between two PDB files using matched CA atoms.

    Args:
        ref_pdb (str): Path to the reference PDB file.
        target_pdb (str): Path to the predicted PDB file.

    Returns:
        float: The RMSD value in Angstroms. Returns NaN if no common CA atoms are found.
    """
    ref_atoms, target_atoms = get_common_ca_atoms(ref_pdb, target_pdb)

    if not ref_atoms:
        return float("nan")

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    return superimposer.rms


def _iterative_superimposition(
    ref_atoms: List[Atom], target_atoms: List[Atom], max_iterations: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs iterative superimposition to find a robust alignment,
    similar to how TM-score and GDT-TS are calculated in standard tools.
    """
    ref_coords = np.array([a.get_coord() for a in ref_atoms], dtype=np.float32)
    original_target_coords = np.array([a.get_coord() for a in target_atoms], dtype=np.float32)
    current_target_coords = original_target_coords.copy()

    L = len(ref_coords)
    d0 = max(0.5, 1.24 * (max(L, 15) - 15) ** (1 / 3) - 1.8)

    best_tm = -1.0
    best_target_coords = current_target_coords.copy()
    best_distances = np.zeros(L, dtype=np.float32)

    si = Superimposer()

    # Create fixed length list for atoms to avoid re-allocation
    # Using the original atoms for reference as they don't change
    for _ in range(max_iterations):
        # We need a subset of atoms for the superimposer
        # Initial iteration uses all atoms
        if best_tm < 0:
            si.set_atoms(ref_atoms, target_atoms)
        else:
            mask = distances < max(d0, 2.0)
            if np.sum(mask) < 3:
                break
            subset_ref = [ref_atoms[i] for i, m in enumerate(mask) if m]
            subset_target = [target_atoms[i] for i, m in enumerate(mask) if m]
            si.set_atoms(subset_ref, subset_target)

        rot, tran = si.rotran
        # Vectorized apply: (N, 3) @ (3, 3) + (3,)
        current_target_coords = np.dot(original_target_coords, rot) + tran

        # Vectorized distance calculation
        diff = ref_coords - current_target_coords
        distances = np.sqrt(np.sum(diff * diff, axis=1))

        tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L

        if tm_score > best_tm:
            best_tm = tm_score
            best_target_coords = current_target_coords.copy()
            best_distances = distances.copy()
        else:
            # If TM-score doesn't improve, we might be oscillating or converged
            if best_tm > 0:
                break

    return best_target_coords, best_distances, float(best_tm)


def calculate_tm_score(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates a robust TM-score between two PDB files using iterative CA alignment.

    Args:
        ref_pdb: Path to the reference PDB file.
        target_pdb: Path to the predicted PDB file.

    Returns:
        float: The TM-score value (0 to 1).
    """
    ref_atoms, target_atoms = get_common_ca_atoms(ref_pdb, target_pdb)

    if not ref_atoms:
        return 0.0

    _, _, tm_score = _iterative_superimposition(ref_atoms, target_atoms)
    return float(tm_score)


def calculate_gdt_ts(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates robust GDT-TS (Global Distance Test - Total Score).
    GDT-TS = (GDT_1 + GDT_2 + GDT_4 + GDT_8) / 4, where GDT_P is the percentage
    of residues under P Angstroms after robust superimposition.
    """
    ref_atoms, target_atoms = get_common_ca_atoms(ref_pdb, target_pdb)

    if not ref_atoms:
        return 0.0

    # For GDT-TS, we use the distances from a robust iterative alignment
    _, distances, _ = _iterative_superimposition(ref_atoms, target_atoms)

    thresholds = [1.0, 2.0, 4.0, 8.0]
    percentages = []
    for t in thresholds:
        percentages.append(np.mean(distances <= t))

    return float(np.mean(percentages))


def calculate_plddt(pdb_path: str) -> float:
    """
    Extracts average pLDDT from the B-factor column of CA atoms in a PDB file.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        float: Average pLDDT value (0 to 100).
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("struct", pdb_path)
        # Only consider the first model for consistency
        model = struct[0]
    except Exception as e:
        logger.error(f"Error parsing PDB for pLDDT: {e}")
        return 0.0

    bfactors = []
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                ca_atom = residue["CA"]
                if ca_atom.is_disordered():
                    try:
                        ca_atom = ca_atom.disordered_get()
                    except Exception:
                        ca_atom = list(ca_atom.child_dict.values())[0]
                bfactors.append(ca_atom.get_bfactor())

    if not bfactors:
        logger.warning(f"No CA atoms found in {pdb_path} for pLDDT calculation.")
        return 0.0
    return float(np.mean(bfactors))


def calculate_metrics(ref_pdb: str, target_pdb: str) -> Dict[str, float]:
    """
    Calculate multiple structural metrics.

    Args:
        ref_pdb: Path to the reference PDB file.
        target_pdb: Path to the predicted PDB file.

    Returns:
        Dict[str, float]: Dictionary of metric names and values.
    """
    metrics = {}
    try:
        metrics["rmsd"] = calculate_rmsd(ref_pdb, target_pdb)
        metrics["tm_score"] = calculate_tm_score(ref_pdb, target_pdb)
        metrics["gdt_ts"] = calculate_gdt_ts(ref_pdb, target_pdb)
        metrics["plddt"] = calculate_plddt(target_pdb)
    except Exception as e:
        logger.exception(f"Unexpected error calculating metrics for {target_pdb}: {e}")
        metrics["rmsd"] = metrics.get("rmsd", float("nan"))
        metrics["tm_score"] = metrics.get("tm_score", 0.0)
        metrics["gdt_ts"] = metrics.get("gdt_ts", 0.0)
        metrics["plddt"] = metrics.get("plddt", 0.0)
    return metrics
