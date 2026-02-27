import numpy as np
from Bio.PDB import PDBParser, Superimposer, Atom
import os
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

def _get_ca_atoms_dict(structure) -> Dict[Tuple, Atom.Atom]:
    """Helper to extract CA atoms with (chain_id, res_id) as keys."""
    ca_atoms = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                if "CA" in residue:
                    res_id = residue.get_id()
                    ca_atoms[(chain_id, res_id)] = residue["CA"]
    return ca_atoms

def get_common_ca_atoms(ref_pdb: str, target_pdb: str) -> Tuple[List[Atom.Atom], List[Atom.Atom]]:
    """
    Parses two PDB files and returns lists of CA atoms common to both,
    matched by chain and residue ID.
    """
    parser = PDBParser(QUIET=True)
    try:
        ref_struct = parser.get_structure("ref", ref_pdb)
        target_struct = parser.get_structure("target", target_pdb)
    except Exception as e:
        logger.error(f"Error parsing PDB files for atom matching: {e}")
        return [], []

    ref_ca = _get_ca_atoms_dict(ref_struct)
    target_ca = _get_ca_atoms_dict(target_struct)

    common_keys = sorted(set(ref_ca.keys()) & set(target_ca.keys()))

    if not common_keys:
        logger.warning(f"No common CA atoms found between {ref_pdb} and {target_pdb}")
        return [], []

    if len(common_keys) != len(ref_ca) or len(common_keys) != len(target_ca):
        logger.info(f"Matched {len(common_keys)} CA atoms (Ref total: {len(ref_ca)}, Target total: {len(target_ca)})")

    ref_list = [ref_ca[k] for k in common_keys]
    target_list = [target_ca[k] for k in common_keys]

    return ref_list, target_list

def calculate_rmsd(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates RMSD between two PDB files using matched CA atoms.

    Args:
        ref_pdb: Path to the reference PDB file.
        target_pdb: Path to the predicted PDB file.

    Returns:
        float: The RMSD value.
    """
    ref_atoms, target_atoms = get_common_ca_atoms(ref_pdb, target_pdb)

    if not ref_atoms:
        return float('nan')

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    return superimposer.rms

def calculate_tm_score(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates an approximate TM-score between two PDB files using matched CA atoms.

    Args:
        ref_pdb: Path to the reference PDB file.
        target_pdb: Path to the predicted PDB file.

    Returns:
        float: The TM-score value (0 to 1).
    """
    ref_atoms, target_atoms = get_common_ca_atoms(ref_pdb, target_pdb)

    if not ref_atoms:
        return 0.0

    # Superimpose using Bio.PDB.Superimposer (uses Kabsch algorithm internally)
    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    superimposer.apply(target_atoms)

    ref_coords = np.array([a.get_coord() for a in ref_atoms])
    target_coords = np.array([a.get_coord() for a in target_atoms])

    distances = np.linalg.norm(ref_coords - target_coords, axis=1)

    # TM-score formula: 1/L_target * sum(1 / (1 + (d_i/d0)^2))
    # Note: L_target should be the length of the reference protein
    L_target = len(ref_atoms)
    if L_target == 0:
        return 0.0

    d0 = 1.24 * (max(L_target, 15) - 15)**(1/3) - 1.8
    if d0 <= 0.5:
        d0 = 0.5

    tm_score = np.sum(1.0 / (1.0 + (distances / d0)**2)) / L_target
    return tm_score

def calculate_plddt(pdb_path: str) -> float:
    """
    Extracts average pLDDT from the B-factor column of CA atoms in a PDB file.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        float: Average pLDDT value.
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("struct", pdb_path)
    except Exception as e:
        logger.error(f"Error parsing PDB for pLDDT: {e}")
        return 0.0

    bfactors = []
    for model in struct:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    bfactors.append(residue["CA"].get_bfactor())

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
        metrics["plddt"] = calculate_plddt(target_pdb)
    except Exception as e:
        logger.exception(f"Unexpected error calculating metrics for {target_pdb}: {e}")
        metrics["rmsd"] = metrics.get("rmsd", float('nan'))
        metrics["tm_score"] = metrics.get("tm_score", 0.0)
        metrics["plddt"] = metrics.get("plddt", 0.0)
    return metrics
