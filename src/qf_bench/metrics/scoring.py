import numpy as np
from Bio.PDB import PDBParser, Superimposer
import os
from typing import Dict, Optional, List

def calculate_rmsd(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates RMSD between two PDB files using CA atoms.

    Args:
        ref_pdb: Path to the reference PDB file.
        target_pdb: Path to the predicted PDB file.

    Returns:
        float: The RMSD value.
    """
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", ref_pdb)
    target_struct = parser.get_structure("target", target_pdb)

    ref_atoms = []
    target_atoms = []

    for model in ref_struct:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ref_atoms.append(residue["CA"])

    for model in target_struct:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    target_atoms.append(residue["CA"])

    # Ensure same number of atoms
    min_len = min(len(ref_atoms), len(target_atoms))
    if min_len == 0:
        return float('nan')

    ref_atoms = ref_atoms[:min_len]
    target_atoms = target_atoms[:min_len]

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    return superimposer.rms

def calculate_tm_score(ref_pdb: str, target_pdb: str) -> float:
    """
    Calculates an approximate TM-score between two PDB files.

    Args:
        ref_pdb: Path to the reference PDB file.
        target_pdb: Path to the predicted PDB file.

    Returns:
        float: The TM-score value (0 to 1).
    """
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", ref_pdb)
    target_struct = parser.get_structure("target", target_pdb)

    ref_coords = []
    target_coords = []

    for model in ref_struct:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ref_coords.append(residue["CA"].get_coord())

    for model in target_struct:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    target_coords.append(residue["CA"].get_coord())

    min_len = min(len(ref_coords), len(target_coords))
    if min_len == 0:
        return 0.0

    ref_coords = np.array(ref_coords[:min_len])
    target_coords = np.array(target_coords[:min_len])

    # Superimpose to get minimal distances
    # We use a simple Kabsch alignment for CA atoms
    def kabsch(A, B):
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_B - np.dot(centroid_A, R)
        return R, t

    R, t = kabsch(ref_coords, target_coords)
    ref_aligned = np.dot(ref_coords, R) + t

    distances = np.linalg.norm(ref_aligned - target_coords, axis=1)

    L_target = len(ref_coords)
    d0 = 1.24 * (max(L_target, 15) - 15)**(1/3) - 1.8
    if d0 <= 0.5: d0 = 0.5

    tm_score = np.sum(1.0 / (1.0 + (distances / d0)**2)) / L_target
    return tm_score

def calculate_plddt(pdb_path: str) -> float:
    """
    Extracts average pLDDT from the B-factor column of a PDB file.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        float: Average pLDDT value.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("struct", pdb_path)
    bfactors = []
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    bfactors.append(atom.get_bfactor())
    if not bfactors:
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
        print(f"Error calculating metrics: {e}")
        metrics["rmsd"] = float('nan')
        metrics["tm_score"] = 0.0
        metrics["plddt"] = 0.0
    return metrics
