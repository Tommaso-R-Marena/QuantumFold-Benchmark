import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO, Polypeptide
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def save_to_pdb(
    sequence: str,
    coords: np.ndarray,
    output_path: str,
    pdb_id: str = "predicted",
    plddts: Optional[np.ndarray] = None,
    occupancy: float = 1.0
) -> str:
    """
    Saves a protein sequence and coordinates to a PDB file.

    Args:
        sequence (str): Amino acid sequence.
        coords (np.ndarray): Nx3 array of coordinates (typically CA atoms).
        output_path (str): Path to save the PDB file.
        pdb_id (str): ID for the structure.
        plddts (np.ndarray, optional): Array of pLDDT scores for each residue.
        occupancy (float): Occupancy value for atoms.

    Returns:
        str: Path to the saved PDB file.
    """
    if len(sequence) != len(coords):
        raise ValueError(f"Sequence length ({len(sequence)}) must match coordinates length ({len(coords)})")

    struct = Structure.Structure(pdb_id)
    model = Model.Model(0)
    chain = Chain.Chain("A")

    for i, (aa, coord) in enumerate(zip(sequence, coords)):
        try:
            res_name = Polypeptide.one_to_three(aa)
        except Exception:
            res_name = "UNK"

        res = Residue.Residue((" ", i + 1, " "), res_name, i + 1)

        b_factor = plddts[i] if plddts is not None else 100.0

        atom = Atom.Atom(
            "CA",
            coord.astype('f'),
            b_factor,
            occupancy,
            " ",
            "CA",
            i + 1,
            "C"
        )
        res.add(atom)
        chain.add(res)

    model.add(chain)
    struct.add(model)

    io = PDBIO()
    io.set_structure(struct)
    io.save(output_path)

    return output_path
