# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import glob
import json
import os
from typing import Iterable, List, Optional

import numpy as np
from tqdm.auto import tqdm


def parse_PDB_chain(
    path_to_pdb: str,
    atoms: Iterable[str] = ["N", "CA", "C"],
    chain: Optional[str] = None,
):
    """
    input:  path_to_pdb = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """

    def N_to_AA(x):
        return ["".join([AA_N_1.get(a, "-") for a in x])]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(path_to_pdb, "r"):
        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi
                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(AA_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))

        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(seq_)
    except TypeError:
        return None, None


def parse_PDBs(
    pdb_path: str,
    jsonl_path: Optional[str] = None,
    input_chain_list: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Parse PDB file(s).

    Parameters
    ----------
    """
    # get pdb files
    if os.path.isfile(pdb_path):
        pdb_files = [pdb_path]
    else:
        pdb_files = glob.glob(os.path.join(pdb_path, "*.pdb"))

    # parse pdb files
    pdb_infos = []
    for pdb_file in tqdm(pdb_files, desc="Parsing PDB files", disable=not verbose):
        pdb_info = {}
        concat_seq = ""

        # get chains
        num_chains = 0
        if input_chain_list is not None:
            chain_alphabet = input_chain_list
        else:
            chains = []
            with open(pdb_file, "r") as pdb:
                for line in pdb:
                    if line.startswith("ATOM"):
                        chain = line[21:22]
                        chains.append(chain)
            chain_alphabet = list(set(chains))

        # process chains
        for chain in chain_alphabet:
            coords, seq = parse_PDB_chain(
                pdb_file, atoms=["N", "CA", "C", "O"], chain=chain
            )
            if coords is not None:
                concat_seq += seq[0]
                pdb_info[f"seq_chain_{chain}"] = seq[0]
                coords_dict_chain = {}
                coords_dict_chain[f"N_chain_{chain}"] = coords[:, 0, :].tolist()
                coords_dict_chain[f"CA_chain_{chain}"] = coords[:, 1, :].tolist()
                coords_dict_chain[f"C_chain_{chain}"] = coords[:, 2, :].tolist()
                coords_dict_chain[f"O_chain_{chain}"] = coords[:, 3, :].tolist()
                pdb_info[f"coords_chain_{chain}"] = coords_dict_chain
                num_chains += 1

        # summary info
        pdb_info["name"] = os.path.basename(pdb_file).split(".")[0]
        pdb_info["num_of_chains"] = num_chains
        pdb_info["seq"] = concat_seq
        pdb_infos.append(pdb_info)

    # outputs
    if jsonl_path is not None:
        if not os.path.exists(os.path.dirname(jsonl_path)):
            os.makedirs(os.path.dirname(jsonl_path))
        with open(jsonl_path, "w") as f:
            for pdb_info in pdb_infos:
                f.write(json.dumps(pdb_info) + "\n")
    else:
        return pdb_infos


# amino acids
ALPHA_1 = list("ARNDCQEGHILKMFPSTWYV-")
ALPHA_3 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "GAP",
]

# amino acid lookups
AA_1_N = {a: n for n, a in enumerate(ALPHA_1)}
AA_3_N = {a: n for n, a in enumerate(ALPHA_3)}
AA_N_1 = {n: a for n, a in enumerate(ALPHA_1)}
AA_1_3 = {a: b for a, b in zip(ALPHA_1, ALPHA_3)}
AA_3_1 = {b: a for a, b in zip(ALPHA_1, ALPHA_3)}
