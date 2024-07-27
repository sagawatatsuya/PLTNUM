# https://github.com/westlake-repl/SaProt/blob/main/utils/foldseek_util.py

# MIT License

# Copyright (c) 2023 westlake-repl

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import numpy as np
import sys

sys.path.append(".")


# Get structural seqs from pdb file
def get_struc_seq(
    foldseek,
    path,
    chains: list = None,
    process_id: int = 0,
    plddt_path: str = None,
    plddt_threshold: float = 70.0,
) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek
        path: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_path: Path to plddt file. If None, plddt will not be used.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(
        plddt_path
    ), f"Plddt file not found: {plddt_path}"

    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]

            # Mask low plddt
            if plddt_path is not None:
                with open(plddt_path, "r") as r:
                    plddts = np.array(json.load(r)["confidenceScore"])

                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)

            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join(
                        [a + b.lower() for a, b in zip(seq, struc_seq)]
                    )
                    seq_dict[chain] = (seq, struc_seq, combined_seq)

    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


if __name__ == "__main__":
    foldseek = "/sujin/bin/foldseek"
    # test_path = "/sujin/Datasets/PDB/all/6xtd.cif"
    test_path = "/sujin/Datasets/FLIP/meltome/af2_structures/A0A061ACX4.pdb"
    plddt_path = "/sujin/Datasets/FLIP/meltome/af2_plddts/A0A061ACX4.json"
    res = get_struc_seq(
        foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.0
    )
    print(res["A"][1].lower())
