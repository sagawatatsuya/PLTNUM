import argparse
import glob
import multiprocessing as mp
import os
import random

import pandas as pd
from foldseek_util import get_struc_seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default="./pdb_files",
        help="Directory containing PDB files.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
        help="Number of processes to use for multiprocessing. Default is 2.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory.",
    )
    return parser.parse_args()


def get_foldseek_seq(pdb_path):
    parsed_seqs = get_struc_seq(
        "bin/foldseek",
        pdb_path,
        ["A"],
        process_id=random.randint(0, 10000000),
    )["A"]
    return parsed_seqs


if __name__ == "__main__":
    config = parse_args()

    pdb_files = glob.glob(os.path.join(config.pdb_dir, "*.pdb"))

    with mp.Pool(config.num_processes) as pool:
        output = pool.map(get_foldseek_seq, pdb_files)

    aa, foldseek, aa_foldseek = zip(*output)

    result = {}
    result["file"] = pdb_files
    result["aa"] = aa
    result["foldseek"] = foldseek
    result["aa_foldseek"] = aa_foldseek

    df = pd.DataFrame(result)

    df.to_csv(os.path.join(config.output_dir, "foldseek_result.csv"), index=False)
