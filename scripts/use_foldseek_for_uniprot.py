import os
import random
import argparse
import pandas as pd
import multiprocessing as mp
from foldseek_util import get_struc_seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the file containing uniprotid information.",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default="Sheet1",
        help="Name of the sheet to read (for Excel files). Default is 'Sheet1'.",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default="pdb_files/UP000000589_10090_MOUSE_v4",
        help="Directory containing PDB files.",
    )
    parser.add_argument(
        "--uniprotid_column",
        type=str,
        help="Name of the column containing UniprotID information.",
    )
    parser.add_argument(
        "--uniprotids_column",
        type=str,
        help="Name of the column containing multiple UniprotIDs (separated by semicolons). The first ID will be used.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
        help="Number of processes to use for multiprocessing. Default is 2.",
    )
    return parser.parse_args()


def validate_columns(config, df):
    if config.uniprotid_column is None and config.uniprotids_column is None:
        raise ValueError("Either --uniprotid_column or --uniprotids_column must be provided.")
    if config.uniprotids_column:
        df = df.dropna(subset=[config.uniprotids_column]).reset_index(drop=True)
        df["uniprotid"] = df[config.uniprotids_column].apply(lambda x: x.split(";")[0].split("-")[0])
        config.uniprotid_column = "uniprotid"
    return df.dropna(subset=[config.uniprotid_column]).reset_index(drop=True)


def find_pdb_files(pdb_dir, uniprot_ids):
    pdf_files = os.listdir(pdb_dir)
    pdb_paths = []
    for uniprot_id in uniprot_ids:
        matches = [pdf_file for pdf_file in sorted(pdf_files) if uniprot_id in pdf_file]
        pdb_paths.append(matches[0] if matches else None)
    return pdb_paths


def get_foldseek_seq(pdb_path):
    parsed_seqs = get_struc_seq(
        "bin/foldseek",
        os.path.join(config.pdb_dir, pdb_path),
        ["A"],
        process_id=random.randint(0, 10000000),
    )["A"]
    return parsed_seqs


config = parse_args()

if config.file_path.endswith(".xls") or config.file_path.endswith(".xlsx"):
    df = pd.read_excel(
        config.file_path,
        sheet_name=config.sheet_name,
    )
else:
    df = pd.read_csv(config.file_path)
df = validate_columns(config, df)

df = df.dropna(subset=[config.uniprotid_column]).reset_index(drop=True)

uniprot_ids = df[config.uniprotid_column].tolist()
pdb_paths = find_pdb_files(config.pdb_dir, uniprot_ids)
df["pdb_path"] = pdb_paths
df = df.dropna(subset=["pdb_path"]).reset_index(drop=True)
df = df.drop_duplicates(subset=[config.uniprotid_column]).reset_index(drop=True)

with mp.Pool(config.num_processes) as pool:
    output = pool.map(get_foldseek_seq, df["pdb_path"].tolist())

aa, foldseek, aa_foldseek = zip(*output)

df["aa"] = aa
df["foldseek"] = foldseek
df["aa_foldseek"] = aa_foldseek
df.to_csv(f"{config.file_path.split('.')[0]}_foldseek.csv", index=False)
