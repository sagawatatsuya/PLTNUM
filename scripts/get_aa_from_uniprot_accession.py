import argparse
import multiprocessing as mp
from io import StringIO

import numpy as np
import pandas as pd
import requests as r
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the file that have a column cotaining uniprotid information.",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default="Sheet1",
        help="Name of the sheet to read. Default is Sheet1.",
    )
    parser.add_argument(
        "--uniprotid_column",
        type=str,
        help="Name of the column that have uniprotid information. Default is None.",
    )
    parser.add_argument(
        "--uniprotids_column",
        type=str,
        help="Name of the column that have uniprotids information. Default is None. The ids are expected to be separated by semi-colon, and the first id is used.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
        help="Number of processes to use.",
    )
    return parser.parse_args()


def fetch_sequence(row, cfg):
    try:
        baseURL = "http://www.uniprot.org/uniprot/"
        uniprot_id = row[cfg.uniprotid_column]
        URL = baseURL + uniprot_id + ".fasta"
        response = r.post(URL)
        Data = "".join(response.text)
        Seq = StringIO(Data)
        pSeq = list(SeqIO.parse(Seq, "fasta"))
        return str(pSeq[0].seq)
    except:
        return None


def process_rows(df_chunk, cfg):
    return [fetch_sequence(row, cfg) for idx, row in df_chunk.iterrows()]


if __name__ == "__main__":
    config = parse_args()

    if config.file_path.endswith(".xls"):
        df = pd.read_excel(
            config.file_path,
            sheet_name=config.sheet_name,
        )
    else:
        df = pd.read_csv(config.file_path)

    if config.uniprotid_column is None and config.uniprotids_column is None:
        raise ValueError(
            "Either uniprotid_column or uniprotids_column should be provided."
        )
    if config.uniprotids_column is not None:
        df = df.dropna(subset=[config.uniprotids_column]).reset_index(drop=True)
        # use the first id and ignore the subunit and domain information
        df["uniprotid"] = df[config.uniprotids_column].apply(
            lambda x: x.split(";")[0].split("-")[0]
        )
        config.uniprotid_column = "uniprotid"

    df_split = np.array_split(df, config.num_processes)

    with mp.Pool(processes=config.num_processes) as pool:
        results = pool.map(lambda x: process_rows(x, config), df_split)

    aas = [seq for result in results for seq in result]

    df["aa"] = aas
    df.to_csv(f"{config.file_path.split('.')[0]}_with_aa.csv", index=False)
