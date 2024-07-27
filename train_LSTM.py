import gc
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.append("../")
from utils import get_logger, seed_everything
from datasets import LSTMDataset
from models import LSTMModel
from train import train_fn, valid_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     required=True,
    #     help="The path to data used for training. CSV file that contains ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT'] columns is expected. If there are missing values, please fill them with ' '."
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="westlake-repl/SaProt_650M_AF2",
        help="The name of a pretrained model or path to a model which you want to use for training. You can use your local models or models uploaded to hugging face.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed for reproducibility.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use amp for mixed precision training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max length of input sequence. Two tokens are used fo <cls> and <eos> tokens. So the actual length of input sequence is max_length - 2. Padding or truncation is applied to make the length of input sequence equal to max_length.",
    )
    parser.add_argument(
        "--used_sequence",
        type=str,
        default="left",
        help="How to use input sequence. 'left': use the left part of the sequence, 'right': use the right part of the sequence, 'both': use both side of the sequence, 'internal': use the internal part of the sequence.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds for cross validation.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=300,
        help="Frequency of printing log.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home2/sagawa/protein-half-life-prediction/ver20_56_2/",
        help="Output directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task. 'classification' or 'regression'.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="Protein half-life average [h]",
        help="The column name of protein half-life.",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default="aa_foldseek",
        help="The column name of amino acid sequence.",
    )

    return parser.parse_args()


config = parse_args()
config.token_length = 2 if "SaProt" in config.model else 1
config.device = device

if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)


if config.used_sequence == "both":
    config.max_length = config.max_length + 1


config.logger = get_logger(os.path.join(config.output_dir, "output"))
seed_everything(config.seed)


up = set(
    [
        i.split("AF-")[1].split("-")[0]
        for i in os.listdir(
            "/home2/sagawa/protein-half-life-prediction/PHLprediction/UP000000589_10090_MOUSE_v4"
        )
    ]
)
ids = up

train_df = pd.read_csv(
    "/home2/sagawa/protein-half-life-prediction/ver20_56_2/41586_2011_BFnature10098_MOESM304_ESM_with_aa.csv"
)
train_df.dropna(subset=["Uniprot IDs"], inplace=True)
train_df["Uniprot ID"] = train_df["Uniprot IDs"].apply(
    lambda x: x.split(";")[0].split("-")[0]
)
train_df["exist"] = train_df["Uniprot ID"].apply(lambda x: x in ids)

up = pd.read_csv(
    "/home2/sagawa/protein-half-life-prediction/PHLprediction/uniprot_mouse_filepath_and_seq_and_foldseek.csv"
)
up["id"] = up["pdb_path"].apply(lambda x: x.split("AF-")[1].split("-")[0])
train_df = train_df[train_df["exist"] == True].reset_index(drop=True)
train_df = train_df.merge(
    up[["id", "aa_foldseek"]], left_on="Uniprot ID", right_on="id", how="left"
)
train_df["aa"] = train_df["aa_foldseek"].apply(lambda x: x[::2])

train_df["aa"] = train_df[config.sequence_col]

train_df = train_df.drop_duplicates(subset=["aa"], keep="first").reset_index(drop=True)


train_df["T1/2 [h]"] = train_df[config.target_col]

if config.task == "classification":
    train_df["target"] = (
        train_df["T1/2 [h]"] > np.median(train_df["T1/2 [h]"])
    ).astype(int)
    train_df["class"] = train_df["target"]

elif config.task == "regression":
    train_df["log1p(T1/2 [h])"] = np.log1p(train_df["T1/2 [h]"])
    train_df["log1p(T1/2 [h])"] = (
        train_df["log1p(T1/2 [h])"] - min(train_df["log1p(T1/2 [h])"])
    ) / (max(train_df["log1p(T1/2 [h])"]) - min(train_df["log1p(T1/2 [h])"]))
    train_df["target"] = train_df["log1p(T1/2 [h])"]

    def get_class(row, class_num=5):
        denom = 1 / class_num
        num = row["log1p(T1/2 [h])"]
        for target in range(class_num):
            if denom * target <= num and num < denom * (target + 1):
                break
        row["class"] = target
        return row

    train_df = train_df.apply(get_class, axis=1)

train_df["fold"] = -1
kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
for fold, (trn_ind, val_ind) in enumerate(kf.split(train_df, train_df["class"])):
    train_df.loc[val_ind, "fold"] = int(fold)


def train_loop(folds, fold, cfg):
    cfg.logger.info(f"================== fold: {fold} training ======================")
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)

    train_dataset = LSTMDataset(cfg, train_folds, train=True)
    valid_dataset = LSTMDataset(cfg, valid_folds, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = LSTMModel(cfg)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=cfg.epochs, num_cycles=0.5
    )
    # use f1_score for classification, mseloss for regression
    if cfg.task == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        best_score = 0
    elif cfg.task == "regression":
        criterion = nn.MSELoss(reduction="mean")
        best_score = float("inf")

    for epoch in range(cfg.epochs):
        start_time = time.time()
        # train
        avg_loss, train_score = train_fn(
            train_loader, model, criterion, optimizer, epoch, cfg
        )
        scheduler.step()
        # eval
        val_score, val_score2, predictions = valid_fn(
            valid_loader, model, criterion, cfg
        )

        elapsed = time.time() - start_time

        if cfg.task == "classification":
            cfg.logger.info(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  train_acc: {train_score:.4f}  valid_acc: {val_score2:.4f}  valid_f1: {val_score:.4f}  time: {elapsed:.0f}s"
            )
        elif cfg.task == "regression":
            cfg.logger.info(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  train_r2: {train_score:.4f}  valid_r2: {val_score2:.4f}  valid_loss: {val_score:.4f}  time: {elapsed:.0f}s"
            )

        if (cfg.task == "classification" and best_score < val_score) or (
            cfg.task == "regression" and best_score > val_score
        ):
            best_score = val_score
            cfg.logger.info(f"Epoch {epoch+1} - Save Best Score: {val_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                os.path.join(cfg.output_dir, f"model_fold{fold}.pth"),
            )
    predictions = torch.load(
        os.path.join(cfg.output_dir, f"model_fold{fold}.pth"), map_location="cpu"
    )["predictions"]
    valid_folds["prediction"] = predictions
    cfg.logger.info(f"[Fold{fold}] Best score: {best_score}")
    torch.cuda.empty_cache()
    gc.collect()
    return valid_folds


oof_df = pd.DataFrame()
for fold in range(config.n_folds):
    _oof_df = train_loop(train_df, fold, config)
    oof_df = pd.concat([oof_df, _oof_df])
oof_df = oof_df.reset_index(drop=True)

oof_df.to_pickle(os.path.join(config.output_dir, "oof_df.pkl"))
