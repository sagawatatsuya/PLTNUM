import gc
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.append(".")
from utils import AverageMeter, get_logger, seed_everything, timeSince
from datasets import PLTNUMDataset, LSTMDataset
from models import PLTNUM, LSTMModel

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for protein half-life prediction."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="westlake-repl/SaProt_650M_AF2",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="SaProt",
        help="Model architecture: 'ESM2', 'SaProt', or 'LSTM'.",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use AMP for mixed precision training.",
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
        help="Maximum input sequence length. Two tokens are used fo <cls> and <eos> tokens. So the actual length of input sequence is max_length - 2. Padding or truncation is applied to make the length of input sequence equal to max_length.",
    )
    parser.add_argument(
        "--used_sequence",
        type=str,
        default="left",
        help="Which part of the sequence to use: 'left', 'right', 'both', or 'internal'.",
    )
    parser.add_argument(
        "--padding_side",
        type=str,
        default="right",
        help="Padding side: 'right' or 'left'.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.05,
        help="Ratio of mask tokens for augmentation.",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.2,
        help="Probability to apply mask augmentation",
    )
    parser.add_argument(
        "--random_delete_ratio",
        type=float,
        default=0.1,
        help="Ratio of deleting tokens in augmentation.",
    )
    parser.add_argument(
        "--random_delete_prob",
        type=float,
        default=-1,
        help="Probability to apply random delete augmentation.",
    )
    parser.add_argument(
        "--random_change_ratio",
        type=float,
        default=0,
        help="Ratio of changing tokens in augmentation.",
    )
    parser.add_argument(
        "--truncate_augmentation_prob",
        type=float,
        default=-1,
        help="Probability to apply truncate augmentation.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=300,
        help="Log print frequency.",
    )
    parser.add_argument(
        "--fleeze_layer",
        type=int,
        default=-1,
        help="Freeze layers of the model. -1 means no layers are frozen.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task type: 'classification' or 'regression'.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="Protein half-life average [h]",
        help="Column name of the target.",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default="aa_foldseek",
        help="Column name fot the input sequence.",
    )

    return parser.parse_args()


def train_fn(train_loader, model, criterion, optimizer, epoch, cfg):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    losses = AverageMeter()
    label_list, pred_list = [], []
    start = time.time()

    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        labels = (
            labels.float()
            if cfg.task == "classification"
            else labels.to(dtype=torch.half)
        )
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            y_preds = model(inputs)
        loss = criterion(y_preds, labels.view(-1, 1))
        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        label_list += labels.tolist()
        pred_list += y_preds.tolist()

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            if cfg.task == "classification":
                pred_list_new = (torch.Tensor(pred_list) > 0.5).to(dtype=torch.long)
                acc = accuracy_score(label_list, pred_list_new > 0.5)
                cfg.logger.info(
                    f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                    f"Elapsed {timeSince(start, float(step + 1) / len(train_loader))} "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                    f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                    f"Accuracy: {acc:.4f}"
                )
            elif cfg.task == "regression":
                r2 = r2_score(label_list, pred_list)
                cfg.logger.info(
                    f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                    f"Elapsed {timeSince(start, float(step + 1) / len(train_loader))} "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                    f"R2 Score: {r2:.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.8f}"
                )
    if cfg.task == "classification":
        pred_list_new = (torch.Tensor(pred_list) > 0.5).to(dtype=torch.long)
        acc = accuracy_score(label_list, pred_list_new)
        return losses.avg, acc
    elif cfg.task == "regression":
        return losses.avg, r2_score(label_list, pred_list)


def valid_fn(valid_loader, model, criterion, cfg):
    losses = AverageMeter()
    model.eval()
    label_list, pred_list = [], []
    start = time.time()

    for step, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        labels = (
            labels.float()
            if cfg.task == "classification"
            else labels.to(dtype=torch.half)
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                y_preds = (
                    torch.sigmoid(model(inputs))
                    if cfg.task == "classification"
                    else model(inputs)
                )
        loss = criterion(y_preds, labels.view(-1, 1))
        losses.update(loss.item(), labels.size(0))

        label_list += labels.tolist()
        pred_list += y_preds.tolist()

        if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
            if cfg.task == "classification":
                pred_list_new = (torch.Tensor(pred_list) > 0.5).to(dtype=torch.long)
                acc = accuracy_score(label_list, pred_list_new > 0.5)
                f1 = f1_score(label_list, pred_list_new, average="macro")
                cfg.logger.info(
                    f"EVAL: [{step}/{len(valid_loader)}] "
                    f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader))} "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                    f"Accuracy: {acc:.4f} "
                    f"F1 Score: {f1:.4f}"
                )
            elif cfg.task == "regression":
                r2 = r2_score(label_list, pred_list)
                cfg.logger.info(
                    f"EVAL: [{step}/{len(valid_loader)}] "
                    f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader))} "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                    f"R2 Score: {r2:.4f}"
                )

    if cfg.task == "classification":
        pred_list_new = (torch.Tensor(pred_list) > 0.5).to(dtype=torch.long)
        return (
            f1_score(label_list, pred_list_new, average="macro"),
            accuracy_score(label_list, pred_list_new),
            pred_list,
        )
    elif cfg.task == "regression":
        return losses.avg, r2_score(label_list, pred_list), np.array(pred_list)


def train_loop(folds, fold, cfg):
    cfg.logger.info(f"================== fold: {fold} training ======================")
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)

    if cfg.architecture in ["ESM2", "SaProt"]:
        train_dataset = PLTNUMDataset(cfg, train_folds, train=True)
        valid_dataset = PLTNUMDataset(cfg, valid_folds, train=False)
    elif cfg.architecture == "LSTM":
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

    if cfg.architecture in ["ESM2", "SaProt"]:
        model = PLTNUM(cfg)
        if cfg.fleeze_layer >= 0:
            for name, param in model.named_parameters():
                if f"model.encoder.layer.{cfg.fleeze_layer}" in name:
                    break
                param.requires_grad = False
        torch.save(model.config, os.path.join(cfg.output_dir, "config.pth"))
    elif cfg.architecture == "LSTM":
        model = LSTMModel(cfg)

    model.to(cfg.device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    if cfg.architecture in ["ESM2", "SaProt"]:
        scheduler = CosineAnnealingLR(
            optimizer,
            **{"T_max": 2, "eta_min": 1.0e-6, "last_epoch": -1},
        )
    elif cfg.architecture == "LSTM":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=cfg.epochs, num_cycles=0.5
        )

    criterion = nn.BCEWithLogitsLoss() if cfg.task == "classification" else nn.MSELoss()
    best_score = 0 if cfg.task == "classification" else float("inf")

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
                predictions,
                os.path.join(cfg.output_dir, f"predictions.pth"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(cfg.output_dir, f"model_fold{fold}.pth"),
            )

    predictions = torch.load(
        os.path.join(cfg.output_dir, f"predictions.pth"), map_location="cpu"
    )
    valid_folds["prediction"] = predictions
    cfg.logger.info(f"[Fold{fold}] Best score: {best_score}")
    torch.cuda.empty_cache()
    gc.collect()
    return valid_folds


def get_embedding(folds, fold, path):
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_dataset = PLTNUMDataset(config, valid_folds, train=False, is_test=True)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PLTNUM(config)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.to(device)

    model.eval()
    embedding_list = []
    for inputs, _ in valid_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                embedding = model.create_embedding(inputs)
        embedding_list += embedding.tolist()

    torch.cuda.empty_cache()
    gc.collect()
    return embedding_list


if __name__ == "__main__":
    config = parse_args()
    config.token_length = 2 if config.architecture == "SaProt" else 1
    config.device = device

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if config.used_sequence == "both":
        config.max_length += 1

    LOGGER = get_logger(os.path.join(config.output_dir, "output"))
    config.logger = LOGGER

    seed_everything(config.seed)

    train_df = (
        pd.read_csv(config.data_path)
        .drop_duplicates(subset=[config.sequence_col], keep="first")
        .reset_index(drop=True)
    )
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
    kf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )
    for fold, (trn_ind, val_ind) in enumerate(kf.split(train_df, train_df["class"])):
        train_df.loc[val_ind, "fold"] = int(fold)

    if config.architecture in ["ESM2", "SaProt"]:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, padding_side=config.padding_side
        )
        tokenizer.save_pretrained(config.output_dir)
        config.tokenizer = tokenizer

    oof_df = pd.DataFrame()
    for fold in range(config.n_folds):
        _oof_df = train_loop(train_df, fold, config)
        oof_df = pd.concat([oof_df, _oof_df], axis=0)

    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_pickle(os.path.join(config.output_dir, "oof_df.pkl"))
