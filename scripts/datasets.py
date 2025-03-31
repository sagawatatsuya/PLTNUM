import random

import numpy as np
import torch
from augmentation import (
    mask_augmentation,
    random_change_augmentation,
    random_delete_augmentation,
    truncate_augmentation,
)
from torch.utils.data import Dataset


def tokenize_input(cfg, text):
    inputs = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg.max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=False,
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def one_hot_encoding(aa, amino_acids, cfg):
    aa = aa[: cfg.max_length].ljust(cfg.max_length, " ")
    one_hot = np.zeros((len(aa), len(amino_acids)))
    for i, a in enumerate(aa):
        if a in amino_acids:
            one_hot[i, amino_acids.index(a)] = 1
    return one_hot


def one_hot_encode_input(text, cfg):
    inputs = one_hot_encoding(
        text,
        (
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            " ",
        ),
        cfg,
    )
    return torch.tensor(inputs, dtype=torch.float)


class PLTNUMDataset(Dataset):
    def __init__(self, cfg, df, train=True):
        self.df = df
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        aas = self._adjust_sequence_length(data[self.cfg.sequence_col])

        if self.train:
            aas = self._apply_augmentation(aas)

        aas = aas.replace("__", "<pad>")

        inputs = tokenize_input(self.cfg, aas)

        if "target" in data:
            return inputs, torch.tensor(data["target"], dtype=torch.float32)
        return inputs, np.nan

    def _adjust_sequence_length(self, aas):
        max_length = (self.cfg.max_length - 2) * self.cfg.token_length
        if len(aas) > max_length:
            if self.cfg.used_sequence == "left":
                return aas[:max_length]
            elif self.cfg.used_sequence == "right":
                return aas[-max_length:]
            elif self.cfg.used_sequence == "both":
                half_max_len = max_length // 2
                return aas[:half_max_len] + "__" + aas[-half_max_len:]
            elif self.cfg.used_sequence == "internal":
                offset = (len(aas) - max_length) // 2
                return aas[offset : offset + max_length]
        return aas

    def _apply_augmentation(self, aas):
        if self.cfg.random_change_ratio > 0:
            aas = random_change_augmentation(aas, self.cfg)
        if (
            random.random() <= self.cfg.random_delete_prob
        ) and self.cfg.random_delete_ratio > 0:
            aas = random_delete_augmentation(aas, self.cfg)
        if (random.random() <= self.cfg.mask_prob) and self.cfg.mask_ratio > 0:
            aas = mask_augmentation(aas, self.cfg)
        if random.random() <= self.cfg.truncate_augmentation_prob:
            aas = truncate_augmentation(aas, self.cfg)
        return aas


class LSTMDataset(Dataset):
    def __init__(self, cfg, df, train=True):
        self.df = df
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        aas = data[self.cfg.sequence_col]
        aas = self._adjust_sequence_length(aas)
        aas = aas.replace("__", "<pad>")

        inputs = one_hot_encode_input(aas, self.cfg)

        return inputs, torch.tensor(data["target"], dtype=torch.float32)

    def _adjust_sequence_length(self, aas):
        max_length = (self.cfg.max_length - 2) * self.cfg.token_length
        if len(aas) > max_length:
            if self.cfg.used_sequence == "left":
                return aas[:max_length]
            elif self.cfg.used_sequence == "right":
                return aas[-max_length:]
            elif self.cfg.used_sequence == "both":
                half_max_len = max_length // 2
                return aas[:half_max_len] + "__" + aas[-half_max_len:]
            elif self.cfg.used_sequence == "internal":
                offset = (len(aas) - max_length) // 2
                return aas[offset : offset + max_length]
        return aas
