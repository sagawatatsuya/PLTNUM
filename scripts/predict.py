import gc
import os
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(".")
from utils import seed_everything
from models import PLTNUM
from datasets import PLTNUMDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to data used for prediction.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="westlake-repl/SaProt_650M_AF2",
        help="The name of a pretrained model or path to a model which you want to use for training. You can use your local models or models uploaded to hugging face.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="SaProt",
        help="The name of a model architecture. 'ESM2', 'SaProt' or 'LSTM'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to a model which you want to use for prediction.",
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
        "--padding_side",
        type=str,
        default="right",
        help="Padding side. 'right' or 'left'.",
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
        "--sequence_col",
        type=str,
        default="aa_foldseek",
        help="The column name of amino acid sequence.",
    )

    return parser.parse_args()


config = parse_args()
config.token_length = 2 if config.architecture == "SaProt" else 1

if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)


if config.used_sequence == "both":
    config.max_length = config.max_length + 1


seed_everything(config.seed)

train_df = pd.read_csv(config.data_path)

tokenizer = AutoTokenizer.from_pretrained(
    config.model, padding_side=config.padding_side
)
config.tokenizer = tokenizer


def predict_fn(valid_loader, model, device):
    model.eval()
    pred_list = []
    for inputs in valid_loader:
        inputs = inputs.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                if config.task == "classification":
                    y_preds = torch.sigmoid(model(inputs))
                elif config.task == "regression":
                    y_preds = model(inputs)
        # if config.task == 'classification':
        #     y_preds = torch.Tensor(y_preds)
        #     y_preds = (y_preds > 0.5).to(dtype=torch.long)

        pred_list += y_preds.tolist()
    return pred_list


def predict(folds, model_path):
    valid_dataset = PLTNUMDataset(config, folds, train=False, is_test=True)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PLTNUM(config)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    predictions = predict_fn(valid_loader, model, device)

    folds["prediction"] = predictions
    torch.cuda.empty_cache()
    gc.collect()
    return folds


result = predict(train_df, config.model_path)
result.to_csv(os.path.join(config.output_dir, "result.csv"), index=False)
