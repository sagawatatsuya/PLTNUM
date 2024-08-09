import os
import glob
import sys
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer
import shap
import pickle

sys.path.append("../")
from utils import seed_everything
from models import PLTNUM


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
        "--folds",
        type=int,
        default=10,
        help="The number of folds for prediction.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="The path to a directory containing models.",
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
        "--output_dir",
        type=str,
        default="/home2/sagawa/protein-half-life-prediction/ver20_56_2/",
        help="Output directory.",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default="aa_foldseek",
        help="The column name of amino acid sequence.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=5000,
        help="The number of evaluations for shap values calculation.",
    )

    return parser.parse_args()

config = parse_args()

if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)

seed_everything(config.seed)

oof_df = pd.read_pickle(config.data_path)

# load a model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.model, padding_side=config.padding_side
)
config.tokenizer = tokenizer

# define a prediction function
def calculate_shap_fn(texts):
    if len(texts) == 1:
        texts = texts[0]
    else:
        texts = texts.tolist()

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=config.max_length)
    print(inputs.input_ids.shape)
    inputs = {k: v.device() for k, v in inputs.items()}
    with torch.no_grad():
        # with torch.cuda.amp.autocast(enabled=config.apex):
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    try:
        return outputs
    except:
        print(outputs)

model_weights = glob.glob(os.path.join(config.model_path, "/*.pth"))
for fold in range(config.folds):
    model = PLTNUM(config).device()
    model_weight = [w for w in model_weights if f"fold{fold}.pth" in w]
    model.load_state_dict(torch.load(model_weight, map_location='cpu'))
    model.eval()
    df = oof_df[oof_df['fold'] == fold].reset_index(drop=True)

    # build an explainer using a token masker
    explainer = shap.Explainer(calculate_shap_fn, config.tokenizer)

    # explain the model's predictions on two sentences
    shap_values = explainer(df[config.sequence_col].values.tolist(), batch_size=config.batch_size, max_evals=config.max_evals)

    pickle.dump(shap_values, open(os.path.join(config.output_dir, f"shap_values_fold{fold}.pickle"), "wb"))