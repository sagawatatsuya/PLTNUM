import os
import glob
import sys
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer
import shap

sys.path.append("../")
from utils import seed_everything, save_pickle
from models import PLTNUM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate SHAP values with a pretrained protein half-life prediction model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data.",
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
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="The number of folds for prediction.",
    )
    parser.add_argument(
        "--do_cross_validation",
        action="store_true",
        default=False,
        help="Use cross validation for prediction. If True, you have to specify the 'data_path' that contanins fold information, 'folds' for the number of folds, and 'model_path' for the directory of the model weights.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to the model weight(s).",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum input sequence length. Two tokens are used fo <cls> and <eos> tokens. So the actual length of input sequence is max_length - 2. Padding or truncation is applied to make the length of input sequence equal to max_length.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory.",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default="aa_foldseek",
        help="Column name fot the input sequence.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=5000,
        help="Number of evaluations for SHAP values calculation.",
    )

    return parser.parse_args()


def calculate_shap_fn(texts, model, cfg):
    if len(texts) == 1:
        texts = texts[0]
    else:
        texts = texts.tolist()

    inputs = cfg.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
    )
    inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    return outputs


if __name__ == "__main__":
    config = parse_args()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    seed_everything(config.seed)

    df = pd.read_csv(config.data_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    config.tokenizer = tokenizer

    if config.do_cross_validation:
        model_weights = glob.glob(os.path.join(config.model_path, "/*.pth"))
        for fold in range(config.folds):
            model = PLTNUM(config).to(config.device)
            model_weight = [w for w in model_weights if f"fold{fold}.pth" in w]
            model.load_state_dict(torch.load(model_weight, map_location="cpu"))
            model.eval()
            df_fold = df[df["fold"] == fold].reset_index(drop=True)

            # build an explainer using a token masker
            explainer = shap.Explainer(lambda x: calculate_shap_fn(x, model, config), config.tokenizer)

            shap_values = explainer(
                df_fold[config.sequence_col].values.tolist(),
                batch_size=config.batch_size,
                max_evals=config.max_evals,
            )

            save_pickle(os.path.join(config.output_dir, f"shap_values_fold{fold}.pickle"), shap_values)
    else:
        model = PLTNUM(config).to(config.device)
        model.load_state_dict(torch.load(config.model_path, map_location="cpu"))
        model.eval()

        # build an explainer using a token masker
        explainer = shap.Explainer(lambda x: calculate_shap_fn(x, model, config), config.tokenizer)

        shap_values = explainer(
            df[config.sequence_col].values.tolist(),
            batch_size=config.batch_size,
            max_evals=config.max_evals,
        )

        save_pickle(
            os.path.join(config.output_dir, "shap_values.pickle"), shap_values
        )