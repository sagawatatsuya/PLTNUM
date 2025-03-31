import argparse
import os
import shutil
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import PLTNUM


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert the model implemented with nn.Module to a model implemented with transformers' PreTrainedModel."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to a model weight which you want to convert.",
    )
    parser.add_argument(
        "--config_and_tokenizer_path",
        type=str,
        help="The path to a config and tokenizer of the model which you want to convert.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The name of the base model of the finetuned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the prediction.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
    )

    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model = PLTNUM(config)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))

    torch.save(model.state_dict(), os.path.join(config.output_dir, "pytorch_model.bin"))
    for filename in [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ]:
        shutil.copy(
            os.path.join(config.config_and_tokenizer_path, filename),
            os.path.join(config.output_dir, filename),
        )
