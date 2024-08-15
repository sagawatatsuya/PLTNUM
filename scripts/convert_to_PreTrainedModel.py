import os
import sys
import argparse

import torch
from transformers import AutoTokenizer, AutoConfig

# Append the utils module path
sys.path.append("../")
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
        "--tokenizer_and_config_name_or_path",
        type=str,
        help="The path to a tokenizer of the model which you want to convert.",
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

    config.tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_and_config_name_or_path, return_tensors="pt"
    )

    model = PLTNUM(config, config_path=os.path.join(config.tokenizer_and_config_name_or_path, "config.pth"), pretrained=False)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))

    model_config = AutoConfig.from_pretrained(config.model)
    config.vocab_size = len(config.tokenizer)

    config.tokenizer.save_pretrained(config.output_dir)
    torch.save(model.state_dict(), os.path.join(config.output_dir, "pytorch_model.bin"))
    model_config.save_pretrained(config.output_dir)