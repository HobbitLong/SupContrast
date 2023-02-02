"""Train a supcon model given a folder, train a linear head on top of the model, evaluate the model on a test set, and save the model to a file.
Convert the full model to onnx using export.py, and then test the onnx model"""

import argparse
import yaml
from main_supcon import main as train_supcon
from main_linear import main as train_linear_head
from export import main as export_onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="path to config file"
    )

    return parser.parse_args()


def pipeline():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # train a supcon model
    train_supcon(config)

    # train a linear head on top of the model
    train_linear_head(config)

    # convert the full model to onnx
    export_onnx(config)


if __name__ == "__main__":
    pipeline()
