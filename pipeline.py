"""Train a supcon model given a folder, train a linear head on top of the model, evaluate the model on a test set, and save the model to a file.
Convert the full model to onnx using export.py, and then test the onnx model"""

import argparse
import yaml
import os
import shutil
from main_supcon import main as train_supcon
from main_linear import main as train_linear_head
from export import main as export_onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="path to config file"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/input/",
        help="path to the folder containing the protocol data",
    )

    return parser.parse_args()


def split_dataset(data_folder: str, test_size: float) -> None:
    """Split the dataset into train and test. This will create two new folders in the data folder, train and test, and copy the images into the appropriate folder.

    :param data_folder: Path to the data folder
    :test_size: Proportion of the data to use for the test set
    """
    print("Splitting dataset into train and test...")
    train_folder = os.path.join(data_folder, "train")
    test_folder = os.path.join(data_folder, "test")
    if not os.path.exists(train_folder) and not os.path.exists(test_folder):
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
    else:
        print("Train and test folders already exist. Skipping split.")
        return

    for folder in os.listdir(data_folder):
        if folder in ["train", "test"]:
            continue
        folder_path = os.path.join(data_folder, folder)
        len_folder = len(os.listdir(folder_path))
        for i, image in enumerate(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image)
            if i < len_folder * (1 - test_size):
                dest_folder = os.path.join(train_folder, folder)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                shutil.copy(image_path, dest_folder)
            else:
                dest_folder = os.path.join(test_folder, folder)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                shutil.copy(image_path, dest_folder)
    print("Done splitting dataset into train and test.")
    return


def pipeline():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["common"]["data_folder"] = args.data_folder + "/train"
    config["common"]["val_folder"] = args.data_folder + "/test"

    # Split the dataset into train and test
    split_dataset(config["common"]["data_folder"], config["common"]["test_size"])

    # train a supcon model
    train_supcon(config)

    # train a linear head on top of the model
    train_linear_head(config)

    # convert the full model to onnx
    export_onnx(config)


if __name__ == "__main__":
    pipeline()
