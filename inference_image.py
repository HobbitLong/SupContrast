"""Run the model on an image"""
import argparse
import os
import time
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from networks.resnet_big import SupConResNet, LinearClassifier
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/input/15objects_lab/samples/test/Thrash",
        help="Path to image file",
    )
    parser.add_argument(
        "--n_cls",
        type=int,
        help="Number of unique classes in the dataset",
        default=14,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model architecture",
        default="resnet50",
    )
    parser.add_argument(
        "--supcon_path",
        type=str,
        default="data/weights/supcon.pth",
        help="Path to SupCon network weights",
    )
    parser.add_argument(
        "--clf_path",
        type=str,
        default="data/weights/clf.pth",
        help="Path to classification head weights",
    )

    return parser.parse_args()


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2675, 0.2565, 0.2761)
NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)

val_transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(size=(240, 320), scale=(0.99, 1), ratio=(0.99, 1)),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def load_models(args, device="cuda"):
    # Load model
    SupCon = SupConResNet(name=args.model)
    classifier = LinearClassifier(name=args.model, num_classes=args.n_cls)

    # Load supcon weights
    ckpt_supcon = torch.load(args.supcon_path, map_location="cpu")
    state_dict_supcon = ckpt_supcon["model"]

    new_state_dict_supcon = {}
    for k, v in state_dict_supcon.items():
        k = k.replace("module.", "")
        new_state_dict_supcon[k] = v

    state_dict_supcon = new_state_dict_supcon

    SupCon.load_state_dict(new_state_dict_supcon)

    # Load classification head weights
    ckpt_clf = torch.load(args.clf_path, map_location="cpu")
    state_dict_clf = ckpt_clf["model"]

    new_state_dict_clf = {}
    for k, v in state_dict_clf.items():
        # k = k.replace("fc.", "")
        new_state_dict_clf[k] = v

    state_dict_clf = new_state_dict_clf

    classifier.load_state_dict(state_dict_clf)

    SupCon = SupCon.to(device)
    classifier = classifier.to(device)

    return SupCon, classifier


if __name__ == "__main__":
    # Parser arguments
    args = parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    # Load model
    model, clf = load_models(args, device)

    # Load image
    for file in os.listdir(args.image_folder):
        file_path = os.path.join(args.image_folder, file)
        # Load image
        image = Image.open(file_path).convert("RGB")

        # Apply transforms
        image = val_transform(image)

        # Add batch dimension
        image = image.float().unsqueeze(0)

        # Move to device
        image = image.to(device)

        # Inference
        with torch.no_grad():
            model.eval()
            clf.eval()
            prediction = clf(model.encoder(image))
            _, pred = prediction.topk(1, 1, True, True)
            print(pred)
            # prediction = torch.softmax(prediction, dim=1)
            # prediction_class = torch.argmax(prediction, dim=1)

        # Print prediction
        print(prediction)
