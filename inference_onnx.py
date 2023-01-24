"""Perform inference on image(s) using ONNX runtime."""
import argparse
import os
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
from preprocess import load_onnx

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2675, 0.2565, 0.2761)
CLASSES = [
    "50mL Tube",
    "50mL Tube Rack",
    "5mL Syringe",
    "8 Channel Finnett Pipette",
    "96 Well Plate",
    "Eppendorf Repeater",
    "Micropipette",
    "Picogreen Buffer",
    "Picogreen Kit",
    "Pipette Tip Box",
    "Reservoir",
    "Styrofoam Tube Rack",
    "Thrash",
    "Vortexer",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        help="Path to image for test input",
        default="data/input/15objects_lab/samples/test/Reservior",
    )
    parser.add_argument(
        "--onnx_path",
        help="Path to exported SupCon Onnx model",
        default="data/weights/supcon.onnx",
    )
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """Load an image and preprocess to correct format.

    1. Convert to RGB
    2. Convert to float
    3. Transpose to (C, H, W)
    4. Normalize
    5. Add batch dimension

    :param image_path: Path to the image

    :returns: Preprocessed image as a numpy array (1, 3, 32, 32) / (B, C, H, W)
    """
    image = Image.open(image_path)  # (H, W, 3)
    image = image.resize((32, 32))  # (32, 32, 3)
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))  # (3, 32, 32)
    image = image / 255.0
    for i in range(3):
        image[i] = (image[i] - MEAN[i]) / STD[i]
    image = np.expand_dims(image, axis=0)  # (1, 3, 32, 32)

    return image


def load_images(image_folder_path: str) -> np.ndarray:
    """Load all images in a folder and preprocess to correct format.

    :param image_folder_path: Path to the folder containing images

    :returns: Array of preprocessed images (B, 3, 32, 32) / (B, C, H, W)
    """
    for image_path in os.listdir(image_folder_path):
        image = load_image(os.path.join(image_folder_path, image_path))
        if image_path == os.listdir(image_folder_path)[0]:
            images = image
        else:
            images = np.concatenate(
                (images, image), axis=0
            )  # (1, 3, 32, 32) to (B, 3, 32, 32)
    return images


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inference_onnx_images(onnx_path: str, image_path: str) -> list:
    """Inference on (an) image(s) using ONNX runtime."""
    ort_session = load_onnx(onnx_path)

    # Compute ONNX Runtime output prediction for all classes
    if os.path.isdir(image_path):
        image = load_images(image_path)
    else:
        image = load_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)  # (B, N_classes)

    # Get the top 1 prediction and its probability for each image
    predicted_classes = []
    now = time.time()
    for i in range(ort_outs[0].shape[0]):
        probas = _softmax(ort_outs[0][i])
        top_1 = np.argmax(ort_outs[0][i])
        predicted_class = CLASSES[top_1]
        predicted_classes.append(predicted_class)
        print(
            f"Predicted class/probability for image {i}: {predicted_class} -  {probas[top_1]:.3f}"
        )

    end = time.time()
    print(
        f"Time taken: {end - now:.3f} seconds for {len(predicted_classes)} images at {1 / (end - now):.3f} fps"
    )

    return predicted_classes


def main():
    args = parse_args()
    predicted_classes = inference_onnx_images(args.onnx_path, args.image_path)
    return predicted_classes


if __name__ == "__main__":
    main()
