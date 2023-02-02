"""Export the trained Supcon (+ classifier) model to ONNX format."""

import argparse
import os
from PIL import Image
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime as ort
from networks.resnet_big import SupCEResNet
from preprocess import load_onnx


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--supcon_path",
        help="Path to SupCon network weights",
        default="data/weights/supcon.pth",
        type=str,
    )
    parser.add_argument(
        "--clf_path",
        help="Path to classification head weights",
        default="data/weights/clf.pth",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the exported ONNX model",
        default="data/weights/supcon.onnx",
        type=str,
    )
    parser.add_argument("--data_folder", help="Path to the data directory", type=str)
    return parser.parse_args()


def load_supcon(args, device="cuda"):
    """Load the SupCon model and the classification head"""
    n_cls = len(os.listdir(args.data_folder))
    print("Number of classes: %d" % n_cls)
    model = SupCEResNet(name="resnet50", num_classes=n_cls)
    weights_encoder = torch.load(args.supcon_path, map_location="cpu")["model"]
    weights_clf = torch.load(args.clf_path, map_location="cpu")["model"]

    state_dict_encoder = {}
    for k, v in weights_encoder.items():
        k = k.replace("encoder.module.", "")
        state_dict_encoder[k] = v
    state_dict_encoder = {
        k: v for k, v in state_dict_encoder.items() if "head" not in k
    }

    state_dict_clf = {}
    for k, v in weights_clf.items():
        k = k.replace("fc.", "")
        state_dict_clf[k] = v

    model.encoder.load_state_dict(state_dict_encoder)
    model.fc.load_state_dict(state_dict_clf)
    model = model.to(device)

    return model


def export_onnx(model, output_path, dummy_input):
    """Export pytorch Supcon model to ONNX format."""
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def check_onnx(output_path):
    """Check whether exported ONNX model is well formed."""
    onnx_model = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model is well formed!")
    except onnx.checker.ValidationError as e:
        print("ONNX model is invalid: %s" % e)


def _to_numpy(tensor):
    """Convert a tensor to numpy array."""
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def check_onnx_runtime(output_path, model, x):
    """Check whether exported ONNX model is equivalent to PyTorch model.

    :param output_path: Path to the exported ONNX model
    :param model: PyTorch model
    :param x: Input tensor

    :return: None
    """
    ort_session = load_onnx(output_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: _to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    torch_out = model(x)
    np.testing.assert_allclose(
        _to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05
    )

    print("Predictions are equivalent between PyTorch and ONNX Runtime!")


def main(config=None):
    # Parser arguments
    print("Exporting to ONNX format...")
    args = parse_args()
    if config is not None:
        for k, v in config["common"].items():
            setattr(args, k, v)
        print(f'Running with config: {config["common"]}')

    # Load the SupCon model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. CUDA not found.")
    model = load_supcon(args, device=device)
    model.eval()

    # Export the model to ONNX format
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    export_onnx(model, args.output_path, dummy_input)

    # Check the exported model
    check_onnx(args.output_path)

    # Check the exported model with ONNX runtime
    check_onnx_runtime(args.output_path, model, dummy_input)


if __name__ == "__main__":
    main()
