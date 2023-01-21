import argparse
import os
from PIL import Image
import time
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from networks.resnet_big import SupConResNet, LinearClassifier
import onnxruntime as ort
from preprocess import (
    _letterbox,
    load_onnx,
    _inference_onnx,
    _crop_bbox,
)

OBJECT_CLASSES = [
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
    "Syringe",
    "Thrash",
    "Vortexer",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/videos/eval.mp4",
        help="Path to video file",
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
        "--yolo_path",
        type=str,
        default="data/weights/yolov7.onnx",
        help="path to Yolov7 network weights (ONNX format)",
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
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/output/",
        help="Folder path for output video",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.2,
        help="Confidence threshold for object localization",
    )

    return parser.parse_args()


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2675, 0.2565, 0.2761)
NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)

transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def extract_bboxes(output, confidence):
    # Extract the bounding boxes, class labels, and scores from the model output
    boxes = output[0].reshape(-1, 4)
    classes = output[1].reshape(-1)
    scores = output[2].reshape(-1)

    # Keep only boxes with scores higher than the specified confidence threshold
    mask = scores > confidence
    boxes = boxes[mask]
    classes = classes[mask]
    scores = scores[mask]

    return boxes, classes, scores


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

    # Load Yolov7
    yolo_session = load_onnx(args.yolo_path)
    input_name = yolo_session.get_inputs()[0].name
    output_name = yolo_session.get_outputs()[0].name

    return SupCon, classifier, yolo_session, input_name, output_name


def main():
    # Parser arguments
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the models
    supcon, clf, yolo_session, input_name, output_name = load_models(args, device)
    supcon.eval()
    clf.eval()

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), "Cannot capture source"

    # Create output video
    out = cv2.VideoWriter(
        "data/output/out.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    print(f"Processing at {int(cap.get(cv2.CAP_PROP_FPS))} fps")

    # Run models on video
    since = time.time()
    for frame_num in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # Read frame
        ret, frame = cap.read()
        assert ret, f"Could not read frame {frame_num} from video"

        # Run Yolov7
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = _letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        output = _inference_onnx(yolo_session, {input_name: im}, [output_name])

        detected_bboxes = []
        detected_object_names = []

        if len(output[0]) > 0:
            # Run OCN and classification head on cropped objects
            for detection in output[0]:
                if detection[6] > args.confidence:
                    # Crop object
                    bbox = np.array(
                        [
                            detection if detection > 0 else 0
                            for detection in detection[1:5]
                        ]
                    )  # Remove negative values
                    obj = _crop_bbox(frame, bbox, dwdh, ratio)

                    # Run combined model on cropped object
                    if len(obj) > 0:
                        detected_bboxes.append(bbox)
                        obj = cv2.cvtColor(obj, cv2.COLOR_RGB2BGR)
                        # obj = obj.transpose((2, 0, 1))  # HWC to CHW
                        obj = Image.fromarray(obj)
                        # Convert to torch tensor and normalize
                        obj = transforms(obj)
                        obj = obj.float().unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = clf(supcon.encoder(obj))
                        output = output.detach().cpu().numpy()
                        # Get class with highest probability and its name
                        output = OBJECT_CLASSES[int(np.argmax(output, axis=1))]
                        detected_object_names.append(output)

            # Draw bounding boxes and labels on frame
            for bbox, obj in zip(detected_bboxes, detected_object_names):
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    obj,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

        #

        # save video
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    time_elapsed = time.time() - since
    print(
        f"Processing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s at {int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / time_elapsed:.2f} fps"
    )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
