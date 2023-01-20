import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from networks.resnet_big import SupCEResNet
import onnxruntime as ort
from preprocess import load_onnx


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
        required=True,
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


def load_models(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Supcon and Classifier Models
    model = SupCEResNet(args.model, args.n_cls)
    model.encoder.load_state_dict(torch.load(args.supcon_path, map_location=device))
    model.fc.load_state_dict(torch.load(args.clf_path, map_location=device))

    # Load Yolov7
    yolo_session = load_onnx(args.yolo_path)
    input_name = yolo_session.get_inputs()[0].name
    output_name = yolo_session.get_outputs()[0].name

    return model, yolo_session, input_name, output_name


def main():
    # Parser arguments
    args = parse_args()

    # Load the models
    model, session, input_name, output_name = load_models(args)

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), "Cannot capture source"

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to match the input size of the model
        frame = cv2.resize(frame, (1280, 1280))

        # Run the model on the frame and get the output
        output = session.run([output_name], {input_name: frame})[0]

        # Extract the bounding boxes and class labels from the output
        boxes, classes, scores = extract_bboxes(output, args.confidence)

        # Draw the bounding boxes on the frame
        for box, class_id, score in zip(boxes, classes, scores):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_id} {score:0.2f}",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Display the frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
