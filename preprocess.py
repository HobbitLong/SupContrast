"""Load videos per object and sample high confidence frames. Split into train and test set and augment images. Upload back to s3."""

import argparse
import os
import shutil
import json

import numpy as np
import cv2
from tqdm import tqdm
import boto3
import onnxruntime as ort
import torchvision.transforms as T
from torchvision.transforms import Resize
from torchvision.io import read_image
from torchvision.utils import save_image

CONF_THRESH = 0.1
FRAME_DIFF = 4
RESIZE = 224

### Load arguments
def parse_args():
    """Parse inputs arguments."""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--local",
        action="store_true",
        help="whether to run locally or on sagemaker.",
    )

    parser.add_argument("--s3_bucket", type=str, default="brent-lab-sagemaker")

    parser.add_argument(
        "--s3_model_prefix",
        type=str,
        default="trained_models/general_objects/20230105.onnx",
    )

    parser.add_argument(
        "--s3_protocol_name",
        type=str,
        default="15objects_lab",
    )

    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="Datasets/OCN",
    )

    parser.add_argument(
        "--train_split_percentage",
        type=float,
        default=0.8,
    )

    return parser.parse_args()


def load_videos_from_s3(s3_bucket: str, protocol_name: str, data_dir: str):
    """Load videos from s3 bucket into local folder."""

    assert "/" not in protocol_name, "protocol_name cannot contain '/'"

    protocol_dir = os.path.join(data_dir, protocol_name, "input_vids")
    s3_prefix = f"Datasets/OCN/{protocol_name}/input_vids"

    # Create local folder if it doesn't exist
    if not os.path.exists(protocol_dir):
        os.makedirs(protocol_dir)

    # Load videos from s3
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket)

    assert (
        len(list(bucket.objects.filter(Prefix=s3_prefix))) > 0
    ), f"No files found in s3 bucket '{s3_bucket}' with prefix '{s3_prefix}'."

    print(
        f"Downloading files from s3 bucket '{s3_bucket}' with prefix '{s3_prefix}', saving to {protocol_dir}..."
    )
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        savepath = os.path.join(protocol_dir, obj.key.split(s3_prefix + "/")[1])
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
        if obj.key[-1] != "/":  # skip folders
            if not os.path.exists(savepath):
                bucket.download_file(obj.key, savepath)
            else:
                print(
                    f"{obj.key.split('/')[-1]} already exists locally. Skipping download."
                )

    print("Done.")


def load_model_from_s3(s3_bucket: str, s3_model_prefix: str, data_dir: str) -> str:
    """Load model from s3 bucket into local folder."""

    assert s3_model_prefix.endswith(".onnx"), "s3_model_prefix must end with '.onnx'"

    local_dir = os.path.join(data_dir, "models")
    file_path = os.path.join(local_dir, s3_model_prefix.split("/")[-1])

    # Create local folder if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Load model from s3
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket)

    assert (
        len(list(bucket.objects.filter(Prefix=s3_model_prefix))) > 0
    ), f"No model found in s3 bucket '{s3_bucket}' at prefix '{s3_model_prefix}'."

    print(f"Downloading model file {s3_model_prefix}...")

    if not os.path.exists(file_path):
        bucket.download_file(s3_model_prefix, file_path)
    else:
        print("Model already exists locally. Skipping download.")

    return file_path


def upload_samples_to_s3(s3_bucket: str, protocol_name: str, data_dir: str):
    """Upload sampled frames to protocol/sampled_frames folder in s3 bucket."""

    # Check inputs
    samples_folder = os.path.join(data_dir, protocol_name, "samples")
    s3_prefix = f"Datasets/OCN/{protocol_name}/sampled_frames_supcon"

    assert os.path.exists(samples_folder), f"No samples folder found in {data_dir}"
    assert len(os.listdir(samples_folder)) > 0, f"No samples found in {samples_folder}"

    # Upload to s3
    print(f"Uploading files to s3 bucket '{s3_bucket}' with prefix '{s3_prefix}'...")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket)

    for root, dirs, files in os.walk(samples_folder):
        for file in files:
            obj_name = os.path.split(root)[1]
            split = os.path.split(os.path.split(root)[0])[1]
            s3path = s3_prefix + "/" + split + "/" + obj_name + "/" + file

            filepath = os.path.join(root, file)
            if filepath.endswith(".jpg"):
                bucket.upload_file(filepath, s3path)
    print("Done.")


def _letterbox(
    im,
    new_shape=(1280, 1280),
    color=(114, 114, 114),
    auto=True,
    scaleup=True,
    stride=32,
):
    """Resize and pad image while meeting stride-multiple constraints."""
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def load_onnx(model_path: str, device: str = "GPU") -> ort.InferenceSession:
    """Load ONNX model.
    Args:
        model_path: path to ONNX model
        cpu: whether to use cpu for inference. Default is False (use GPU).
    Returns:
        ort.InferenceSession: ONNXRuntime session
    """
    # Check inputs
    assert os.path.isfile(model_path), f"Model file not found: {model_path}"
    print(f"Loading model from {model_path}...")

    # Load model in ONNXRuntime
    if device == "CPU":
        providers = ["CPUExecutionProvider"]
    else:
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)

    return session


def _inference_onnx(onnx_session, input: dict, outname: str):
    """Run inference using ONNX model on image.

    Args:
        onnx_session (ONNXRuntime Session): session of ONNX model
        input (dict): {inputname: image}
        outname (str): output name of ONNX model
    """
    return onnx_session.run(outname, input)


def _crop_bbox(
    img: np.ndarray, box: np.ndarray, dwdh: tuple, ratio: float
) -> np.ndarray:
    """Crop bounding box from image.

    Args:
        img: image
        bbox: bounding box [x1, y1, x2, y2]


    Returns:
        np.ndarray: cropped image
    """
    # Check inputs
    assert len(box) == 4, "bbox should have 4 coordinates]"

    # Get bbox coordinates
    box -= np.array(dwdh * 2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()

    # Get cropped image
    cropped_img = img[box[1] : box[3], box[0] : box[2]]

    return cropped_img


def _sample_frames_from_video(
    video_path: str,
    data_dir: str,
    onnx_session: ort.InferenceSession,
    protocol_name: str,
    object_name: str,
    splits: dict,
):
    """Apply object detection in ONNXRuntime and sample frames with high confidence detections.
    Save samples to data_dir/protocol/samples.
    """
    # Check inputs
    assert os.path.isfile(video_path), f"Video file not found: {video_path}"

    samples_dir = os.path.join(data_dir, protocol_name, "samples")

    # Load video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Video file not found: {video_path}"

    # Get input/output names for model
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Sample frames
    prev_framecount = -1 - FRAME_DIFF  # for sampling frames not too close to each other
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # for train/val split

    for frame_count in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if ret == False:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = img.copy()
        image, ratio, dwdh = _letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        output = _inference_onnx(
            onnx_session, {input_name: im}, [output_name]
        )  # batch_id,x0,y0,x1,y1,cls_id,score

        if prev_framecount + FRAME_DIFF < frame_count:
            if (len(output[0]) == 1) and (
                (output[0][0][6] > CONF_THRESH)
            ):  # Exactly one detection higher than threshold

                # Train/val/test split
                if frame_count < splits["train"] * total_frames:
                    split = "train"
                else:
                    split = "val"

                # Crop and save frame
                prev_framecount = frame_count

                if not os.path.exists(os.path.join(samples_dir, split, object_name)):
                    os.makedirs(os.path.join(samples_dir, split, object_name))

                frame_path = os.path.join(
                    samples_dir,
                    split,
                    object_name,
                    f"{os.path.basename(video_path)}_{frame_count}.jpg",
                )
                bbox = np.array(
                    [
                        detection if detection > 0 else 0
                        for detection in output[0][0][1:5]
                    ]
                )
                cropped_img = _crop_bbox(frame, bbox, dwdh, ratio)

                # Resize to 224x224
                cropped_img = cv2.resize(cropped_img, (224, 224))

                try:
                    cv2.imwrite(frame_path, cropped_img)
                except cv2.error:
                    print(f"skipped frame {frame_path}")

        frame_count += 1

    cap.release()


def sample_frames_from_videos(
    data_dir: str,
    protocol_name: str,
    onnxruntime_session: ort.InferenceSession,
    splits: dict,
):
    """
    Sample frames from all videos in data_dir. Save samples frames to data_dir/protocol_name/samples while splitting into train/val.
    """
    vid_dir = os.path.join(data_dir, protocol_name, "input_vids")

    # Create samples folder and remove old files
    samples_dir = os.path.join(data_dir, protocol_name, "samples")
    if os.path.exists(samples_dir):
        print("Removing old samples...")
        shutil.rmtree(samples_dir)

    os.makedirs(samples_dir)

    for object_folder in os.listdir(vid_dir):
        obj_name = object_folder.split(".")[0]
        for vid in os.listdir(os.path.join(vid_dir, object_folder)):
            video_path = os.path.join(vid_dir, object_folder, vid)

            if os.path.isfile(video_path):
                print(f"Sampling frames from {video_path}...")
                _sample_frames_from_video(
                    video_path,
                    data_dir,
                    onnxruntime_session,
                    protocol_name,
                    obj_name,
                    splits,
                )


def _augment_image(
    img_path: str,
):
    """Apply augmentations to image and save to same directory."""
    # Check inputs
    assert os.path.isfile(img_path), f"Image file not found: {img_path}"

    # Load image in torch
    img = read_image(img_path)

    # Apply augmentations
    augmenter = T.TrivialAugmentWide()
    for i in range(
        5
    ):  # Original paper only applied one time, but we don't have a lot of data...
        aug_img = augmenter(img)
        aug_img = aug_img / 255.0

        # Save image
        new_img_path = img_path.replace(".jpg", f"_aug{i}.jpg")
        save_image(aug_img, new_img_path)


def augment_data(data_dir: str, protocol_name: str):
    """Perform data augmentation on sampled images."""
    samples_dir = os.path.join(data_dir, protocol_name, "samples")
    print("Augmenting samples...")
    for split in ["train", "val"]:
        split_dir = os.path.join(samples_dir, split)
        for obj in os.listdir(split_dir):
            obj_dir = os.path.join(split_dir, obj)
            for img in os.listdir(obj_dir):
                img_path = os.path.join(obj_dir, img)
                if os.path.isfile(img_path):
                    _augment_image(img_path)


def preprocess_pipeline():
    """Preprocess pipeline."""
    args = parse_args()

    # Load vids from S3 (if doesn't exist in local)
    load_videos_from_s3(args.s3_bucket, args.s3_protocol_name, "data/input")

    # Load model from S3 (if doesn't exist in local)
    model_path = load_model_from_s3(args.s3_bucket, args.s3_model_prefix, "data/input")

    # Load model in ONNXRuntime
    onnxruntime_session = load_onnx(model_path)

    # Sample frames
    splits = {
        "train": args.train_split_percentage,
        "val": 1 - args.train_split_percentage,
    }
    sample_frames_from_videos(
        "data/input", args.s3_protocol_name, onnxruntime_session, splits
    )

    # Perform data augmentation
    # augment_data("data/input", args.s3_protocol_name)

    # Upload sampled frames to S3
    upload_samples_to_s3(args.s3_bucket, args.s3_protocol_name, "data/input")


if __name__ == "__main__":
    preprocess_pipeline()
