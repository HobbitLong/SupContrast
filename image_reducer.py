import argparse
import pathlib
from PIL import Image
import os
from PIL import ImageFile
import time
import shutil
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 550


def main(
    root_dir: str,
    destination_dir: str,
):
    if root_dir[-1] != "/":
        root_dir = f"{root_dir}/"
    all_json = [path for path in Path(root_dir).rglob("*.json")]
    base_name = os.path.basename(root_dir)
    all_dirs = [
        os.path.join(
            destination_dir,
            os.path.join(base_name, os.path.dirname(file_path).split(root_dir)[1]),
        )
        for file_path in all_json
    ]
    if not os.path.exists(destination_dir):
        print(f"Making dir {destination_dir}.")
    print(
        f"Copying folder structure from {root_dir} to {destination_dir} and copying json files."
    )
    for json_file, dir_path in zip(all_json, all_dirs):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_file, dir_path)
    print("Copied folder structure and json files.")
    all_images = sorted(
        [path for path in Path(root_dir).rglob("*.jpg")] +
        [path for path in Path(root_dir).rglob("*.jpeg")]
    )
    image_mapper = {}
    for image_path in all_images:
        image_path = str(image_path)
        image_name = os.path.basename(image_path)
        if image_name in image_mapper:
            image_mapper[image_name].append(image_path.replace(image_name, ""))
        else:
            image_mapper[image_name] = [image_path.replace(image_name, "")]
    print("Resizing and copying images.")
    num_images = 0
    for image_name, dirs in image_mapper.items():
        num_images += 1
        base_dir = dirs.pop(0)
        new_dir = os.path.join(base_name, os.path.dirname(base_dir).split(root_dir)[1])
        image_path = os.path.join(destination_dir, new_dir, image_name)
        img = Image.open(os.path.join(base_dir, image_name))
        if img.size[0] >= 2000 or img.size[1] >= 2000:
            if img.size[0] > img.size[1]:
                percent = IMAGE_SIZE / float(img.size[0])
                hsize = int((float(img.size[1]) * float(percent)))
                wsize = IMAGE_SIZE
            else:
                percent = IMAGE_SIZE / float(img.size[1])
                wsize = int((float(img.size[0]) * float(percent)))
                hsize = IMAGE_SIZE
            img = img.resize((wsize, hsize)).convert("RGB")
            time.sleep(0.01)
        img.save(image_path, quality=100)
        for base_dir in dirs:
            base_dir = os.path.join(
                base_name, os.path.dirname(base_dir).split(root_dir)[1]
            )
            im_path = os.path.join(destination_dir, base_dir, image_name)
            shutil.copy2(image_path, im_path)
        if not num_images % 100:
            print(f"Print copied {num_images} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--destination_dir", type=str)
    args = parser.parse_args()
    main(
        root_dir=args.root_dir,
        destination_dir=args.destination_dir,
    )
