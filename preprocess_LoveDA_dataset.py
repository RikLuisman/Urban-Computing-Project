import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
import random
import json
import base64
from PIL import Image
import io
import zlib

TAG = "urban"

INPUT_IMAGE_DIR = "LoveDa Dataset/img"
INPUT_LABEL_DIR = "LoveDa Dataset/ann"
OUTPUT_TRAIN_IMAGE_DIR = f"Preprocessed Datasets/LoveDa/{TAG}/train/images"
OUTPUT_TRAIN_MASK_DIR = f"Preprocessed Datasets/LoveDa/{TAG}/train/masks"
OUTPUT_TEST_IMAGE_DIR = f"Preprocessed Datasets/LoveDa/{TAG}/test/images"
OUTPUT_TEST_MASK_DIR = f"Preprocessed Datasets/LoveDa/{TAG}/test/masks"

CLASS_MAPPING = {
    "background": 0,
    "building": 1,
    "road": 2,
    "water": 3,
    "forest": 4,
    "agriculture": 5,
    "barren": 6
}

def create_mask(annotation_path, class_mapping):
    """Create segmentation mask from JSON annotations."""
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    image_size = annotation.get("size", {})
    height, width = image_size.get("height", 0), image_size.get("width", 0)

    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in annotation.get("objects", []):
        class_title = obj["classTitle"]
        if class_title in class_mapping:
            class_value = class_mapping[class_title]
            bitmap = obj["bitmap"]

            if "data" in bitmap and "origin" in bitmap:

                origin = bitmap["origin"]
                data = bitmap["data"]

                decoded_data = base64.b64decode(data)
                decompressed_data = zlib.decompress(decoded_data)
                image = Image.open(io.BytesIO(decompressed_data)).convert("L")

                patch = np.array(image)

                x_start, y_start = origin
                y_end = min(y_start + patch.shape[0], height)
                x_end = min(x_start + patch.shape[1], width)

                patch = patch[:y_end - y_start, :x_end - x_start]

                mask_region = mask[y_start:y_end, x_start:x_end]
                patch_non_zero = patch > 0

                # No overlapping considered
                mask_region[patch_non_zero & (mask_region == 0)] = class_value

                # # Allowing overlapping
                # mask_region[patch_non_zero] = class_index

                mask[y_start:y_end, x_start:x_end] = mask_region

        else:
            print(f"No mapping for class {class_title}")
    return mask


def filter_urban_images(input_image_dir, input_label_dir, tag_type):
    """Filter out images with the 'urban' tag from the dataset."""
    urban_images = []
    for image_name in os.listdir(input_image_dir):
        if not image_name.endswith(".png"):
            continue
        annotation_name = image_name + ".json"
        annotation_path = os.path.join(input_label_dir, annotation_name)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
                tags = annotation.get("tags", [])
                if any(tag.get("name") == tag_type for tag in tags):
                    urban_images.append(image_name)
    return urban_images


def prepare_output_dirs():
    """Ensure output directories exist."""
    os.makedirs(OUTPUT_TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_MASK_DIR, exist_ok=True)


if __name__ == "__main__":
    prepare_output_dirs()

    # If we want only one class: urban / rural
    image_names = filter_urban_images(INPUT_IMAGE_DIR, INPUT_LABEL_DIR, TAG)
    print(f"There are {len(image_names)} {TAG} images")

    # # If we want all the images, no matter the class:
    # image_names = [name for name in os.listdir(INPUT_IMAGE_DIR) if name.endswith(".png")]
    # print(f"There are {len(image_names)} images in total")

    random.seed(42)
    random.shuffle(image_names)

    split_index = int(len(image_names) * 0.9)
    train_images = image_names[:split_index]
    test_images = image_names[split_index:]

    for split, images in [("train", train_images), ("test", test_images)]:
        output_image_dir = OUTPUT_TRAIN_IMAGE_DIR if split == "train" else OUTPUT_TEST_IMAGE_DIR
        output_mask_dir = OUTPUT_TRAIN_MASK_DIR if split == "train" else OUTPUT_TEST_MASK_DIR

        for image_name in images:
            if not image_name.endswith(".png"):
                continue
            image_path = os.path.join(INPUT_IMAGE_DIR, image_name)
            annotation_name = image_name + ".json"
            annotation_path = os.path.join(INPUT_LABEL_DIR, annotation_name)

            # Load the image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None or len(image.shape) != 3 or image.shape[2] != 3:
                print(f"Skipping {image_name}: not an RGB image.")
                continue

            cropped_image = image[:224, :224, :]
            cropped_image = np.transpose(cropped_image, (2, 0, 1))
            print(f"I am transforming image {image_name} into tiff format.")

            # Save image as TIF
            output_image_path = os.path.join(output_image_dir, image_name.replace(".png", ".tif"))
            transform = from_origin(0, 0, 1, 1)
            with rasterio.open(
                    output_image_path,
                    'w',
                    driver='GTiff',
                    height=224,
                    width=224,
                    count=cropped_image.shape[0],
                    dtype=cropped_image.dtype,
                    transform=transform,
                    crs="EPSG:4326"
            ) as dst:
                dst.write(cropped_image)

            # Process corresponding annotation
            if os.path.exists(annotation_path):
                mask = create_mask(annotation_path, CLASS_MAPPING)

                cropped_mask = mask[:224, :224]
                # Save mask as TIF
                output_mask_path = os.path.join(output_mask_dir, image_name.replace(".png", ".tif"))
                with rasterio.open(
                        output_mask_path,
                        'w',
                        driver='GTiff',
                        height=224,
                        width=224,
                        count=1,
                        dtype=cropped_mask.dtype,
                        transform=transform,
                        crs="EPSG:4326"
                ) as dst:
                    dst.write(cropped_mask, 1)
            else:
                print(f"No annotation found for {image_name}, skipping mask creation.")
