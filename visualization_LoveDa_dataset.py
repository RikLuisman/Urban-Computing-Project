import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import base64
import zlib
import math
from PIL import Image
import io

INPUT_IMAGE_DIR = "LoveDA Dataset/img"
INPUT_LABEL_DIR = "LoveDA Dataset/ann"

CLASS_COLOR_MAP = {
    'background': [255, 255, 255],
    'building': [255, 0, 0],
    'road': [255, 255, 0],
    'water': [0, 0, 255],
    'barren': [159, 129, 183],
    'forest': [0, 255, 0],
    'agriculture': [255, 195, 128]
}

CLASS_TITLE_TO_INDEX = {
    "background": 0,
    "building": 1,
    "road": 2,
    "water": 3,
    "barren": 4,
    "forest": 5,
    "agriculture": 6
}

def parse_loveda_annotation(annotation_path, target_tag="urban"):
    """
    Parse the LoveDA JSON annotation file and filter objects based on the tag.

    Args:
        annotation_path (str): Path to the JSON annotation file.
        target_tag (str): Filter annotations by this tag (e.g., "urban").

    Returns:
        np.ndarray: Binary mask for the annotations.
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    # Check if the image tag matches the desired type (e.g., urban)
    tags = annotation.get("tags", [])
    is_target = any(tag.get("name") == target_tag for tag in tags)
    if not is_target:
        return None

    # Get image size
    image_size = annotation.get("size", {})
    height, width = image_size.get("height", 0), image_size.get("width", 0)

    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each object in the annotation
    for obj in annotation.get("objects", []):
        class_title = obj.get("classTitle", "unknown")
        print(class_title)
        if class_title not in CLASS_TITLE_TO_INDEX:
            print(f"Warning: There is no class named '{class_title}' in the available classes list. Skipping...")
            continue

        class_index = CLASS_TITLE_TO_INDEX[class_title]
        bitmap = obj.get("bitmap", {})

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
            mask_region[patch_non_zero & (mask_region == 0)] = class_index

            # # Allowing overlapping
            # mask_region[patch_non_zero] = class_index

            mask[y_start:y_end, x_start:x_end] = mask_region

    return mask

def visualize_image(image_files, index, target_tag="urban"):
    """
    Visualize the image and its annotation mask at the given index in the image_files array.
    """
    if index < 0 or index >= len(image_files):
        print(f"Invalid index. Please select between 0 and {len(image_files) - 1}.")
        return

    image_name = image_files[index]
    image_path = os.path.join(INPUT_IMAGE_DIR, image_name)
    annotation_name = image_name + ".json"
    annotation_path = os.path.join(INPUT_LABEL_DIR, annotation_name)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    if not os.path.exists(annotation_path):
        print(f"No annotations found for image: {image_name}")
        return

    mask = parse_loveda_annotation(annotation_path, target_tag=target_tag)
    if mask is None:
        print(f"Image {image_name} does not match target tag '{target_tag}'.")
        return

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_name, color in CLASS_COLOR_MAP.items():
        class_idx = list(CLASS_COLOR_MAP.keys()).index(class_name)
        color_mask[mask == class_idx] = color

    color_mask_image = Image.fromarray(color_mask)

    overlayed_image = Image.blend(
        Image.fromarray(image),
        color_mask_image,
        alpha=0.3
    )

    print(f"I am showing you the visualization of image {image_name}")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title(f"Image")

    axes[1].imshow(color_mask_image)
    axes[1].axis("off")
    axes[1].set_title(f"Mask")


    axes[2].imshow(overlayed_image)
    axes[2].axis("off")
    axes[2].set_title(f"Image + Mask")

    plt.suptitle(f"Image: {image_name}",  fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    LoveDA_images = sorted(os.listdir(INPUT_IMAGE_DIR))
    visualize_image(LoveDA_images, 1003, target_tag="urban")