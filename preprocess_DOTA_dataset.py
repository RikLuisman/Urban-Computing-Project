import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

INPUT_IMAGE_DIR = "DOTA Dataset/images"
INPUT_LABEL_DIR = "DOTA Dataset/labels"
OUTPUT_IMAGE_DIR = "Preprocessed Datasets/DOTA/images"
OUTPUT_MASK_DIR = "Preprocessed Datasets/DOTA/masks"

CATEGORY_MAPPING = {
    "storage-tank": "building",
    "baseball-diamond": "green_space",
    "tennis-court": "green_space",
    "soccer-ball-field": "green_space",
    "swimming-pool": "blue_space",
    "roundabout": "road",
    "bridge": "road",
    "small-vehicle": "vehicle",
    "large-vehicle": "vehicle",
    "plane": "vehicle",
    "helicopter": "vehicle",
    "ground-track-field": "field",
    "basketball-court": "field",
    "ship": "on_water",
    "harbor": "on_water"
}

CLASS_MAPPING = {
    "building": 1,
    "green_space": 2,
    "blue_space": 3,
    "road": 4,
    "vehicle": 5,
    "field": 6,
    "on_water": 7
}

def parse_label_file_with_mapping(label_file_path, category_mapping):
    """Parse label file and remap categories based on the provided mapping."""
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    bounding_boxes = []
    for line in lines[2:]:  # Skip metadata lines
        parts = line.strip().split()
        bbox = list(map(int, parts[:8]))  # Polygon coordinates
        category = parts[8]
        if category in category_mapping:
            mapped_category = category_mapping[category]
            bounding_boxes.append((bbox, mapped_category))
        else: print(category)
    return bounding_boxes

def create_mask(image_shape, bounding_boxes, class_mapping):
    """Create a segmentation mask with remapped class values."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Grayscale mask
    for bbox, category in bounding_boxes:
        points = np.array(bbox).reshape(-1, 2)
        class_value = class_mapping[category]
        cv2.fillPoly(mask, [points], class_value)  # Fill polygon with class value
    return mask


if __name__ == "__main__":
    # image_names = os.listdir(INPUT_IMAGE_DIR)
    # image_path = os.path.join(INPUT_IMAGE_DIR, image_names[1200])
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # cropped_image = image[:224, :224]
    # print(cropped_image.shape)

    for image_name in os.listdir(INPUT_IMAGE_DIR):
        if not image_name.endswith(".png"):
            continue

        image_path = os.path.join(INPUT_IMAGE_DIR, image_name)
        label_name = image_name.replace(".png", ".txt")
        label_path = os.path.join(INPUT_LABEL_DIR, label_name)

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"Skipping {image_name}: not an RGB image (shape: {image.shape})")
            continue

        cropped_image = image[:224, :224, :]
        cropped_image = np.transpose(cropped_image, (2, 0, 1)) # rasterio wants the number of bands to be the first
        print(f"I am transforming image {image_name} into tiff format.")

        # Save image as TIF
        output_image_path = os.path.join(OUTPUT_IMAGE_DIR, image_name.replace(".png", ".tif"))
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

        # Process corresponding label
        if os.path.exists(label_path):
            bounding_boxes = parse_label_file_with_mapping(label_path, CATEGORY_MAPPING)
            mask = create_mask(image.shape, bounding_boxes, CLASS_MAPPING)

            cropped_mask = mask[:224, :224]
            # Save mask as TIF
            output_mask_path = os.path.join(OUTPUT_MASK_DIR, label_name.replace(".txt", ".tif"))
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
            print(f"No label found for {image_path}, skipping mask creation.")