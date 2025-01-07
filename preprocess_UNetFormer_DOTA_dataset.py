import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
import random

#input and output directories for GeoSeg
INPUT_IMAGE_DIR = "DOTA Dataset/images"
INPUT_LABEL_DIR = "DOTA Dataset/labels"
OUTPUT_TRAIN_IMAGE_DIR = "Preprocessed Datasets UNetFormer/DOTA/train/images"
OUTPUT_TRAIN_MASK_DIR = "Preprocessed Datasets UNetFormer/DOTA/train/masks"
OUTPUT_VAL_IMAGE_DIR = "Preprocessed Datasets UNetFormer/DOTA/val/images"
OUTPUT_VAL_MASK_DIR = "Preprocessed Datasets UNetFormer/DOTA/val/masks"
OUTPUT_TEST_IMAGE_DIR = "Preprocessed Datasets UNetFormer/DOTA/test/images"
OUTPUT_TEST_MASK_DIR = "Preprocessed Datasets UNetFormer/DOTA/test/masks"


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

IMAGE_SIZE = 512  #target crop size for GeoSeg

def parse_label_file_with_mapping(label_file_path, category_mapping):
    """Parse label file and remap categories based on the provided mapping."""
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    bounding_boxes = []
    for line in lines[2:]:  #skip metadata lines
        parts = line.strip().split()
        bbox = list(map(int, parts[:8]))  #polygon coordinates
        category = parts[8]
        if category in category_mapping:
            mapped_category = category_mapping[category]
            bounding_boxes.append((bbox, mapped_category))
        else:
            print(category)
    return bounding_boxes

def create_mask(image_shape, bounding_boxes, class_mapping):
    """Create a segmentation mask with remapped class values."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  #grayscale mask
    for bbox, category in bounding_boxes:
        points = np.array(bbox).reshape(-1, 2)
        class_value = class_mapping[category]
        cv2.fillPoly(mask, [points], class_value)  #fill polygon with class value
    return mask

def prepare_output_dirs():
    """Ensure output directories exist."""
    os.makedirs(OUTPUT_TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VAL_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VAL_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_MASK_DIR, exist_ok=True)

if __name__ == "__main__":
    prepare_output_dirs()

    image_names = [name for name in os.listdir(INPUT_IMAGE_DIR) if name.endswith(".png")]

    random.seed(42)
    random.shuffle(image_names)

    train_ratio = 0.8
    val_ratio = 0.1

    train_split_index = int(len(image_names) * train_ratio)
    val_split_index = int(len(image_names) * (train_ratio + val_ratio))

    train_images = image_names[:train_split_index]
    val_images = image_names[train_split_index:val_split_index]
    test_images = image_names[val_split_index:]

    print(f"There are {len(train_images)} train images.")
    print(f"There are {len(val_images)} validation images.")
    print(f"There are {len(test_images)} test images.")

    for split, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        if split == "train":
            output_image_dir = OUTPUT_TRAIN_IMAGE_DIR
            output_mask_dir = OUTPUT_TRAIN_MASK_DIR
        elif split == "val":
            output_image_dir = OUTPUT_VAL_IMAGE_DIR
            output_mask_dir = OUTPUT_VAL_MASK_DIR
        else:
            output_image_dir = OUTPUT_TEST_IMAGE_DIR
            output_mask_dir = OUTPUT_TEST_MASK_DIR

        for image_name in images:
            if not image_name.endswith(".png"):
                continue

            image_path = os.path.join(INPUT_IMAGE_DIR, image_name)
            label_name = image_name.replace(".png", ".txt")
            label_path = os.path.join(INPUT_LABEL_DIR, label_name)

            #load image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"Skipping {image_name}: not an RGB image (shape: {image.shape})")
                continue

            cropped_image = image[:IMAGE_SIZE, :IMAGE_SIZE, :]
            cropped_image = np.transpose(cropped_image, (2, 0, 1))  
            print(f"Processing image {image_name} into TIFF format.")

            #save image as TIF
            output_image_path = os.path.join(output_image_dir, image_name.replace(".png", ".tif"))
            transform = from_origin(0, 0, 1, 1)
            with rasterio.open(
                    output_image_path,
                    'w',
                    driver='GTiff',
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    count=cropped_image.shape[0],
                    dtype=cropped_image.dtype,
                    transform=transform,
                    crs="EPSG:4326"
            ) as dst:
                dst.write(cropped_image)

            #process corresponding label
            if os.path.exists(label_path):
                bounding_boxes = parse_label_file_with_mapping(label_path, CATEGORY_MAPPING)
                mask = create_mask(image.shape, bounding_boxes, CLASS_MAPPING)

                cropped_mask = mask[:IMAGE_SIZE, :IMAGE_SIZE]
                #save mask as TIF
                output_mask_path = os.path.join(output_mask_dir, label_name.replace(".txt", ".tif"))
                with rasterio.open(
                        output_mask_path,
                        'w',
                        driver='GTiff',
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                        count=1,
                        dtype=cropped_mask.dtype,
                        transform=transform,
                        crs="EPSG:4326"
                ) as dst:
                    dst.write(cropped_mask, 1)
            else:
                print(f"No label found for {image_path}, skipping mask creation.")
