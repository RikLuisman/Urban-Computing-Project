import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

INPUT_IMAGE_DIR = "DOTA Dataset/images"
INPUT_LABEL_DIR = "DOTA Dataset/labels"

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

def parse_label_file(label_path):
    """
    Parse DOTA annotation file to extract bounding boxes and their categories.

    Args:
        label_path (str): Path to the label file.

    Returns:
        list: A list of bounding boxes and categories.
              Format: [([x1, y1, x2, y2, x3, y3, x4, y4], category), ...]
    """
    bounding_boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines[2:]:  # Skip metadata lines
        parts = line.strip().split()
        bbox = list(map(int, parts[:8]))  # First 8 values are the bounding box
        original_category = parts[8]  # The 9th value is the category
        remapped_category = CATEGORY_MAPPING.get(original_category, "unknown")
        bounding_boxes.append((bbox, remapped_category))
    return bounding_boxes


def draw_bounding_boxes(image, bounding_boxes):
    """
    Draw bounding boxes on the image.

    Args:
        image (np.ndarray): Image array.
        bounding_boxes (list): Bounding boxes with categories.

    Returns:
        np.ndarray: Annotated image.
    """
    annotated_image = image.copy()
    for bbox, category in bounding_boxes:
        points = np.array(bbox).reshape(-1, 2)  # Convert to (x, y) points
        cv2.polylines(annotated_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)  # Red lines
        # Draw category name
        text_position = tuple(points[0])  # Use the first point as the text position
        cv2.putText(annotated_image, category, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return annotated_image


def visualize_image(image_files, index):
    """
    Visualize the image and annotations at the given index in the image_files array.
    """
    if index < 0 or index >= len(image_files):
        print(f"Invalid index. Please select between 0 and {len(image_files) - 1}.")
        return

    image_name = image_files[index]
    image_path = os.path.join(INPUT_IMAGE_DIR, image_name)
    label_name = image_name.replace(".png", ".txt")
    label_path = os.path.join(INPUT_LABEL_DIR, label_name)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    if not os.path.exists(label_path):
        print(f"No annotations found for image: {image_name}")
        return

    bounding_boxes = parse_label_file(label_path)
    annotated_image = draw_bounding_boxes(image, bounding_boxes)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(f"Image: {image_name}")
    # plt.savefig(f"images_with_rectangles/{image_name}")
    plt.show()


if __name__ == "__main__":
    DOTA_images = sorted(os.listdir(INPUT_IMAGE_DIR))
    visualize_image(DOTA_images, 0)
