from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import rasterio
import numpy as np
import os
import torch
from collections import Counter

class TIFFDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()  # Shape: (C, H, W)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)  # Single-channel mask

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

class FullModel(nn.Module):
    def __init__(self, encoder, segmentation_head, embed_dim):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.segmentation_head = segmentation_head
        self.embed_dim = embed_dim

    def forward(self, x):
        # Pass the input through the encoder
        features, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)
        features = features[:, 1:, :]  # Drop CLS token
        side_length = int(features.shape[1] ** 0.5)  # Assume square feature map
        reshaped_features = features.view(-1, side_length, side_length, self.embed_dim).permute(0, 3, 1, 2)
        return self.segmentation_head(reshaped_features)



def create_full_model(encoder, segmentation_head, embed_dim):
    """
    Factory function to create the FullModel.
    """
    return FullModel(encoder, segmentation_head, embed_dim)


def create_segmentation_head(embed_dim, num_classes):

    upscaling_block = lambda in_channels, out_channels: nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )
    embed_dims = [embed_dim // (2 ** i) for i in range(5)]
    segmentation_head = nn.Sequential(
        *[upscaling_block(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
        nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)
    )
    return segmentation_head

def calculate_class_weights(train_loader, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.float32)

    for _, masks in train_loader:
        masks = masks.numpy().flatten()  # Flatten to count pixels
        counts = Counter(masks)
        for class_idx, count in counts.items():
            class_counts[class_idx] += count

    print("Class Distribution:")
    for class_idx, count in enumerate(class_counts):
        print(f"Class {class_idx}: {count:.0f} pixels")

    # Compute inverse class frequencies
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)  # Normalize weights

    return torch.tensor(class_weights, dtype=torch.float32), class_counts

