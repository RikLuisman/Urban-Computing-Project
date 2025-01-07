import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from tqdm import tqdm
from UNetFormer import UNetFormer
from collections import Counter

#dataset class for GeoSeg-compatible input
class GeoSegTIFFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.transform = transform
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #load the image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()  #shape: (C, H, W)
        image = np.transpose(image, (1, 2, 0))  #convert to (H, W, C)

        #load the mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)  #single-channel mask

        #apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return torch.tensor(image).permute(2, 0, 1).float(), torch.tensor(mask).long()

#function to initialize GeoSeg model
def initialize_geoseg_model(num_classes):
    """
    Initialize a GeoSeg model with the required number of output classes.
    """
    model = UNetFormer(backbone_name="resnet50", num_classes=num_classes) 
    return model

def calculate_class_weights(train_loader, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.float32)

    for _, masks in train_loader:
        masks = masks.numpy().flatten()  #flatten to count pixels
        counts = Counter(masks)

        #loop through all counted classes and only update the valid class counts
        for class_idx, count in counts.items():
            if 0 <= class_idx < num_classes:
                class_counts[class_idx] += count
            else:
                print(f"Warning: Found an invalid class index {class_idx} in the mask. Skipping.")

    print("Class Distribution:")
    for class_idx, count in enumerate(class_counts):
        print(f"Class {class_idx}: {count:.0f} pixels")

    #compute inverse class frequencies with small epsilon to avoid division by zero
    epsilon = 1e-6
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts + epsilon)

    return torch.tensor(class_weights, dtype=torch.float32)

if __name__ == "__main__":
    ####################### Dataset and DataLoader #######################
    train_dataset = GeoSegTIFFDataset(
        image_dir = os.path.abspath('Preprocessed Datasets UNetFormer/DOTA/train/images'),
        mask_dir = os.path.abspath('Preprocessed Datasets UNetFormer/DOTA/train/masks')
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = GeoSegTIFFDataset(
        image_dir = os.path.abspath('Preprocessed Datasets UNetFormer/DOTA/val/images'),
        mask_dir = os.path.abspath('Preprocessed Datasets UNetFormer/DOTA/val/masks')
    )
    val_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    ####################### Model and Training Setup #######################
    num_classes = 8  # Set this to the number of segmentation classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #initialize GeoSeg model
    model = initialize_geoseg_model(num_classes=num_classes).to(device)
    class_weights = calculate_class_weights(train_loader, num_classes)
    print(f"Class Weights: {class_weights}")

    #move class weights to the correct device
    class_weights = class_weights.to(device)

    #loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    ####################### Training Loop #######################
    epochs = 20
    learning_rate = 1e-4

    best_loss = float('inf')
    best_val_loss = float('inf')
    os.makedirs("FineTuned_UNetFormer_models_DOTA", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, masks = images.to(device), masks.to(device)

            #forward pass
            outputs = model(images)  #shape: (B, num_classes, H, W)

            if isinstance(outputs, tuple):
                outputs = outputs[0]  #use only the primary output tensor

            #now you can compute the loss using the correct tensor.
            loss = criterion(outputs, masks)

            #backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}")

        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, masks = images.to(device), masks.to(device)

            #forward pass
            outputs = model(images)  #shape: (B, num_classes, H, W)

            if isinstance(outputs, tuple):
                outputs = outputs[0]  #use only the primary output tensor

            #now you can compute the loss using the correct tensor.
            loss = criterion(outputs, masks)

            #backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_val_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.4f}")

        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Avg Val Loss: {avg_val_loss:.4f}")

        #save model if the loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       f"FineTuned_UNetFormer_models_DOTA/best_segmentation_model_{epochs}epochs_{learning_rate}.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")

    torch.save(model.state_dict(),
               f"FineTuned_UNetFormer_models_DOTA/final_segmentation_model_{epochs}epochs_{learning_rate}.pth")
    print("Final model saved.")