import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import rasterio
import numpy as np
import os
from Prithvi import MaskedAutoencoderViT
import torch
import yaml
from tqdm import tqdm
from collections import Counter
from finetuning_Prithvi_helper import TIFFDataset, create_full_model, create_segmentation_head, calculate_class_weights

if __name__ == '__main__':
    ############### Parameters Setting ####################
    TAG = "urban"
    TRAIN_IMAGES_PATH = f"Preprocessed Datasets Prithvi/LoveDa/{TAG}/train/images"
    TRAIN_MASKS_PATH = f"Preprocessed Datasets Prithvi/LoveDa/{TAG}/train/masks"
    VAL_IMAGES_PATH = f"Preprocessed Datasets Prithvi/LoveDa/{TAG}/val/images"
    VAL_MASKS_PATH = f"Preprocessed Datasets Prithvi/LoveDa/{TAG}/val/masks"
    CLASS_DISTRIB_FIX = False
    EPOCHS = 20
    NUM_CLASSES = 7
    LEARNING_RATE = 1e-4

    ################ Import Model #########################
    weights_path = "Prithvi_100M.pt"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    model_cfg_path = "Prithvi_100M_config.yaml"
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]
    model_args["num_frames"] = 1  # Use a single frame
    encoder = MaskedAutoencoderViT(**model_args)
    encoder.eval()

    del checkpoint['pos_embed']
    del checkpoint['decoder_pos_embed']
    _ = encoder.load_state_dict(checkpoint, strict=False)

    ################ Create Segmentation Head #################
    segmentation_head = create_segmentation_head(embed_dim=model_args["embed_dim"], num_classes=NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = create_full_model(encoder, segmentation_head, model_args["embed_dim"]).to(device)

    ################ Training ############################

    train_dataset = TIFFDataset(image_dir=TRAIN_IMAGES_PATH, mask_dir=TRAIN_MASKS_PATH)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = TIFFDataset(image_dir=VAL_IMAGES_PATH, mask_dir=VAL_MASKS_PATH)
    val_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    if CLASS_DISTRIB_FIX:
        class_weights, class_counts = calculate_class_weights(train_loader, NUM_CLASSES)
        print(f"Class Weights: {class_weights}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            images = images.unsqueeze(2)  # Add temporal dimension (T=1)
            images = images.repeat(1, 2, 1, 1, 1)
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)  # Shape: (B, num_classes, H, W)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.unsqueeze(2)
                val_images = val_images.repeat(1, 2, 1, 1, 1)
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                # Forward pass
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)

                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Avg Val Loss: {avg_val_loss:.4f}")

        # Save model if the loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       f"FineTuned_models_LoveDa/{TAG}/best_segmentation_model_{EPOCHS}epochs_{LEARNING_RATE}.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")

    torch.save(model.state_dict(), f"FineTuned_models_LoveDa/{TAG}/final_segmentation_model_{EPOCHS}epochs_{LEARNING_RATE}.pth")
    print("Final model saved.")

