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

class TIFFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.transform = transform
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()  # Shape: (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)  # Single-channel mask

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return torch.tensor(image).permute(2, 0, 1).float(), torch.tensor(mask).long()

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

if __name__ == '__main__':

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
    num_classes = 8
    segmentation_head = create_segmentation_head(embed_dim=model_args["embed_dim"], num_classes=num_classes)


    ################ Create full model: Combine encoder and segmentation head #################
    class FullModel(nn.Module):
        def __init__(self, encoder, segmentation_head):
            super(FullModel, self).__init__()
            self.encoder = encoder
            self.segmentation_head = segmentation_head

        def forward(self, x):
            features, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)
            features = features[:, 1:, :]  # Drop CLS token
            side_length = int(features.shape[1] ** 0.5)
            reshaped_features = features.view(-1, side_length, side_length, model_args["embed_dim"]).permute(0, 3, 1, 2)
            return self.segmentation_head(reshaped_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = FullModel(encoder, segmentation_head).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    ################ Training ############################
    train_dataset = TIFFDataset(image_dir="Preprocessed Datasets/DOTA/train/images", mask_dir="Preprocessed Datasets/DOTA/train/masks")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    epochs = 5
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.unsqueeze(2)
            images = images.repeat(1, 2, 1, 1, 1)
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)  # Shape: (B, num_classes, H, W)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        # Save model if the loss improves
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "FineTuned_models_DOTA/best_segmentation_model.pth")
            print(f"Model saved at epoch {epoch + 1} with loss {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(), "FineTuned_models_DOTA/final_segmentation_model.pth")
    print("Final model saved.")



