import numpy as np
import torch
from torch.utils.data import DataLoader
from finetuning_Prithvi import TIFFDataset, create_segmentation_head
from Prithvi import MaskedAutoencoderViT
import yaml
import torch.nn as nn

def compute_miou(preds, targets, num_classes):
    """
    This function calculates the IoU for each class and computes the mean (mIoU).
    Classes with no instances in the batch are ignored using NaN.
    """

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    ious = []
    for c in range(num_classes):
        intersection = np.logical_and(preds == c, targets == c).sum()
        union = np.logical_or(preds == c, targets == c).sum()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append(intersection / union)

    # Compute mean IoU, ignoring NaN values (for classes not present in the batch)
    mean_iou = np.nanmean(ious)
    return ious, mean_iou

def compute_acc(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    correct_pixels = np.sum(preds == targets)
    total_pixels = np.prod(targets.shape)
    accuracy = correct_pixels / total_pixels

    return accuracy

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


if __name__ == '__main__':

    weights_path = "Prithvi_100M.pt"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    model_cfg_path = "Prithvi_100M_config.yaml"
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]
    model_args["num_frames"] = 1
    encoder = MaskedAutoencoderViT(**model_args)

    num_classes = 8
    segmentation_head = create_segmentation_head(embed_dim=model_args["embed_dim"], num_classes=num_classes)

    model = FullModel(encoder, segmentation_head)
    model.load_state_dict(torch.load("FineTuned_models_DOTA/final_segmentation_model.pth", map_location="cpu"))
    model.eval()
    model.to("cpu")

    all_mean_ious = []
    all_class_ious = []
    all_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = TIFFDataset(
        image_dir="Preprocessed Datasets/DOTA/test/images",
        mask_dir="Preprocessed Datasets/DOTA/test/masks"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.unsqueeze(2)
            images = images.repeat(1, 2, 1, 1, 1)
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            class_ious, mean_iou = compute_miou(preds.squeeze(0), masks.squeeze(0), num_classes)
            all_class_ious.append(class_ious)
            all_mean_ious.append(mean_iou)

            acc = compute_acc(preds.squeeze(0), masks.squeeze(0))
            all_acc.append(acc)

    # Average IoU across batches for each class
    class_ious_avg = np.nanmean(np.array(all_class_ious), axis=0)

    for c in range(num_classes):
        print(f"Class {c} IoU: {class_ious_avg[c]:.4f}")

    overall_miou = np.nanmean(all_mean_ious)
    print(f"Mean IoU (mIoU) for all classes: {overall_miou:.4f}")

    overall_accuracy = np.mean(all_acc)
    print(f"Pixel Accuracy: {overall_accuracy:.4f}")