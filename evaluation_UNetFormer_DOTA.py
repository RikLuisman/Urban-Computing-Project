import numpy as np
import torch
from torch.utils.data import DataLoader
from finetuning_UNetFormer_for_DOTA import GeoSegTIFFDataset
from UNetFormer import UNetFormer
import torch.nn as nn
import os
from evaluation_helper import compute_miou, compute_acc

if __name__ == '__main__':

    #load UNetFormer model
    num_classes = 8
    model = UNetFormer(backbone_name="resnet50", num_classes=num_classes)  # Replace 'resnet50' if needed
    model.load_state_dict(torch.load("FineTuned_UNetFormer_models_DOTA/final_segmentation_model_5epochs_1e-05.pth", map_location="cpu", weights_only=True))
    model.eval()
    model.to("cpu")

    all_mean_ious = []
    all_class_ious = []
    all_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = GeoSegTIFFDataset(
        image_dir = os.path.abspath('Preprocessed Datasets UNetFormer/DOTA/test/images'),
        mask_dir = os.path.abspath('Preprocessed Datasets UNetFormer/DOTA/test/masks')
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            #get model outputs
            outputs = model(images)
            if isinstance(outputs, tuple):  #handle tuple outputs
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)

            class_ious, mean_iou = compute_miou(preds.squeeze(0), masks.squeeze(0), num_classes)
            all_class_ious.append(class_ious)
            all_mean_ious.append(mean_iou)

            acc = compute_acc(preds.squeeze(0), masks.squeeze(0))
            all_acc.append(acc)

    #average IoU across batches for each class
    class_ious_avg = np.nanmean(np.array(all_class_ious), axis=0)

    for c in range(num_classes):
        print(f"Class {c} IoU: {class_ious_avg[c]:.4f}")

    overall_miou = np.nanmean(all_mean_ious)
    print(f"Mean IoU (mIoU) for all classes: {overall_miou:.4f}")

    overall_accuracy = np.mean(all_acc)
    print(f"Pixel Accuracy: {overall_accuracy:.4f}")
