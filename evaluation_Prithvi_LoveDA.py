import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from Prithvi import MaskedAutoencoderViT
from finetuning_Prithvi_helper import create_full_model, create_segmentation_head, TIFFDataset
from evaluation_helper import compute_miou, compute_acc

if __name__ == '__main__':

    ############### Parameters Setting ####################
    TAG = "urban"
    TEST_IMAGES_PATH = f"Preprocessed Datasets Prithvi/LoveDa/{TAG}/test/images"
    TEST_MASKS_PATH = f"Preprocessed Datasets Prithvi/LoveDa/{TAG}/test/masks"
    NUM_CLASSES = 7
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    SAVED_MODEL_PATH = f"FineTuned_Prithvi_models_LoveDa/{TAG}/final_segmentation_model_{EPOCHS}epochs_{LEARNING_RATE}.pth"

    ################ Load Weights and Saved Model #########################
    weights_path = "Prithvi_100M.pt"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    model_cfg_path = "Prithvi_100M_config.yaml"
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]
    model_args["num_frames"] = 1
    encoder = MaskedAutoencoderViT(**model_args)

    segmentation_head = create_segmentation_head(embed_dim=model_args["embed_dim"], num_classes=NUM_CLASSES)
    model = create_full_model(encoder, segmentation_head, model_args["embed_dim"])

    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location="cpu"))
    model.eval()
    model.to("cpu")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = TIFFDataset(
        image_dir=TEST_IMAGES_PATH,
        mask_dir=TEST_MASKS_PATH,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_mean_ious = []
    all_class_ious = []
    all_acc = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.unsqueeze(2)
            images = images.repeat(1, 2, 1, 1, 1)
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            class_ious, mean_iou = compute_miou(preds.squeeze(0), masks.squeeze(0), NUM_CLASSES)
            all_class_ious.append(class_ious)
            all_mean_ious.append(mean_iou)

            acc = compute_acc(preds.squeeze(0), masks.squeeze(0))
            all_acc.append(acc)

    class_ious_avg = np.nanmean(np.array(all_class_ious), axis=0)

    for c in range(NUM_CLASSES):
        print(f"Class {c} IoU: {class_ious_avg[c]:.4f}")

    overall_miou = np.nanmean(all_mean_ious)
    print(f"Mean IoU (mIoU) for all classes: {overall_miou:.4f}")

    overall_accuracy = np.mean(all_acc)
    print(f"Pixel Accuracy: {overall_accuracy:.4f}")