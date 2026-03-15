import torch
from torch.utils.data import DataLoader
import os 
from dataset import RoadSegmentationDataset
from tqdm import tqdm
import random

torch.cuda.empty_cache()

from config import get_config
from model import get_model
from utils import display_prediction, getIOU, viz_maps
from merge import merge_inference, get_batch_merged_prediction, getIOU_nb


def predict_with_tta(model, images, device):
    model.eval()
    images = images.to(device)

    B, C, H, W = images.shape
    
    total_pred = torch.zeros((B, 1, H, W), device=device)
    
    with torch.no_grad():
        # 1. Prediction ---
        out = model(images)
        total_pred += torch.sigmoid(out[:, 0:1, :, :])
        
        # 2. Horizontal Flip 
        img_hflip = torch.flip(images, dims=[3])
        out_hflip = model(img_hflip)
        pred_hflip = torch.sigmoid(out_hflip[:, 0:1, :, :])
        total_pred += torch.flip(pred_hflip, dims=[3]) # Flip back
        
        # 3. Vertical Flip 
        img_vflip = torch.flip(images, dims=[2])
        out_vflip = model(img_vflip)
        pred_vflip = torch.sigmoid(out_vflip[:, 0:1, :, :])
        total_pred += torch.flip(pred_vflip, dims=[2]) # Flip back
        
        # 4. Transpose 
        img_trans = torch.transpose(images, 2, 3)
        out_trans = model(img_trans)
        pred_trans = torch.sigmoid(out_trans[:, 0:1, :, :])
        total_pred += torch.transpose(pred_trans, 2, 3) # Transpose back

    # Average the 4 predictions
    return total_pred / 4.0


def evaluate_with_labels(config, test_loader, model_path, device):
    model = get_model(config)
    model.to(device)

    # Load the best model weights
    path = os.path.join(f"{model_path}/best_model_{config.model.name}.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    total_iou = 0.0
    aggresive_iou = 0.0 
    transform_iou = 0.0
    i = 0
    with torch.no_grad():
        for images, labels, dist_maps in tqdm(test_loader, desc="Evaluating"):

            images, labels, dist_maps = images.to(device), labels.to(device), dist_maps.to(device)
            outputs = model(images)
            out_bin = (torch.sigmoid(outputs) > 0.5).float()
            outputs_new = predict_with_tta(model, images, device)
            out_bin_new = (outputs_new > 0.5).float()

            if i == 0: # random.randint(0, len(test_loader)) % 3 == 0:
                if len(out_bin[0]) == 2:
                    display_dict = {"image":images[0], "label":labels[0], "output":out_bin[0][0], "prob_map":out_bin[0][1]}
                    display_prediction(display_dict, i)
                    p_mask, p_center, p_final = merge_inference(outputs)
                    viz_maps(images[0], labels[0], p_mask, p_center, p_final, i)
                    i += 1
                else:
                    display_dict = {"image":images[0], "label":labels[0], "output":out_bin[0]}
                    display_prediction(display_dict, i)
                    i += 1

            if len(out_bin[0]) == 2:
                p_final_batch = get_batch_merged_prediction(outputs, device)
                aggresive_iou += getIOU_nb(p_final_batch, labels, is_binary=True)
                transform_iou += getIOU_nb(out_bin_new, labels)
            total_iou += getIOU(out_bin, labels)

    avg_iou = total_iou / len(test_loader)
    avg_agg_iou = aggresive_iou / len(test_loader)
    avg_tra_iou = transform_iou / len(test_loader)
    print(f"Average IoU: Original {avg_iou:.4f}, Merge: {avg_agg_iou:.4f}, Transformed: {avg_tra_iou:.4f}")


def evaluate(config, test_loader, model_path, device):
    model = get_model(config)
    model.to(device)

    # Load the best model weights
    path = os.path.join(f"{model_path}/best_model_{config.model.name}.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    i = 0.0
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Evaluating"):

            images = images.to(device)
            outputs = model(images)
            out_bin = (torch.sigmoid(outputs) > 0.5).float()

            if random.randint(0, len(test_loader)) % 7 == 0:
                display_dict = {"image":images[0], "output":out_bin[0][0], "prob_map":out_bin[0][1]}
                display_prediction(display_dict, i)
                i += 1

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loaded config file.")

    # Load dataset parameters
    batch_size = config.training.batch_size
    dataset_path = config.dataset.path
    model_path = config.eval.model_path

    print("Model:",model_path)

    csv_path = os.path.join(dataset_path, "metadata.csv")

    # for models that have been trained only on 1000 training images
    if config.eval.labels: 
        test_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="train", clip=100, offset=6126) 
        #clip=50, offset=3101)
        print(f"Datasets loaded: Test images: {len(test_dataset)}.")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        evaluate_with_labels(config, test_loader, model_path, device)
    else: # evaluating final model on test dataset
        test_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="test", clip=0, offset=0)
        print(f"Datasets loaded: Test images: {len(test_dataset)}.")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        evaluate(config, test_loader, model_path, device)

    print("Evaluation is done.")


if __name__ == "__main__":
    main()