import torch
import os 
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

from dataset import RoadSegmentationDataset
from config import get_config
from model import get_model
from utils import display_prediction, getIOU

def evaluate_with_labels(config, test_loader, device):
    model = get_model(config)
    model.to(device)

    total_iou = 0.0
    i = 0
    with torch.no_grad():
        for images, labels, dist_maps in tqdm(test_loader, desc="Evaluating"):
            images, labels, dist_maps = images.to(device), labels.to(device), dist_maps.to(device)
            outputs = model(images)
            # Thresholding the output if x > 0.5: road else: not road
            out_bin = (torch.sigmoid(outputs) > 0.5).float()
            # Select random images to visually evaluate
            if random.randint(0, len(test_loader)) % 3 == 0:
                    display_dict = {"image":images[0], "label":labels[0], "output":out_bin[0][0]}
                    display_prediction(display_dict, i)
                    i += 1
            total_iou += getIOU(out_bin, labels)

    avg_iou = total_iou / len(test_loader)
    print(f"Average IoU: Original {avg_iou:.4f}")


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
            # Thresholding the output if x > 0.5: road else: not road
            out_bin = (torch.sigmoid(outputs) > 0.5).float()
            # Select random images to visually evaluate
            if random.randint(0, len(test_loader)) % 7 == 0:
                display_dict = {"image":images[0], "output":out_bin[0][0]}
                display_prediction(display_dict, i)
                i += 1

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loaded config file.")

    # Load parameters
    batch_size = config.training.batch_size
    dataset_path = config.dataset.path

    print("Model:",config.model.name, config.model.encoder_name, config.training.loss_function)

    csv_path = os.path.join(dataset_path, "metadata.csv")

    # for models that have been trained only on X training images
    if config.eval.labels: 
        test_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="train", clip=100, clip_offset=6126) 
        #clip=50, clip_offset=3101)
        print(f"Datasets loaded: Test images: {len(test_dataset)}.")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        evaluate_with_labels(config, test_loader, device)
    else: # evaluating final model on test dataset
        test_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="test", clip=0, clip_offset=0)
        print(f"Datasets loaded: Test images: {len(test_dataset)}.")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        evaluate(config, test_loader, device)

    print("Evaluation is done.")


if __name__ == "__main__":
    main()