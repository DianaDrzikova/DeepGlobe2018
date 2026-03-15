import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import random

from loss import MyDiceLoss, BCEDiceLoss, CenterLoss
from dataset import RoadSegmentationDataset, Augmentation    
from config import get_config, Config
from utils import plot_metrics, getIOU
from model import get_model
    

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Randomness locked with seed: {seed}")

seed_everything(42)


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)


def validate(model, val_loader, loss_fn, config, device):
    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for images, labels, dist_maps in val_loader:
            images, labels, dist_maps = images.to(device), labels.to(device), dist_maps.to(device)
            outputs = model(images)
            # Compute loss
            if config.training.loss_function == "MyDiceLoss" or config.training.loss_function == "BCEDiceLoss":
                loss = loss_fn(outputs, labels)
            elif config.training.loss_function == "CenterLoss":
                loss = loss_fn(outputs, labels, dist_maps)
            val_loss += loss.item()
            out_bin = (torch.sigmoid(outputs) > 0.5).float()
            val_iou += getIOU(out_bin, labels)

    val_iou /= len(val_loader)
    val_loss /= len(val_loader)

    return val_iou, val_loss
    
    
def train(config, train_loader, val_loader, device):
    # Initialize model
    model = get_model(config)

    continue_training = config.training.continue_training

    if model is None:
        raise ValueError(f"Model {config.model.name} could not be initialized.")

    if continue_training:
        model_path = config.eval.model_path
        path = os.path.join(f"{model_path}/best_model_{config.model.name}.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device) 
        print("Continue training")
    else:
        # Move model to device
        model = model.to(device) 

    # Training parameters
    lr = config.training.learning_rate
    num_epochs = config.training.num_epochs

    # Define loss and optimizer
    if config.training.loss_function == "MyDiceLoss":
        loss_fn = MyDiceLoss()
    elif config.training.loss_function == "BCEDiceLoss":
        loss_fn = BCEDiceLoss()
    elif config.training.loss_function == "CenterLoss":
        loss_fn = CenterLoss()
    else:
        raise KeyError(f"Loss function {config.training.loss_function} not recognized.")
    
    if config.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise KeyError(f"Optimizer {config.training.optimizer} not recognized.")

    train_losses, val_losses, train_ious, val_ious = [], [], [], []
    best_iou_val = 0.0

    for epoch in range(1, num_epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        for images, labels, dist_maps in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, labels, dist_maps = images.to(device), labels.to(device), dist_maps.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            if config.training.loss_function == "MyDiceLoss" or config.training.loss_function == "BCEDiceLoss":
                loss = loss_fn(outputs, labels)
            elif config.training.loss_function == "CenterLoss":
                loss = loss_fn(outputs, labels, dist_maps)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Thresholding the output if x > 0.5: road else: not road
            out_bin = (torch.sigmoid(outputs) > 0.5).float()
            train_iou += getIOU(out_bin, labels)

        train_iou /= len(train_loader)
        train_loss /= len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training IOU: {train_iou:.4f}, Training Loss: {train_loss:.4f}")
        
        train_ious.append(train_iou)
        train_losses.append(train_loss)

        # Validate model
        val_iou, val_loss = validate(model, val_loader, loss_fn, config, device)

        # Store best model based on highest val IoU
        if val_iou > best_iou_val:
            best_iou_val = val_iou
            print(f"Saving best model, epoch {epoch + 1} with validation IOU: {val_iou:.4f}...")
            torch.save(model.state_dict(), f"checkpoints/best_model_{config.model.name}.pth")
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation IOU: {val_iou:.4f}, Validation Loss: {val_loss:.4f}")

        # Store val IoU and Loss values
        val_ious.append(val_iou)
        val_losses.append(val_loss)

    # Plot loss and IoU graphs
    plot_metrics(train_losses, val_losses, train_ious, val_ious)

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loaded config file.")

    # Load dataset parameters
    batch_size = config.training.batch_size
    dataset_path = config.dataset.path
    image_size = config.dataset.image_size
    train_split_size = config.dataset.train_split_size
    train_size = config.dataset.train_size

    csv_path = os.path.join(dataset_path, "metadata.csv")
    df = pd.read_csv(csv_path)

    # Parial vs Full training mode
    if train_size == "partial":
        dataset_length_raw = 1000
    if train_size == "full":
        dataset_length_raw = len(df[df['split'] == "train"]) - 100

    # Train/Val split
    train_size = int(train_split_size * dataset_length_raw)
    val_size = dataset_length_raw - train_size

    # Load datasets
    train_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="train",
                                transform=Augmentation(img_size = image_size),
                                clip=train_size)

    val_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="train",
                                clip=val_size, offset=train_size)

    test_dataset = RoadSegmentationDataset(dataset_path, csv_path, split="test")

    print(f"Datasets loaded: Train images: {len(train_dataset)}, Test images: {len(test_dataset)}, Val images: {len(val_dataset)}.")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    # Train
    train(config, train_loader, val_loader, device)

    # Training report config file
    with open(f"checkpoints/report_{config.model.name}.txt", "w") as f:
        for i in config.__dict__:   
            if isinstance(config.__dict__[i], Config):
                for j in config.__dict__[i].__dict__:
                    f.write(f"{i}.{j}: {config.__dict__[i].__dict__[j]}\n")
            else:
                f.write(f"{i}: {config.__dict__[i]}\n")


if __name__ == "__main__":
    main()