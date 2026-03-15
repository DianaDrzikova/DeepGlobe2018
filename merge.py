import torch
import numpy as np


def getIOU_nb(preds, targets, thresh=0.5, eps=1e-6, is_binary=False):
    # If the input is NOT binary
    if not is_binary:
        preds = torch.sigmoid(preds)
        preds = (preds > thresh).float()
    
    if len(preds.shape) == 3:
        preds = preds.unsqueeze(1)
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def merge_inference(output, pred_mask=None, raw_center=None):

    if output is not None:
        pred_mask = torch.sigmoid(output[:, 0, :, :]).cpu().numpy()[0]
        raw_center = torch.sigmoid(output[:, 1, :, :]).cpu().numpy()[0]
   
    # 1. Get prediction
    base_prediction = (pred_mask > 0.5).astype(float)

    # 2. Connect disconnected roads, if the center is strong (> 0.2) we add it back in 
    gaps_filled = ((pred_mask < 0.5) & (raw_center > 0.2)).astype(float)

    # 3. Combine
    final_prediction = base_prediction + gaps_filled
    final_prediction = np.clip(final_prediction, 0, 1)
    
    return pred_mask, raw_center, final_prediction


def updated_merge_inference(output, pred_mask=None, raw_center=None):

    if output is not None:
        pred_mask = torch.sigmoid(output[:, 0, :, :]).cpu().numpy()[0]
        raw_center = torch.sigmoid(output[:, 1, :, :]).cpu().numpy()[0]

    # 1. Normalize the Center Map per image
    center_min = raw_center.min()
    center_max = raw_center.max()
    normalized_center = (raw_center - center_min) / (center_max - center_min + 1e-8)
    
    # 2. Thresholding
    center_structure = (normalized_center > 0.1).astype(float)
    
    # 3. The Merge
    final_prediction = np.maximum((pred_mask > 0.5).astype(float), center_structure)
    
    return pred_mask, normalized_center, final_prediction


def get_batch_merged_prediction(outputs, device):

    batch_size = outputs.shape[0]
    final_preds = []

    # Process each image in the batch individually
    for i in range(batch_size):
        single_output = outputs[i].detach().cpu()
        
        pred_mask = torch.sigmoid(single_output[0]).numpy()
        raw_center = torch.sigmoid(single_output[1]).numpy()

        # 1. Normalize center
        center_min = raw_center.min()
        center_max = raw_center.max()
        normalized_center = (raw_center - center_min) / (center_max - center_min + 1e-8)
        
        # 2. Merge
        center_structure = (normalized_center > 0.2).astype(float)
        final_prediction = np.maximum((pred_mask > 0.5).astype(float), center_structure)
        
        final_preds.append(torch.tensor(final_prediction).unsqueeze(0))

    return torch.stack(final_preds).to(device).float()
