import torch
import torch.nn as nn   
from segmentation_models_pytorch.losses import DiceLoss

class MyDiceLoss(nn.Module):
    def __init__(self):
        super(MyDiceLoss, self).__init__()
        self.dice = DiceLoss(mode="binary", from_logits=True, smooth=1.0)

    def forward(self, inputs, target):
        dice_loss = self.dice(inputs, target.float())
        return dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        targets_flat = targets_flat.float()

        # 1. BCE Loss 
        bce_loss = self.bce(inputs_flat, targets_flat)
        
        # 2. Dice Loss
        inputs_prob = torch.sigmoid(inputs_flat)       
        
        smooth = 1.0
        intersection = (inputs_prob * targets_flat).sum()                            
        dice_loss = 1 - ((2. * intersection + smooth) / 
                        (inputs_prob.sum() + targets_flat.sum() + smooth))
        
        return 0.5 * bce_loss + 0.5 * dice_loss


class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode='binary', from_logits=True)
        self.mse = nn.MSELoss()

    def forward(self, prediction, target_mask, target_dist):
 
        pred_mask = prediction[:, 0:1, :, :] # Slice channel 0
        pred_dist = prediction[:, 1:2, :, :] # Slice channel 1
        
        # 1. Dice Loss
        loss_mask = self.dice(pred_mask, target_mask)
        
        # 2. Regression Loss 
        loss_dist = self.mse(torch.sigmoid(pred_dist), target_dist)
        
        # Combine them (You can tweak the weight 0.5)
        return loss_mask + (10.0 * loss_dist)