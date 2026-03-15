import matplotlib.pyplot as plt
import torch


def plot_metrics(train_losses, val_losses, train_ious, val_ious):

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("images/loss.png")  
    

    plt.figure(figsize=(10, 5))
    plt.plot(train_ious, label='Training IOU')
    plt.plot(val_ious, label='Validation IOU')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.savefig("images/metrics.png")


def display_prediction(display_dict, i):

    image = display_dict["image"].permute(1, 2, 0).cpu().numpy()
    output =  display_dict["output"].squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 4))

    if "label" in display_dict:
        label = display_dict["label"].squeeze(0).cpu().numpy()

        if "prob_map" in display_dict:
 
            prob_map = display_dict["prob_map"].squeeze(0).cpu().numpy()

            plt.subplot(1, 4, 1)
            plt.imshow(image)  
            plt.title("Input Image")
            plt.axis("off")
        
            plt.subplot(1, 4, 2)
            plt.imshow(output, cmap='gray')
            plt.title("Model Prediction")
            plt.axis("off")
            
            plt.subplot(1, 4, 3)
            plt.imshow(prob_map, cmap='magma')
            plt.title("Skeletal Prediction")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(label, cmap='gray')
            plt.title("Actual Mask")
            plt.axis("off")
        else:
            plt.subplot(1, 3, 1)
            plt.imshow(image)  
            plt.title("Input Image")
            plt.axis("off")
        
            plt.subplot(1, 3, 2)
            plt.imshow(output, cmap='gray')
            plt.title("Model Prediction")
            plt.axis("off")
        
            plt.subplot(1, 3, 3)
            plt.imshow(label, cmap='gray')
            plt.title("Actual Mask")
            plt.axis("off")
            
    else:
        prob_map = display_dict["prob_map"].squeeze(0).cpu().numpy()
        plt.subplot(1, 3, 1)
        plt.imshow(image)  
        plt.title("Input Image")
        plt.axis("off")
    
        plt.subplot(1, 3, 2)
        plt.imshow(output, cmap='gray')
        plt.title("Model Prediction")
        plt.axis("off")
    
        plt.subplot(1, 3, 3)
        plt.imshow(prob_map, cmap='magma')
        plt.title("Skeletal Prediction")
        plt.axis("off")
    
    plt.savefig(f"images/prediction{int(i)}.png")

def viz_maps(image, original_mask, p_mask, p_center, p_final, i):

    original_mask = original_mask.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 5, 1); plt.title("Input"); plt.imshow(image.permute(1,2,0).cpu().numpy())
    plt.subplot(1, 5, 2); plt.title("Standard Mask"); plt.imshow(p_mask > 0.5, cmap='gray')
    plt.subplot(1, 5, 3); plt.title("Center Head"); plt.imshow(p_center, cmap='magma')
    plt.subplot(1, 5, 4); plt.title("Merge"); plt.imshow(p_final, cmap='gray')
    plt.subplot(1, 5, 5); plt.title("Original"); plt.imshow(original_mask, cmap='gray')
    plt.savefig(f"images/merge{i}.png")


def getIOU(preds, targets, thresh=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > thresh).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
