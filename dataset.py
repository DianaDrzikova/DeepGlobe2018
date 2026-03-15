import torch
import albumentations
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


class RoadSegmentationDataset(Dataset):
    def __init__(self, dataset_dir, csv_path, split, transform=None, clip = 0, offset = 0, img_size = 512):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.offset = offset
        self.csv_path = csv_path
        self.newsize = (img_size, img_size)

        self.image_data = []
        
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='sat_image_path') 
        
        split_df = df[df['split'] == split]

        images = {}
        labels = {}
        image_paths = []

        for i in range(len(split_df)): # Iterate over images in the csv
            set_image_path = split_df.iloc[i]['sat_image_path']
            filename = os.path.basename(set_image_path)
            images.update({filename.split('_')[0]: os.path.join(dataset_dir, set_image_path)})

            # If the split is train, load labels as well
            if split == "train":
                set_label_path = split_df.iloc[i]['mask_path']
                labels.update({filename.split('_')[0] + '_mask': os.path.join(dataset_dir, set_label_path)})
                
        sorted_keys = sorted(list(images.keys()))

        if split == "train" and len(labels) != len(images):
           raise ValueError("Mismatch between number of images and labels in training set")
        elif split == "train":
            image_paths = [(images[key], labels[key + '_mask']) for key in sorted_keys]
        else:
            image_paths = [(images[key], None) for key in images]
        
        # Clipping the dataset
        if(clip > 0): image_paths = image_paths[offset:offset + clip]
        
        self.image_paths = image_paths 

    def __mask_map(self, mask):
        # 1. Compute Euclidean Distance from background
        dist = distance_transform_edt(mask)
        
        # 2. Normalize so the widest part of the road is 1.0
        # Add epsilon to avoid division by zero
        if dist.max() > 0:
            dist = dist / dist.max()
        
        return dist.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image, label = self.image_paths[idx]

        image = Image.open(image).convert('RGB')
        image = np.array(image.resize(self.newsize))

        if label is not None:
            label = Image.open(label).convert('L')
            label = np.array(label.resize(self.newsize))
        if self.transform:
            image, label = self.transform(image, label)

        # Normalization
        image = image / 255.0
        if label is not None:
            label = label / 255.0
            dist_map = self.__mask_map(label)
            return (torch.tensor(image, dtype=torch.float32).permute(2, 0, 1),
                torch.tensor(label, dtype=torch.float32).unsqueeze(0), 
                torch.tensor(dist_map, dtype=torch.float32).unsqueeze(0))
        else:
            return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)


class Augmentation():
  def __init__(self,
               img_size = 256):
    self.img_size = img_size
    self.list = [
        albumentations.RandomCrop(width=self.img_size, height=self.img_size),
        albumentations.Rotate(limit=[-180,180]),
        albumentations.Blur(blur_limit=(5,5)),
    ]
    self.transform = albumentations.Compose(self.list)

  def __call__(self, image, mask):
    result = self.transform(image=image, mask=mask)
    return (result["image"], result["mask"])
