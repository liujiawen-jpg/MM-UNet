import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        mask_path = self.masks_path[index]

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image at path: {image_path}")
            image = image / 255.0  #0-1
            image = np.transpose(image, (2, 0, 1))  #(3, 512, 512)
            image = torch.tensor(image, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise e

        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask at path: {mask_path}")
            mask = mask / 255.0  #0-1
            mask = np.expand_dims(mask, axis=0)  #(1, 512, 512)
            mask = torch.tensor(mask, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading mask: {str(e)}")
            raise e

        return image, mask


    def __len__(self):
        return self.n_samples