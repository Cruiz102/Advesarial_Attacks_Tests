# Add a function for dividing the Dataset
# into testing


# For Now this code is copied from this file
# https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/utils.py


import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from typing import Optional, Union, List, Dict, Any

from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageProcessingDataset(Dataset):

#  A dataset for processing each the images before using them in the model
    def __init__(self, image_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = image_processor

    def __call__(self, features):
        # Convert file paths or PIL images in 'features' to actual images
        
        # Use the feature_extractor to preprocess the images
        batch = self.feature_extractor(images=features["images"], return_tensors="pt")
        
        return batch



def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)



def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()


def compare_images(attack_name: str , model_name:str, og_img: Optional[torch.Tensor] , atk_image:Optional[torch.Tensor]):
    og_img = og_img.squeeze(0).permute(1, 2, 0)
    atk_image = atk_image.squeeze(0).permute(1, 2, 0)
    diff = (og_img - atk_image).abs()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(og_img)
    ax1.set_title("Original Image")
    ax2.imshow(atk_image)
    ax2.set_title(attack_name)
    ax3.imshow(diff)
    ax3.set_title("Difference")
    plt.show()
    pass



def plot_grid(images : list, rows, cols, figsize=(10, 10)):

    fig = plt.figure(figsize=figsize)
    for i in range(1, rows * cols + 1):
        img = images[i - 1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def save_grid_images(images, rows, cols, experiment_name, output_dir='outputs'):
    experiment_dir = os.path.join(output_dir, experiment_name)
    if os.path.exists(experiment_dir):
        for i in range(2, 100):
            tmp_dir = f"{experiment_dir}_{i}"
            if not os.path.exists(tmp_dir):
                experiment_dir = tmp_dir
                break
        
    os.makedirs(experiment_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(10,10))
    for i in range(1, rows*cols + 1):
        img = images[i-1] 
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')
        
    grid_path = os.path.join(experiment_dir, 'grid.png') 
    plt.savefig(grid_path)
    print(f"Saved grid images to {grid_path}")



def embeddings_interpolation(pixel_value):
    """This function should use the patch interpolation
        for all VIT based models as a extra parameter.
        This is for using images that are bigger than 
        the initial size of the model."""