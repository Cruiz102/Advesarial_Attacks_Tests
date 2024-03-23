# Add a function for dividing the Dataset
# into testing


# For Now this code is copied from this file
# https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/utils.py


import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
import torch.nn as nn 
import math
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

class HugginfaceProcessorData(Dataset):
    def __init__(self, processed_dataset):
        self.dataset = processed_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Assuming processed_dataset is indexed like a list or dict
        item = self.dataset[idx]
        # Convert item to the desired format if necessary
        # e.g. convert a PIL image to a tensor

        return {
            "pixel_values": item["pixel_values"],
            "label": item["fine_label"],
            # 'image': item['image']    # Include other relevant fields
        }

def read_yaml(yaml_file: str) -> Optional[Any] :
    if not yaml_file:
        print("No YAML file provided or file not found.")
        return None
    try:
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, PermissionError, yaml.YAMLError) as e:
        print(f"An error occurred: {e}")
        return None


def getYAMLParameter(yaml_config: dict, field: str, key: str = None) -> any:
    """
    Parameters:
    - yaml_config: The YAML configuration represented as a dictionary.
    - field: The top-level field to search within or the exact field if key is None.
    - key: The key whose value is to be retrieved if looking for a value in a nested dictionary.
    """
    if key is None:
        # If no key is provided, return the value of the field directly.
        return yaml_config.get(field)
    else:
        # If a key is provided, first find the field which should be a dict, then return the value of the key within that dict.
        field_content = yaml_config.get(field)
        if isinstance(field_content, dict):
            return field_content.get(key)
    return None


def one_hot_encode(labels, num_classes=100) -> torch.tensor:
    """
    One-hot encode the labels.

    Parameters:
    - labels (torch.Tensor): A tensor of labels, where each label is an integer [0, num_classes-1].
    - num_classes (int): The total number of classes.

    Returns:
    - torch.Tensor: A tensor of shape (N, num_classes) where N is the number of labels, with one-hot encoding.
    """
    # Create a tensor of zeros with shape (len(labels), num_classes)
    labels = torch.tensor(labels)
    one_hot = torch.zeros(len(labels), num_classes)
    
    # Use scatter_ to fill in the appropriate indices with 1
    # labels.unsqueeze(1) creates a column vector for scatter_'s index argument
    # scatter_(dim, index, src) -> dim is the dimension along which to index, index is the tensor of indices, src is the value to fill in
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    return one_hot


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



def l2_distance(predicted_labels, images, adv_images, labels, device="cuda"):
    corrects = (labels.to(device) == predicted_labels)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2

# Code fro the robustbench utils 
# https://github.com/RobustBench/robustbench/blob/master/robustbench/utils.py
def clean_accuracy(
                   predicted_y: torch.Tensor,
                   real_y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def generate_csv_data(output_dir):
    #  A function that calculates the the clean accuracy for each model of a dataset given a model
    clean_accuracy(model, x_test, y_test)
    with open(os.path.join(output_dir, 'clean_accuracy.csv'), 'w') as f:
        f.write('model_name,clean_accuracy\n')
        for model_name, model in models.items():
            clean_acc = clean_accuracy(model, x_test, y_test)
            f.write(f'{model_name},{clean_acc}\n')


def confussion_matrix(model, x, y, device, batch_size=100):
    #  plot a confussion matrix

    pass
def embeddings_interpolation(pixel_value):
    """This function should use the patch interpolation
        for all VIT based models as a extra parameter.
        This is for using images that are bigger than 
        the initial size of the model."""