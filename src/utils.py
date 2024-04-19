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
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns


class ClipDataset(Dataset):
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
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            # 'image': item['image']    # Include other relevant fields
        }


class HugginfaceProcessorData(Dataset):
    def __init__(self, processed_dataset, label_title ,image_feature_title = "pixel_values"):
        self.dataset = processed_dataset
        self.image_feature_title = image_feature_title
        self.label_title = label_title

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Assuming processed_dataset is indexed like a list or dict
        item = self.dataset[idx]
        # Convert item to the desired format if necessary
        # e.g. convert a PIL image to a tensor

        return {
            self.image_feature_title: item[self.image_feature_title],
            self.label_title: item[self.label_title],
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



def l2_distance(predicted_labels, images, adv_images, labels, device="cpu"):
    predicted_labels = predicted_labels.to(device)
    corrects = (labels.to(device) == predicted_labels)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    print(f"L2 distance: {l2}")
    return l2

# Code fro the robustbench utils 

import matplotlib.pyplot as plt
import numpy as np


def plot_image_comparison(original_images, attacked_images, original_preds, attacked_preds, true_labels, highlight_misclass=False, pairs_per_row=2):
    assert len(original_images) == len(attacked_images) == len(original_preds) == len(attacked_preds) == len(true_labels), "All lists must have the same length."
    original_images = original_images.permute(0, 2, 3, 1).cpu()  # Ensure the tensor is on CPU and permute
    attacked_images = attacked_images.permute(0, 2, 3, 1).cpu()  # Ensure the tensor is on CPU and permute
    
    num_images = len(original_images)
    num_rows = (num_images + pairs_per_row - 1) // pairs_per_row  # Calculate number of rows needed
    fig, axes = plt.subplots(nrows=num_rows, ncols=2 * pairs_per_row, figsize=(4 * pairs_per_row, 2 * num_rows))  # Adjust figure size accordingly
    
    if num_images == 1 or num_rows == 1:
        axes = np.expand_dims(axes, 0)  # Handle the case of 1 image or 1 row
    
    for i in range(num_images):
        row = i // pairs_per_row
        col = (i % pairs_per_row) * 2  # Each pair takes 2 columns in the grid
        
        # Plot original image
        ax = axes[row, col] if num_rows > 1 else axes[col]
        orig_img = np.array(original_images[i])  # Already on CPU
        if highlight_misclass and original_preds[i] != true_labels[i]:
            orig_img = np.clip(orig_img * 0.7 + np.array([255, 0, 0])[None, None, :] * 0.3, 0, 255).astype(np.uint8)
            title_color = 'red'
        else:
            title_color = 'black'
        ax.imshow(orig_img, interpolation='nearest')
        ax.set_title(f'Orig\nPred: {original_preds[i]}, True: {true_labels[i]}', color=title_color)
        ax.axis('off')
        
        # Plot attacked image
        ax = axes[row, col + 1] if num_rows > 1 else axes[col + 1]
        att_img = np.array(attacked_images[i])  # Already on CPU
        if highlight_misclass and attacked_preds[i] != true_labels[i]:
            att_img = np.clip(att_img * 0.7 + np.array([255, 0, 0])[None, None, :] * 0.3, 0, 255).astype(np.uint8)
            title_color = 'red'
        else:
            title_color = 'black'
        ax.imshow(att_img, interpolation='nearest')
        ax.set_title(f'Att\nPred: {attacked_preds[i]}, True: {true_labels[i]}', color=title_color)
        ax.axis('off')
    
    # Hide unused axes if the grid does not fill up completely
    for j in range(i + 1, num_rows * pairs_per_row):
        row = j // pairs_per_row
        col = (j % pairs_per_row) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
    
    plt.tight_layout()

    # Convert figure to a PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory

    return img  # Return the PIL image


def conf_matrix_img(pred_labels, true_labels):
    conf_matrix = confusion_matrix(pred_labels, true_labels)
    # Displaying the confusion matrix using seaborn for better visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer to the beginning so it can be read
    img = Image.open(buf)  # Create a PIL image from the buffer
    plt.close()  # Close the plt figure to free memory
    # Optionally, return the PIL image if you need to use it elsewhere
    return img


def embeddings_interpolation(pixel_value):
    """This function should use the patch interpolation
        for all VIT based models as a extra parameter.
        This is for using images that are bigger than 
        the initial size of the model."""