# A File for creating experiments graphs and results
# inside a folder called "results".

# We will give datasets that are going to be used.
# We will have to defined a way for reading yaml config
# files 

import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoConfig, TrainingArguments, Trainer, AutoModel
import torchattacks as ta
from utils import read_yaml
import argparse 
from datasets import load_dataset
import torch.nn as nn
import torch
# Read the yaml config file of the configuration of the attacks
import tqdm
import wandb
from logger import OnePixelLogger, CWLogger
from utils import plot_grid, getYAMLParameter
attacks_config = read_yaml("./configs/attacks.yaml")





def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--adversarial_training", type=str, default="./configs/dataset.yaml", help="Path to the dataset configuration file.")
    args = parser.parse_args() 


    return args


def main(): 
    args = parse_args()

    # Read the yaml config file of the configuration of the attacks
    models_dict = {} # A Dictionary to hold the order of the models to execute
    attacks_config = read_yaml("./configs/attacks.yaml")
    local_checkpoint = attacks_config["model"]["local_model_path"]
    hugginface_model = attacks_config["model"]["hugginface_model"]
    enable_wandb = getYAMLParameter(attacks_config, "enable_wand")
    clip_enable = getYAMLParameter(attacks_config, "embedding_models", "clip_enable")
    embedding_dataset = getYAMLParameter(attacks_config, "embedding_models", "dataset")
    true_embeddings = None
    hugginface_dataset_dir = getYAMLParameter(attacks_config, "dataset", "dataset_path")
    dataset = load_dataset(hugginface_dataset_dir) 
    sample_dataset_number = getYAMLParameter(attacks_config, "dataset", "sample_number")
    dataset.shuffle(seed=42)
    images = dataset["train"]['image']
    label = dataset["train"]["labels"]


    if enable_wandb:
        wandb.login()

    if clip_enable:
        # Load CLIP model and processor
        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = AutoModel.from_config(config)



    if local_checkpoint and hugginface_model:
        raise Exception("""Choose either of the of the following configurations.
                 Local checkpoint and hugginface_model has been specified. Only one should be given""")
    if local_checkpoint:
        model = AutoModelForImageClassification()



    #Load the model from the checkpoint 
    model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint)
    # image_processor = ta.utils.ImageProcessor(model=model, device="cuda")


    if getYAMLParameter(attacks_config, "dataset", "train_on_dataset"):
        # Run the model and get the labels
        model.eval()
        # Get predictions on clean images
        label = model(images[:sample_dataset_number])



    # Attacks Base Configurations
    targeted = attacks_config["attack"]["targeted"]
    new_targeted_labels =attacks_config["attack"]["labels"]
    # One Pixel Attack Parameters
    enable_one_pixel_attack = attacks_config["one_pixel"]["enable_attack"]
    onepixel_steps = attacks_config["one_pixel"]["steps"]
    population = attacks_config["one_pixel"]["population_size"]
    num_pixels = attacks_config["one_pixel"]["pixels"]



    #  Carlini Weiner Parameters
    enable_cw = attacks_config["carlini_wiener"]["enable_attack"]
    steps_cw = attacks_config["carlini_wiener"]["steps"]
    kappa_cw = attacks_config["carlini_wiener"]["kappa"]



    # PGD Parameters
    enable_PGD = attacks_config["PGD"]["enable_attack"]
    steps_pgd = attacks_config["PGD"]["steps"]


    if enable_one_pixel_attack:
        one_pixel_attack = OnePixelLogger(model, num_pixels, onepixel_steps, population)

    if enable_cw:
        cw_attack = CWLogger(model=model, steps=steps_cw, kappa=kappa_cw)

    if enable_PGD:
        pgd_attack = ta.PGD(model, steps=steps_pgd)



    test_attack(one_pixel_attack, cw_attack, pgd_attack, images, labels= label, device="cuda", output_dir="./results", targeted=targeted)



def test_attack(images: torch.Tensor, labels: torch.Tensor,model: nn.Module, device: str, output_dir: str, targeted: bool, *attacks):
    print("Initializing Training of Attacks")
    for attack in attacks:
        if type(attack) == nn.Module:
            print(f"Testing {attack.__class__.__name__}...")
            for image, label in tqdm(zip(images, labels), total=len(images)):
                # Assuming the attack model has a method named `run` to execute the attack
                attack(model,image, label, device=device, targeted=targeted)
        else:
            print(f"Skipping invalid attack model: {type(attack)}")

    # Additional code to plot or save the results can be added here


if __name__ == "__main__":
    main()

    




