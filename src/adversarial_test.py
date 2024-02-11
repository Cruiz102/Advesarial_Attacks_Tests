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
# import tqdm
attacks_config = read_yaml("./configs/attacks.yaml")





def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--adversarial_training", type=str, default="./configs/dataset.yaml", help="Path to the dataset configuration file.")
    args = parser.parse_args() 


    return args


def main(): 
    args = parse_args()

    # Read the yaml config file of the configuration of the attacks
    attacks_config = read_yaml("./configs/attacks.yaml")
    local_checkpoint = attacks_config["model"]["local_model_path"]
    hugginface_model = attacks_config["model"]["hugginface_model"]
    if local_checkpoint and hugginface_model:
        raise Exception("""Choose either of the of the following configurations.
                 Local checkpoint and hugginface_model has been specified. Only one should be given""")
    if local_checkpoint:
        model = AutoModelForImageClassification()

    #Load the model from the checkpoint 
    model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint)
    enable_cw = attacks_config["carlini_wiener"]["enable_attack"]
    


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

    if attacks_config["dataset"]["train_on_dataset"]:
        dataset = load_dataset(args.dataset_config) 
        images =dataset["validation"]["image"]
        labels = dataset["validation"]["label"]


    if enable_one_pixel_attack:
        one_pixel_attack = ta.OnePixel(model= model,pixels= num_pixels, steps= onepixel_steps, popsize= population)

    if enable_cw:
        cw_attack = ta.CW(model=model, steps=steps_cw)

    if enable_PGD:
        pgd_attack = ta.PGD(model, steps=steps_pgd)


def test_attack(attacker: ta.Attack, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str, output_dir: str):
    """
    Test the given attack on the given model.
    """
    model.eval()
    attacker.set_mode_default()
    adversarial_images = attacker(images, labels)

    # Save the adversarial images in a jpeg
    # file.




if __name__ == "__main__":
    main()

    




