# A File for creating experiments graphs and results
# inside a folder called "results".

# We will give datasets that are going to be used.
# We will have to defined a way for reading yaml config
# files 

import matplotlib.pyplot as plt

import torchattacks as ta
from utils import read_yaml
import argparse 

# Read the yaml config file of the configuration of the attacks

attacks_config = read_yaml("./configs/attacks.yaml")


pgd_attack = ta.PGD(model, _)





def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--adversarial_training", type=str, default="./configs/dataset.yaml", help="Path to the dataset configuration file.")
    args = parser.parse_args() 


    return args


def main(): 
    args = parse_args()

    # Read the yaml config file of the configuration of the attacks
    attacks_config = read_yaml("./configs/attacks.yaml")

    #Load the model from the checkpoint 
    model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint)

    enable_one_pixel_attack = attacks_config["one_pixel"]["enable_attack"]
    enable_cw = attacks_config["carlini_wiener"]["enable_attack"]
    enable_PGD = attacks_config["PGD"]["enable_attack"]


    if enable_one_pixel_attack:
        one_pixel_attack = ta.OnePixelAttack(model, _)






