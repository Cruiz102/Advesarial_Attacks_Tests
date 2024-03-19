# A File for creating experiments graphs and results
# inside a folder called "results".

# We will give datasets that are going to be used.
# We will have to defined a way for reading yaml config
# files 

import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoConfig, TrainingArguments, Trainer, AutoModel, AutoImageProcessor
from utils import read_yaml, getYAMLParameter, HugginfaceProcessorData
from datasets import load_dataset
from torch.utils.data import DataLoader
from logger import OnePixelLogger, CWLogger, PGDLogger
from utils import plot_grid, getYAMLParameter, one_hot_encode
import tqdm
import wandb
import math
import argparse 
import torch
import torch.nn as nn



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--adversarial_training", type=str, default="./configs/dataset.yaml", help="Path to the dataset configuration file.")
    args = parser.parse_args() 


    return args


def main(): 
    args = parse_args()

    # Read the yaml config file of the configuration of the attacks
    models_dict = {} # A Dictionary to hold the order of the models to execute
    attacks_config = read_yaml("config/attacks_config.yml")
    local_checkpoint = getYAMLParameter(attacks_config, "model", "local_model_path")
    hugginface_model = getYAMLParameter(attacks_config, "model", "hugginface_model")
    use_preprocessor = getYAMLParameter(attacks_config, "model", "use_preprocessor")
    enable_wandb = getYAMLParameter(attacks_config, "enable_wand")
    enable_gpu = getYAMLParameter(attacks_config, "enable_gpu")
    clip_enable = getYAMLParameter(attacks_config, "embedding_models", "clip_enable")
    embedding_dataset = getYAMLParameter(attacks_config, "embedding_models", "dataset")
    true_embeddings = None
    hugginface_dataset_dir = getYAMLParameter(attacks_config, "dataset", "dataset_path")
    dataset = load_dataset(hugginface_dataset_dir) 
    sample_dataset_number = getYAMLParameter(attacks_config, "dataset", "sample_number")
    dataset.shuffle(seed=42)


    if sample_dataset_number:
        dataset = dataset['train'].select(range(sample_dataset_number))

    # Check if the dataset has a "label" column
    if "label" in dataset.features.keys():
        # Get the labels and their corresponding names
        labels = dataset.features["label"].names

    
    if enable_wandb:
        wandb.login()

    if enable_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if clip_enable:
        # Load CLIP model and processor
        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = AutoModel.from_config(config)
        clip_model.to(device)



    if local_checkpoint and hugginface_model:
        raise Exception("""Choose either of the of the following configurations.
                 Local checkpoint and hugginface_model has been specified. Only one should be given""")
    if local_checkpoint:
        model = AutoModelForImageClassification()


    #Load the model from the checkpoint 
    model = AutoModelForImageClassification.from_pretrained(hugginface_model).to(device)
    image_processor = AutoImageProcessor.from_pretrained(hugginface_model)

    def preprocess_images(examples):
        # Process the images
        examples["pixel_values"] = image_processor(images=examples["image"], return_tensors="pt")["pixel_values"]
        # examples["label"] = one_hot_encode(examples["label"], num_classes=len(labels))    
        return examples

    # Apply preprocessing
    processed_dataset = dataset.map(preprocess_images, batched=True, remove_columns=["image"]).with_format("torch")
    # if isinstance([0], int):
    # # Converting Labels to their one hot encoded representation
    #     one_hot_encode(processed_dataset[], num_classes=len(processed_dataset.features["label"].names))


    # Attacks Base Configurations
    targeted = getYAMLParameter(attacks_config, "attack", "targeted")
    targeted_labels = getYAMLParameter(attacks_config, "attack", "labels")
    # One Pixel Attack Parameters
    enable_one_pixel_attack = getYAMLParameter(attacks_config, "one_pixel", "enable_attack")
    onepixel_steps = getYAMLParameter(attacks_config, "one_pixel", "steps")
    population = getYAMLParameter(attacks_config, "one_pixel", "population_size")
    num_pixels = getYAMLParameter(attacks_config, "one_pixel", "pixels")



    #  Carlini Weiner Parameters
    enable_cw = getYAMLParameter(attacks_config, "carlini_wiener", "enable_attack")
    steps_cw = getYAMLParameter(attacks_config, "carlini_wiener", "steps")
    kappa_cw = getYAMLParameter(attacks_config, "carlini_wiener", "kappa")

    # PGD Parameters
    enable_PGD = getYAMLParameter(attacks_config, "PGD", "enable_attack")
    steps_pgd = getYAMLParameter(attacks_config, "PGD", "steps")
        
    # model = DenseModel(224*224*3,300,500,1000)



    if enable_one_pixel_attack:
        wandb_config_one_pixel = {
        "targeted": targeted,
        "targeted_labels": targeted_labels,
        "population": population,
        "steps": onepixel_steps,
        "num_pixels": num_pixels,
        "attack": "one_pixel"
    }

        one_pixel_attack = OnePixelLogger("OnePixelAttack",wandb_config_one_pixel,model, num_pixels, onepixel_steps, population)

    if enable_cw:
        cw_attack = CWLogger(model=model, steps=steps_cw, kappa=kappa_cw)

    if enable_PGD:
        pgd_attack = PGDLogger(model, steps=steps_pgd)


    dataset = HugginfaceProcessorData(dataset)
    

    if use_preprocessor: 
        processed_dataset = HugginfaceProcessorData(processed_dataset)
        test_attack(
        dataset=processed_dataset, 
        batch_size=200, 
        output_dir="./results", 
        targeted=targeted,
        attacks=[one_pixel_attack]
    )
    else:
        pass
    #     test_attack(
    #     dataset=dataset,
    #     batch_size=200,
    #     device="cuda",
    #     output_dir="./results",
    #     targeted=targeted,
    #     attacks=[one_pixel_attack]
    # )





def test_attack(dataset, batch_size,  output_dir: str, targeted: bool, attacks = []):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(dataloader)
    save_interval = max(1, math.ceil(total_batches / 10))  # Ensure at least 1 to avoid division by zero

    print("Initializing Training of Attacks")
    for attack in attacks:
        print("Attack Name: ", attack)
        batch_counter = 0
        total_sucesses = 0

        for batch in dataloader:
            print("Batch: ", batch["pixel_values"].shape)
            adv_images, sucess_attacks = attack(batch["pixel_values"], batch["label"])
            total_sucesses += sucess_attacks
            # Check if it's time to save grid images
            if batch_counter % save_interval == 0:
                pass
            
            batch_counter += 1

            # Stop if you've saved 10 sets of images
            if batch_counter // save_interval >= 10:
                break
        wandb.log({"Attack Successes": total_sucesses})

if __name__ == "__main__":
    main()

    




