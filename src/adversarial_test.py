# A File for creating experiments graphs and results
# inside a folder called "results".

# We will give datasets that are going to be used.
# We will have to defined a way for reading yaml config
# files 

import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoConfig, TrainingArguments, Trainer, AutoModel, AutoImageProcessor
from transformers import CLIPProcessor, CLIPModel
from utils import read_yaml, getYAMLParameter, HugginfaceProcessorData
from datasets import load_dataset
from torch.utils.data import DataLoader
from logger import OnePixelLogger, CWLogger, PGDLogger,  FGSMLogger
from utils import plot_grid, getYAMLParameter, l2_distance, plot_image_comparison, conf_matrix_img, count_parameters
import tqdm
import wandb
import math
import argparse 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from clip_classifier import clip_classifier
from image_net_classes import IMAGENET2012_CLASSES



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--adversarial_training", type=str, default="./configs/dataset.yaml", help="Path to the dataset configuration file.")
    args = parser.parse_args() 


    return args


def infere_titles(titles):
    """
    This function exist for classyfying on different names of titles.
    it will raise an error if  there is an ambigious titles and the program
    will expect that the user specify the titles of the models.
    """
    import nltk
    from nltk.tokenize import word_tokenize

# Download the NLTK data needed for tokenization
    nltk.download('punkt')

    titles_list = []
    for title in titles:
        title = title.split("/")[-1]
        title = title.split(".")[0]
        titles_list.append(title)
    return titles_list

def main(): 
    args = parse_args()

    # Read the yaml config file of the configuration of the attacks
    attacks_config = read_yaml("config/attacks_config.yml")
    local_checkpoint = getYAMLParameter(attacks_config, "model", "local_model_path")
    hugginface_model = getYAMLParameter(attacks_config, "model", "hugginface_model")
    use_preprocessor = getYAMLParameter(attacks_config, "model", "use_preprocessor")
    enable_wandb = getYAMLParameter(attacks_config, "enable_wand")
    enable_gpu = getYAMLParameter(attacks_config, "enable_gpu")
    clip_enable = getYAMLParameter(attacks_config, "embedding_models", "clip_model_enable")
    embedding_dataset = getYAMLParameter(attacks_config, "embedding_models", "dataset")
    true_embeddings = None
    hugginface_dataset_dir = getYAMLParameter(attacks_config, "dataset", "dataset_path")
    batch_size = getYAMLParameter(attacks_config, "model", "batch_size")
    sample_dataset_number = getYAMLParameter(attacks_config, "dataset", "sample_number")
    train_on_dataset = getYAMLParameter(attacks_config, "dataset", "train_on_dataset")
    image_feature_title  = getYAMLParameter(attacks_config, "dataset", "image_feature_title")
    label_feature_title = getYAMLParameter(attacks_config, "dataset", "label_feature_title")
    random_seed = getYAMLParameter(attacks_config, "dataset", "random_seed")


    highlight_enable = getYAMLParameter(attacks_config, "attack", "highlight_effective_attack")


    dataset = load_dataset(hugginface_dataset_dir) 

    
    if sample_dataset_number:
        if random_seed  > 0:
            dataset = dataset['train'].shuffle(random_seed).select(range(sample_dataset_number))
        else:
            dataset = dataset['train'].shuffle().select(range(sample_dataset_number))

    # Check if the dataset has a "label" column

    labels_names = dataset.features[label_feature_title].names

    import re
    # Check if the dataset is imagenet
    imagenet_pattern = re.compile(r'imagenet', re.IGNORECASE)
    is_imagenet = imagenet_pattern.search(hugginface_dataset_dir)
    if is_imagenet:
        print("'imagenet' was found.")
        labels_names = [ IMAGENET2012_CLASSES[i] for i in labels_names ]

    
    if enable_wandb:
        wandb.login()

    if enable_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if local_checkpoint and hugginface_model:
        raise Exception("""Choose either of the of the following configurations.
                 Local checkpoint and hugginface_model has been specified. Only one should be given""")
    if local_checkpoint:
        model = AutoModelForImageClassification()  


    if not train_on_dataset:
        labels = model(dataset["train"]["images"])
        labels = labels.logits.argmax(dim=1)


    #Load the model from the checkpoint 
    model = AutoModelForImageClassification.from_pretrained(hugginface_model).to(device)
    image_processor = AutoImageProcessor.from_pretrained(hugginface_model)



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


    # FGSM Parameters
    enable_FGSM = getYAMLParameter(attacks_config, "FGSM", "enable_attack")
    epsilon_fgsm = getYAMLParameter(attacks_config, "FGSM", "epsilon")

    # PGD Parameters
    enable_PGD = getYAMLParameter(attacks_config, "PGD", "enable_attack")
    steps_pgd = getYAMLParameter(attacks_config, "PGD", "steps")
        


    if clip_enable:
        print("Loading CLIP model")
        hugginface_model = "openai/clip-vit-base-patch32"
        model  = clip_classifier(clip_model_name=hugginface_model,
                                  labels_name=labels_names, device = device)
        image_processor = CLIPProcessor.from_pretrained(hugginface_model)
        
        def clip_preprocess(examples):
            # Process the images
            inputs = image_processor( images=examples[image_feature_title], return_tensors="pt",padding = True).data
            examples["pixel_values"] = inputs["pixel_values"]
            # examples["input_ids"] = inputs["input_ids"]
            # examples["attention_mask"] = inputs["attention_mask"]
            return examples
        
        processed_dataset =  dataset.map(clip_preprocess, batched=True).with_format("torch")
    else:
        def preprocess_images(examples):
                    # Process the images
                    images = [image.convert('RGB') for image in examples['image']] 
                    examples["pixel_values"] = image_processor(images=images, padding = True)["pixel_values"]
                    return examples
        # Apply preprocessing
        processed_dataset = dataset.map(preprocess_images, batched=True, ).with_format("torch")

    # from custom_models import DenseModel
    # model = DenseModel(224*224*3,300,500,1000)

    one_pixel_attack = (False, None)
    fgsm_attack = (False, None)
    cw_attack = (False, None)
    pgd_attack = (False, None)
    if enable_one_pixel_attack:
        wandb_config_one_pixel = {
        "targeted": targeted,
        "targeted_labels": targeted_labels,
        "population": population,
        "steps": onepixel_steps,
        "num_pixels": num_pixels,
        "dataset": hugginface_dataset_dir,
        "model_name": hugginface_model,
        "parameter_count": count_parameters(model),
        "batch_size": batch_size,
        "attack": "one_pixel"
    }
        one_pixel_attack = (enable_one_pixel_attack, OnePixelLogger(project_name= "one_pixel_attack",
                                          model=model,
                                           wandb_config=wandb_config_one_pixel,
                                           popsize = population,
                                            pixels = num_pixels,
                                             steps = onepixel_steps )
                                        )
        

    if enable_FGSM:

        wandb_config_fgsm = {
        "targeted": targeted,
        "targeted_labels": targeted_labels,
        "dataset": hugginface_dataset_dir,
        "model_name": hugginface_model,
        "epsilon": epsilon_fgsm,
        "parameter_count": count_parameters(model),
        "batch_size": batch_size,
        "attack": "fgsm"
    }
        fgsm_attack = (enable_FGSM , FGSMLogger(
                                project_name= "FGSM_attack",
                                model=model,
                                wandb_config = wandb_config_fgsm,
                                eps=epsilon_fgsm)
                                                    )

    if enable_cw:
        cw_attack = (enable_cw, CWLogger(model=model, steps=steps_cw, kappa=kappa_cw))

    if enable_PGD:
        pgd_attack = (enable_PGD, PGDLogger(model, steps=steps_pgd))








    all_attacks = [one_pixel_attack, fgsm_attack, cw_attack, pgd_attack]
    processed_dataset = HugginfaceProcessorData(processed_dataset, label_feature_title)
    # Start  the Attack
    test_attack(
    dataset=processed_dataset, 
    batch_size= batch_size, 
    output_dir="./results", 
    targeted=targeted,
    attacks= [attack for  enable , attack in all_attacks if enable], 
    hightlight= highlight_enable,
    device=device
    )




def test_attack(dataset, batch_size,  output_dir: str, targeted: bool, attacks = [], device: str = "cpu", hightlight = True):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    image_title = dataset.image_feature_title
    label_title = dataset.label_title
    used_labels = set()
    

    total_batches = len(dataloader)
    save_interval = max(1, math.ceil(total_batches / 10))  # Ensure at least 1 to avoid division by zero

    print("Initializing Training of Attacks")
    for attack in attacks:
        print("Attack Name: ", attack)
        batch_counter = 0
        total_sucesses = 0
        total_og_sucesses = 0
        y_total_pred = torch.tensor([])
        y_og_total_pred = torch.tensor([])
        y_total_labels = torch.tensor([])
        logits_attack_total = torch.tensor([])
        attack.init_wandb()
        for batch in dataloader:

            images = batch[image_title]
            labels = batch[label_title]
            adv_images, num_attack_sucesses,num_og_sucesses, batched_pred, original_pred = attack(images,labels)
            batched_pred = batched_pred.cpu()
            original_pred = original_pred.detach().cpu()


            # Save the used labels
            for  i in labels:
                used_labels.add(i.item())





            l2_distance_metric = l2_distance(batched_pred,images, adv_images, labels, device)
            wandb.log({"L2 Distance": l2_distance_metric})


            wandb.log({"Attacked_Batch_Accuracy": num_attack_sucesses/len(images)})
            wandb.log({"Original_Batch Accuracy": num_og_sucesses/len(images) })



            y_total_pred = torch.cat((y_total_pred, batched_pred) )
            y_og_total_pred = torch.cat((y_og_total_pred, original_pred) )
            y_total_labels = torch.cat((y_total_labels, labels))

            # TODO: Implement this to have an ROC curve
            # logits_attack = torch.cat(logits_attack_total, logits_attk.view(logits_attk.shape[1], -1), dim=1)
            total_sucesses += num_attack_sucesses
            total_og_sucesses += num_og_sucesses

            # Check if it's time to save grid images
            if batch_counter % save_interval == 0:
                comparison_image = plot_image_comparison(images,adv_images,original_pred, batched_pred, labels, hightlight, 4)
                wandb.log({"Comparison Image": [wandb.Image(comparison_image)]})
            batch_counter += 1

        attacked_value = total_sucesses / len(dataset)  # Assuming total_sucesses and dataset are defined
        original_value = total_og_sucesses / len(dataset)  # Assuming total_og_sucesses and dataset are defined

        # Create a wandb.Table with explicit columns
        table = wandb.Table(columns=["Category", "Value"])

        # Add data rows to the table
        table.add_data("Attacked", attacked_value)
        table.add_data("Original", original_value)

        bar_chart = wandb.plot.bar(table, "Category", "Value", title="Total Successes Percentage")
        wandb.log({"Total Successes Percentage": bar_chart})
        #Create and display the confusion matrix
        confusion_img_attk = conf_matrix_img(y_total_pred.flatten(),y_total_labels.flatten())
        confusion_img_og = conf_matrix_img(y_og_total_pred.flatten(),y_total_labels.flatten())
        # Log the confusion matrix image file to wandb
        wandb.log({"confusion_matrix_attacked_images": wandb.Image(confusion_img_attk)})
        wandb.log({"confusion_matrix_original_images": wandb.Image(confusion_img_og)})



        # # ROC Curve
        # wandb.log({"ROC Curve Attacked" : wandb.plot.roc_curve(y_total_labels.flatten(),
        #                          logits_attack, labels=list(used_labels))})
                                 
        wandb.finish()
if __name__ == "__main__":
    main()

    




