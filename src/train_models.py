import argparse
from transformers import AutoModelForImageClassification, AutoConfig, TrainingArguments, Trainer, AutoModel
from transformers import DefaultDataCollator, AutoImageProcessor
from datasets import load_dataset
import torch
import os
import wandb
import safetensors as safetensor
import tqdm
import typing
import yaml
from rai_toolbox.optim import ParamTransformingOptimizer
from accelerate import Accelerator

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, output_dir, epoch):
    """Save model checkpoint in SafeTensor format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"checkpoint-{epoch}.safetensor")
    safetensor.save(model.state_dict(), output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--dataset_config", type=str, default="./config/coco_datasets.yml", help="Path to the dataset configuration file.")
    parser.add_argument("--model_config", type=str, default="./config/models.yml", help="")
    parser.add_argument("--training_config", type=str, default="./config/training_config.yml", help="")
    args = parser.parse_args()
    return args


def create_model_classifier_from_image_classification(model:AutoModelForImageClassification, config):
    """Create a model classifier from an image classification model."""
    # Create a new classifier head
    classifier_head = torch.nn.Linear(config.hidden_size, config.num_labels)
    # Replace the classifier head
    model.classifier = classifier_head
    #Initialize again the weights
    model.post_init()
    return model



def main():
    args = parse_args()

    # model configurations
    model_config = read_yaml(args.model_config)
    dataset_config = read_yaml(args.dataset_config)
    training_config = read_yaml(args.training_config)


    # Models Params and checking    
    hugginface_model: str = model_config["hugginface_model"]
    model_name : str = model_config["model_name"]


    #Dataset Params
    dataset_dir: str = dataset_config["dataset_dir"]
    dataset_name: str = dataset_config["dataset_name"]
    class_numbers: int = dataset_config["class_numbers"]
    class_names: typing.List[str] = dataset_config["class_names"]
    training_distribution: int = dataset_config["base"]["training_distri"]
    val_distribution :int = dataset_config["base"]["val_distri"]
    test_distribution: int = dataset_config["base"]["test_distri"]



# Training Params
    model_checkpoint_path: str = training_config["model_checkpoint"]
    train_batch_size : int = training_config["base"]["train_batch_size"]
    epochs: int = training_config["base"]["epochs"]
    output_dir: str = training_config["output_dir"] 
    enable_wand: bool = training_config["wandb"]["enable_wandb"]
    activate_accelerate = training_config["base"]["activate_accelerate"]
    reporter = None
    hugginface_dataset = training_config["hub"]["hugginface_dataset"]


    if enable_wand:
        reporter = "wandb"
        wandb.login()

    if activate_accelerate:
        # Initialize Accelerator
        accelerator = Accelerator()
    

    if not model_name and not model_checkpoint_path:
        raise Exception("""Neither a hugginface Pretrained model or a Checkpoint has been specified. Please in your model configuration
                            specify the hugginface model_name or specify a checkpoint to start the training.""")
    
    if hugginface_dataset and dataset_dir:
        raise Exception("""Choose either of the of the following configurations.
                     Dataset directory and hugginface_dataset has been specified. Only one should be given""")
    
    if training_distribution + val_distribution + test_distribution!= 1:
        raise Exception("""The sum of the training, validation and test distribution should be equal to 1""")
    

    # Load the model and tokenizer
    config = AutoConfig.from_pretrained("google/vit-base-patch16-224", num_labels=args.num_labels)
    pretrained_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
    classification_model = AutoModelForImageClassification.from_config("google/vit-base-patch16-224", config=config)
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Prepare dataset
    # Check wheter the dataset is in json, csv or txt format
    # Load the dataset
    if dataset_dir.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_dir)
    elif dataset_dir.endswith(".csv"):
        dataset = load_dataset("csv", data_files=dataset_dir)
    elif dataset_dir.endswith(".txt"):
        dataset = load_dataset("text", data_files=dataset_dir)
    else:
    # Hugginface dataset
        dataset = load_dataset(dataset_dir)

#Check in this function wheter al features are not only numbers
# because if it is probrably it didnt got the correct format
# make a warning
    if dataset.features["label"].dtype!= "int64":
        Warning("The label is not an integer")
        

    def preprocess_images(examples):
        return image_processor(images=examples["image"], return_tensors="pt")
#

    # Apply preprocessing
    preprocess_images_before_training = True
    if preprocess_images_before_training:
        processed_dataset = dataset.map(preprocess_images, batched=True)
        processed_dataset.set_format(type="torch", columns=["pixel_values", "label"])

    # Save the Data

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        # logging_dir='./logs',
        # logging_steps=10,
        push_to_hub=False,
        report_to=reporter
    )

    # Initialize Trainer
    trainer = Trainer(
        model=classification_model,
        args=training_args,
        compute_metrics=None,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DefaultDataCollator(),
    )


    if activate_accelerate:
        # Use Accelerate's prepare method
        trainer, model, dataset["train"], dataset["validation"] = accelerator.prepare(
    trainer, model, dataset["train"], dataset["validation"]
)
    
    # Train and save the model
    trainer.train()
    classification_model.save_pretrained(args.output_dir)
    



if __name__ == "__main__":
    main()
