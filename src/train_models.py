import argparse
from transformers import AutoModelForImageClassification, AutoConfig, TrainingArguments, Trainer, AutoModel
from transformers import DefaultDataCollator, AutoImageProcessor
from datasets import load_dataset
import torch
import os
# import wandb
import safetensors as safetensor
import tqdm
import typing
import yaml

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
    parser.add_argument("--dataset_config", type=str, default="./configs/dataset.yaml", help="Path to the dataset configuration file.")
    parser.add_argument("--model_config", type=str, default="./configs/model.yaml", help="")
    parser.add_argument("--training_config", type=str, default="./configs/training_config.yaml", help="")
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


# Training Params
    model_checkpoint_path: str = training_config["model_checkpoint"]
    output_dir: str = training_config["output_dir"] 

    if not model_name and not model_checkpoint_path:
        raise Exception("""Neither a hugginface Pretrained model or a Checkpoint has been specified. Please in your model configuration
                            specify the hugginface model_name or specify a checkpoint to start the training.""")
    

    

    # Load the model and tokenizer
    config = AutoConfig.from_pretrained("google/vit-base-patch16-224", num_labels=args.num_labels)
    pretrained_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
    classification_model = AutoModelForImageClassification.from_config("google/vit-base-patch16-224", config=config)
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Prepare dataset
    datasets = load_dataset(, data_dir=args.dataset_path)
    
    datasets = datasets.map(transform, batched=True)
    datasets.set_format(type='torch', columns=['pixel_values', 'label'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        push_to_hub=False,
        report_to=""
    )

    # Initialize Trainer
    trainer = Trainer(
        model=classification_model,
        args=training_args,
        compute_metrics=None,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=DefaultDataCollator(),
    )

    # Train and save the model
    trainer.train()
    classification_model.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)




    # Save the model and tokenizer in SafeTensor format at the end of training
    save_checkpoint(model, args.output_dir, epoch="final")
    feature_extractor.save_pretrained(args.output_dir)

    # # Close W&B run
    # if args.use_wandb:
    #     wandb.finish()


if __name__ == "__main__":
    main()
