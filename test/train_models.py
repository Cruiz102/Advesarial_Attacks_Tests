import argparse
from transformers import AutoModelForImageClassification, AutoConfig, TrainingArguments, Trainer
from transformers import DefaultDataCollator, AutoFeatureExtractor
from datasets import load_dataset
import torch
import os


# Ensure wandb is installed: pip install wandb
import wandb


def save_checkpoint(model, output_dir, epoch):
    """Save model checkpoint in SafeTensor format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"checkpoint-{epoch}.safetensor")
    safetensor.save(model.state_dict(), output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for image classification.")
    parser.add_argument("--model_name_or_path", type=str, help="Model identifier from Huggingface Models (e.g., google/vit-base-patch16-224).")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset to be used for training.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to store the fine-tuned model.")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for experiment tracking.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load the model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    model = AutoModelForImageClassification.from_pretrained(args.model_name_or_path, config=config)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    # Prepare dataset
    datasets = load_dataset("imagefolder", data_dir=args.dataset_path)
    def transform(examples):
        examples['pixel_values'] = [feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze() for image in examples['image']]
        return examples
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
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=None,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=feature_extractor,
        data_collator=DefaultDataCollator(),
    )

    # Train and save the model
    trainer.train()
    model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)




    # Save the model and tokenizer in SafeTensor format at the end of training
    save_checkpoint(model, args.output_dir, epoch="final")
    feature_extractor.save_pretrained(args.output_dir)

    # Close W&B run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
