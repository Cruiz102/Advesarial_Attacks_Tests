## Project Overview
This project focuses on testing adversarial attacks across various models and datasets, specifically targeting image classification models. The core idea is to evaluate the resilience of these models against adversarial threats in a structured manner.

### Key Features
----
 - Adversarial Attack Testing: Systematic evaluation of image classification models against adversarial attacks.

 - Integration with [TorchAttacks](https://github.com/Harry24k/adversarial-attacks-pytorch/): Utilizes TorchAttacks for implementing adversarial attacks, serving as a seamless replacement for traditional methods.

 - Hugging Face Models: Leverages models from Hugging Face to train on specific datasets and assess their performance under adversarial conditions.

 - Live Inference Script: Includes a script for real-time model inference using a webcam, allowing for dynamic testing environments.


### Using Accelerate 

Accelerate is a very easy to use framework for optimizing the training Models on efficient training pipelines.
For using accelerate make sure to install it with pip3 install accelerate.

Before running accelerate with the Hugginface Trainer you first have to configure your training pipeline by running:
```bash
accelerate config
```

### Contributing
---
In general we would want for the project to have the following things , so any contributions to this areas will be well welcome.

- Configuration Flexibility: Aim to enhance project configurability using files like YAML, making it easily adaptable for different users and scenarios.

- Feature Enhancements: Contributions that introduce new functionalities or features to improve adversarial attack testing are highly encouraged. Feel free to create an issue to discuss your ideas.


Exporting TensorRT engine.

For exporting a trt model and use it in the live_inference script you first must ensure that you  have CUDA and tensorRT installed in your machine.

# Getting Started:
 To run this repository you will need first to install python and their dependencies.

 1. Clone the repository
 2. Install the dependencies using pip3 install -r requirements.txt
  
```bash
git clone https://github.com/Cruiz102/Advesarial_Attacks_Tests.git

cd Advesarial_Attacks_Tests

pip3 install -r requirements.txt

```

## Testing Adversarial Attacks
(Disclaimer it could still have some issues. SORRY. Im going to fix the bugs and problems soon :))

For now if you want to see the results of running the adversarial attacks you will need to use Weight and Biases and and sign up with your account.

For accessing weight and biases and create an account you can click on the following link: [Weight and biases](https://wandb.ai/site)

Once you create an account you can log into it by running the following command(make sure to install the libraries first!!):

```bash
wandb login
```

after loging into your account you are ready to run your first attack.

For running an attack you will have to create a configuration file that will have the following structure. Remember to enable wandb for logging the results.

```yaml
enable_wandb: True

model:
  name: 'VIT_model' 
  hugginface_model: "google/vit-base-patch16-224"
  use_preprocessor: True
  resize_size: 224

dataset:
  train_on_dataset: True # If train on dataset is true it will use the true labels from the dataset. If it is set to False
                         # it will run the model with the images and generate pseudo labels to use for training.
  dataset_path: "mrm8488/ImageNet1K-val"
  sample_number: 20  # Number of samples to use from the dataset for the evaluation


attack:
  targeted: False
  target_list: [(),()] 
  
  

one_pixel:
  enable_attack: True
  steps: 10
  pixels : 1
  population_size: 100
```


After configuring your yaml file with the models, dataset and hyperparameters you want to use  the next step is to run the script for runnning the attacks.


```bash
python3 attack_test.py --config_path=config.yaml
```


Congratulations you are now running your first attack, you will see in you terminal how the attack is being trained. After finishing the attack you will see the results in your wandb account.