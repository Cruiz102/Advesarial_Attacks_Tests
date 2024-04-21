## Project Overview
This project focuses on testing adversarial attacks across various models and datasets, specifically targeting image classification models. The core idea is to evaluate the resilience of these models against adversarial threats in a structured manner.

### Key Features
----
 - Adversarial Attack Testing: Systematic evaluation of image classification models against adversarial attacks.

 - Integration with [TorchAttacks](https://github.com/Harry24k/adversarial-attacks-pytorch/): Utilizes TorchAttacks for implementing adversarial attacks, serving as a seamless replacement for traditional methods.

 - Hugging Face Models: Leverages models from Hugging Face to train on specific datasets and assess their performance under adversarial conditions.

 - Live Inference Script: Includes a script for real-time model inference using a webcam, allowing for dynamic testing environments.


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

For now if you want to see the results of running the adversarial attacks you will need to use Weight and Biases and and sign up with your account.

For accessing weight and biases and create an account you can click on the following link: [Weight and biases](https://wandb.ai/site)

Once you create an account you can log into it by running the following command(make sure to install the libraries first!!):

```bash
wandb login
```

after loging into your account you are ready to run your first attack.

For running an attack you will have to create an configuration file that will have the information of the attacks you want to execute. This file should be named `config.yml` and must be inside the `config` folder. The following is the default example to try out the tool.

```yaml
enable_gpu: true

model:
  name: 'Resnet Microsoft' #the Name you want to give to the model
  hugginface_model: "microsoft/resnet-50"
  batch_size : 16 # Batch size for trainin. If you have problems with memory, you can use a lower batch size.

dataset:
  dataset_path: "mrm8488/ImageNet1K-val"
  sample_number: 50  # Number of samples to use from the dataset for the evaluation
  random_seed: 2 # If you want  perform the test with the same data each time, set a random seed not equal to 0.
  image_feature_title: "image" #Check on the specification of the dataset to see the name of the feature that contains the image
  label_feature_title: "label"
  


embedding_models:
#  Only enable this one if you want to use the CLIP model.
# The model_name specified in the model category will not be used.
  clip_model_enable : False

one_pixel:
  enable_attack: False
  steps: 20
  pixels : 1
  population_size: 100

FGSM:
  enable_attack: True
  epsilon: 0.005

PGD:
  enable_attack: False
  epsilon: 0.3
  alpha: 0.1
  steps: 100

carlini_weiner:
  enable_attack: False
  kappa: 0.01
  steps: 10000 




```

# Run the Attacks
After configuring your yaml file with the models, dataset and hyperparameters you want to use  the next step to run the attacks.


```bash
python3 src/attack_test.py 
```
Congratulations you are now running your first attack, you will see in you terminal how the attack is being trained. After finishing the attack you will see the results in your wandb account.




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