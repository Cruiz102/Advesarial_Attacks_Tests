from tqdm import tqdm
import torchattacks as ta
from torchattacks.attacks._differential_evolution import differential_evolution
import math
import torchvision.utils as vutils
import torch.nn.functional as F
import wandb
import torch
import torch.nn as nn
import torch.optim.optimizer as optim
import numpy as np
from  transformers.modeling_outputs import ImageClassifierOutput, ImageClassifierOutputWithNoAttention
from typing import Tuple
from utils import  l2_distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
from PIL import Image

#  The goal in here will be to implement the loops of the attacks
#  with tqdm and wandb.

class AttackLogger:
    def __init__(self, project_name, wandb_config) -> None:
        self.project_name = project_name
        self.wandb_config = wandb_config
        self.batch_counter = 0
    
    def init_wandb(self):
        # Initialize wandb run
        wandb.init(project=self.project_name,
                   name=self.wandb_config["attack"] + "-" + self.wandb_config["model_name"],
                   config = self.wandb_config)
        

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        if isinstance(logits, ImageClassifierOutput) or isinstance(logits,ImageClassifierOutputWithNoAttention ):
            logits = logits.logits
        return logits
    

    def get_accuracy(self, images,adv_images, labels):
        batched_predictions = self.model(adv_images).logits.argmax(dim=1)
        original_predictions = self.model(images).logits.argmax(dim=1)
        adv_accurary = batched_predictions == labels
        num_sucesses = adv_accurary.sum().item()
        original_accuracy = original_predictions == labels
        num_og_sucesses = original_accuracy.sum().item()

        return num_sucesses,num_og_sucesses, batched_predictions, original_predictions

    def forward(self, images, labels):
        "This function should alwyays return the  following information"
        pass



class OnePixelLogger(ta.OnePixel, AttackLogger):
    def __init__(self,project_name,model,wandb_config=None, *args, **kwargs):

        AttackLogger.__init__(self, project_name=project_name, wandb_config=wandb_config)
        # Forward arguments to the Parent class
        super().__init__(model=model,*args, **kwargs)
    # Wrap the attack call with tqdm for a progress bar
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        if isinstance(logits, ImageClassifierOutput) or isinstance(logits,ImageClassifierOutputWithNoAttention ):
            logits = logits.logits
        return logits
        
    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.get_logits(batch)
                outs.append(out)
        outs = torch.cat(outs)
        return outs.detach().cpu().numpy()
    


    def forward(self, images, labels) -> Tuple[torch.Tensor, int, int]:
        # Initialize wandb run
        print(f"Batch Counter: {self.batch_counter}")
        self.batch_counter += 1

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        print("devices", images.device, labels.device)
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        batch_size, channel, height, width = images.shape

        bounds = [(0, height), (0, width)] + [(0, 1)] * channel
        bounds = bounds * self.pixels

        popmul = max(1, int(self.popsize / len(bounds)))

        adv_images = []
        for idx in tqdm(range(batch_size), desc="Processing Images"):
            image, label = images[idx : idx + 1], labels[idx : idx + 1]

            if self.targeted:
                target_label = target_labels[idx : idx + 1]

                def func(delta):
                    return self._loss(image, target_label, delta)

                def callback(delta, convergence):
                    success = self._attack_success(image, target_label, delta)
                    wandb.log({ "Convergence": convergence})
                    return success

            else:

                def func(delta):
                    return self._loss(image, label, delta)

                def callback(delta, convergence):
                    success = self._attack_success(image, label, delta)
                    wandb.log({ "Convergence": convergence})
                    return success

            delta = differential_evolution(
                func=func,
                bounds=bounds,
                callback=callback,
                maxiter=self.steps,
                popsize=popmul,
                init="random",
                recombination=1,
                atol=-1,
                polish=False,
            ).x
            delta = np.split(delta, len(delta) / len(bounds))
            adv_image = self._perturb(image, delta)
            adv_images.append(adv_image)
        adv_images = torch.cat(adv_images)
    #  Get the label from batched prediction with the hisgtest probability

        num_sucesses, num_og_sucesses,batched_predictions, original_predictions = self.get_accuracy(images,adv_images, labels)
        return adv_images, num_sucesses,num_og_sucesses, batched_predictions, original_predictions






class FGSMLogger(ta.FGSM, AttackLogger):

    def __init__(self,project_name,model,wandb_config=None, eps=8 / 255, *args, **kwargs):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        AttackLogger.__init__(self, project_name=project_name, wandb_config=wandb_config)
        # Forward arguments to the Parent class
        super().__init__(model=model,eps=eps, *args, **kwargs)

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        if isinstance(logits, ImageClassifierOutput) or isinstance(logits,ImageClassifierOutputWithNoAttention ):
            logits = logits.logits
        return logits

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        num_sucesses, num_og_sucesses,batched_predictions, original_predictions = self.get_accuracy(images,adv_images, labels)
        return adv_images, num_sucesses,num_og_sucesses, batched_predictions, original_predictions
    


class PGDLogger(ta.PGD, AttackLogger):
    def __init__(self, project_name,model,wandb_config= None, eps=8 / 255, alpha=2 / 255, steps=10, *args, **kwargs):
        AttackLogger.__init__(self, project_name=project_name, wandb_config=wandb_config)
        # Initialize the ta.PGD base class
        ta.PGD.__init__(self, model=model, eps=eps, alpha=alpha, steps=steps, *args, **kwargs)

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        if isinstance(logits, ImageClassifierOutput) or isinstance(logits,ImageClassifierOutputWithNoAttention ):
            logits = logits.logits
        return logits

    def forward(self, images, labels):
            r"""
            Overridden.
            """

            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            if self.targeted:
                target_labels = self.get_target_label(images, labels)

            loss = nn.CrossEntropyLoss()
            adv_images = images.clone().detach()

            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)
                if type(outputs) == ImageClassifierOutput:
                    outputs = outputs.logits
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                wandb.log({"loss": cost.item()})
                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()


            num_sucesses, num_og_sucesses,batched_predictions, original_predictions = self.get_accuracy(images,adv_images, labels)
            return adv_images, num_sucesses,num_og_sucesses, batched_predictions, original_predictions
        

    



class CWLogger(ta.CW, AttackLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# TODO: Implement the forward loop with tqdm and wandb
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                condition = (pre == target_labels).float()
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images
    