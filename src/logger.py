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
from  transformers.modeling_outputs import ImageClassifierOutput
from typing import Tuple
from utils import clean_accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#  The goal in here will be to implement the loops of the attacks
#  with tqdm and wandb.

class AttackLogger:
    def __init__(self, project_name, wandb_config) -> None:
        self.wandb_config = wandb_config
        self.project_name = project_name
    
    def init_wandb(self):
        # Initialize wandb run
        wandb.init(project=self.project_name,
                   name=self.wandb_config["name"],
                   config = self.wandb_config)



class OnePixelLogger(ta.OnePixel, AttackLogger):
    def __init__(self,project_name,wandb_config=None, *args, **kwargs):
        # Forward arguments to the Parent class
        super().__init__(*args, **kwargs)
        self.image_counter = 0
        self.attack_success_counter = 0
        self.project_name = project_name
        self.wandb_config = wandb_config
    # Wrap the attack call with tqdm for a progress bar
        
    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.get_logits(batch)
                if isinstance(out, ImageClassifierOutput):
                    out = out.logits
                outs.append(out)
        outs = torch.cat(outs)
        prob = F.softmax(outs, dim=1)
        return prob.detach().cpu().numpy()
    


    def forward(self, images, labels) -> Tuple[torch.Tensor, int, int]:
        # Initialize wandb run
        print(f"Image Counter: {self.image_counter}", type(self.image_counter))
        self.image_counter += 1

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        print("devices", images.device, labels.device)

        input_grid = vutils.make_grid(images, nrow=int(math.sqrt(images.size(0))), normalize=True)
        # Wandb have an issue having the number of channels on the first parameter of the shape
        # we need to permute it.
        input_grid = input_grid.permute(1, 2, 0)
        wandb.log({f"Input Images_batch_{self.image_counter}": wandb.Image(input_grid.cpu().numpy(), caption="Input Batch")})

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
                    if success: 
                        self.attack_success_counter += 1
                    wandb.log({"Attack Success": self.attack_success_counter, "Convergence": convergence})
                    return success

            else:

                def func(delta):
                    return self._loss(image, label, delta)

                def callback(delta, convergence):
                    success = self._attack_success(image, label, delta)
                    if success: 
                        self.attack_success_counter += 1
                    wandb.log({"Attack Success": self.attack_success_counter, "Convergence": convergence})
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

        adv_grid = vutils.make_grid(adv_images, nrow=int(math.sqrt(images.size(0))), normalize=True)
        adv_grid = adv_grid.permute(1, 2, 0)
        print(input_grid.shape)
        
        # Optionally log the final output as an artifact, if useful
        wandb.log({"Final Adversarial Images": wandb.Image(adv_grid.cpu().numpy(), caption="Final Adversarial Images")})
    #  Get the label from batched prediction with the hisgtest probability
        batched_predictions = self.model(adv_images).logits.argmax(dim=1)
        print("Batched Predictions", batched_predictions.shape)
        print(batched_predictions.shape)
        original_predictions = self.model(images).logits.argmax(dim=1)
        successes = batched_predictions != labels
        num_sucesses = successes.sum().item()
        original_failures = original_predictions != labels



        wandb.log({f"Attack Success Rate in Attacked_Batch_{self.image_counter}": successes/len(images) })
        wandb.log({f"Attack Success Rate in Original_Batch_{self.image_counter}": original_failures/len(images) })

    


        return adv_image, num_sucesses,original_failures, batched_predictions, original_predictions
    


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
    


class PGDLogger(ta.PGD, AttackLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            return adv_images
