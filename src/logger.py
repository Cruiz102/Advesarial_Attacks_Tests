from tqdm import tqdm
import torchattacks as ta
import wandb
import torch
import torch.nn as nn
import torch.optim
import numpy
from  transformers.modeling_outputs import ImageClassifierOutput
#  The goal in here will be to implement the loops of the attacks
#  with tqdm and wandb.

class OnePixelLogger(ta.OnePixel):
    def __init__(self, *args, **kwargs):
        # Forward arguments to the Parent class
        super().__init__(*args, **kwargs)
    # Wrap the attack call with tqdm for a progress bar

    def forward(self, images, labels):
        # Initialize wandb run
        wandb.init(project="your_project_name", entity="your_wandb_entity")

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

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
                    wandb.log({"Attack Success": success, "Convergence": convergence})
                    return success

            else:

                def func(delta):
                    return self._loss(image, label, delta)

                def callback(delta, convergence):
                    success = self._attack_success(image, label, delta)
                    wandb.log({"Attack Success": success, "Convergence": convergence})
                    return success

            delta = ta.differential_evolution(
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
        
        # Optionally log the final output as an artifact, if useful
        # wandb.log({"Final Adversarial Images": wandb.Image(adv_images)})
        
        # Finish the wandb run
        wandb.finish()

        return adv_images
    


class CWLogger(ta.CW):
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
    


    class PGDLogger(ta.PGD):
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
