from tqdm import tqdm
import torchattacks as ta
import wandb
import torch
import numpy
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


# TODO: Implement the forward loop
    def forward(self, images, labels):
        pass