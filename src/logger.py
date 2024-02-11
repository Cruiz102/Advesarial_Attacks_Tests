from tqdm import tqdm
import torchattacks

def tqdm_decorator(attack_func):
    def wrapper(attack_instance, *args, **kwargs):
        # Wrap the attack call with tqdm for a progress bar
        for _ in tqdm(range(attack_instance.steps), desc="Running PGD Attack"):
            adv_images = attack_func(attack_instance, *args, **kwargs)
        return adv_images
    return wrapper

# Assuming PGD is a class derived from torchattacks.Attack
class PGD(torchattacks.PGD):
    @tqdm_decorator
    def __call__(self, *args, **kwargs):
        # Call the original __call__ method of the torchattacks.PGD class
        return super().__call__(*args, **kwargs)

# Now, when you create an instance of your custom PGD class and call it,
# it will display a tqdm progress bar for the number of steps specified.
model1 = ...  # Your model definition here
pgd_attack = PGD(model1, steps=10000)

# Prepare your inputs and labels
inputs = ...  # Your input tensor here
labels = ...  # Your labels tensor here

# Run the attack with the progress bar
adv_inputs = pgd_attack(inputs, labels)