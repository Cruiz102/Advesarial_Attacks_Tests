from setuptools import setup

setup(
    name='Adversarial_Testings',
    version='0.1.0',
    description='Your project description',
    author='Your Name',
    author_email='cruiznavarro44@gmail.com',
    packages=['hugginface_testing'],  # Replace with the correct name of your package/directory
    install_requires=[
        'scipy>=0.14.0',
        'tqdm>=4.56.1',
        'requests>=2.25.1',
        'pandas>=1.2.4',
        'numpy>=1.19.4',
        'torchattacks',
        'transformers',
        'wandb',
        'rai-toolbox',
    ],
    extras_require={
        'performance': ['accelerate','deepspeed']
    },
    # Include any other necessary setup arguments below
)
