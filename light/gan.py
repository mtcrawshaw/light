""" Training and generation with GANs. """

import random
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def train_gan(config: Dict[str, Any]):
    """ Train GAN with settings from ``config``. """

    # Set random seed.
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Set device.
    if config["cuda"]:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print(
                "Warning: Cuda set to true, but torch.cuda.is_available() is False. "
                "Using CPU."
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Construct data loader.
    dataset = dset.ImageFolder(
        root=config["data_location"],
        transform=transforms.Compose([
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                (config["data_mean"], config["data_mean"], config["data_mean"]),
                (config["data_stdev"], config["data_stdev"], config["data_stdev"])
            ),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # TEMP (test that data loader works)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    print("done")
