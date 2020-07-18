""" Training and generation with GANs. """

import random
from typing import Dict, Any

import torch
import torch.nn as nn
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
        transform=transforms.Compose(
            [
                transforms.Resize(config["image_size"]),
                transforms.CenterCrop(config["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (config["data_mean"], config["data_mean"], config["data_mean"]),
                    (config["data_stdev"], config["data_stdev"], config["data_stdev"]),
                ),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # Construct generator and discriminator networks.
    generator = get_generator(config, device)
    discriminator = get_discriminator(config, device)

    # Construct optimizers and loss function.
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config["learning_rate"]
    )
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config["learning_rate"]
    )
    loss_fn = nn.BCELoss()


def get_generator(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """ Return generator network corresponding to options in ``config``. """

    # Construct layers of generator network.
    generator_layers = [
        nn.ConvTranspose2d(
            in_channels=config["latent_size"],
            out_channels=config["generator_feature_maps"] * 8,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=config["generator_feature_maps"] * 8),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            in_channels=config["generator_feature_maps"] * 8,
            out_channels=config["generator_feature_maps"] * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=config["generator_feature_maps"] * 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            in_channels=config["generator_feature_maps"] * 4,
            out_channels=config["generator_feature_maps"] * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=config["generator_feature_maps"] * 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            in_channels=config["generator_feature_maps"] * 2,
            out_channels=config["generator_feature_maps"],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=config["generator_feature_maps"]),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            in_channels=config["generator_feature_maps"],
            out_channels=config["num_data_channels"],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.Tanh(),
    ]

    # Compile layers, send to device, initialize weights, and return module.
    generator = nn.Sequential(*generator_layers)
    generator = generator.to(device)
    generator.apply(weight_init)

    return generator


def get_discriminator(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """ Return discriminator network corresponding to options in ``config``. """

    # Construct layers of discriminator network.
    # START HERE
    discriminator_layers = [
        nn.Conv2d(
            in_channels=config["num_data_channels"],
            out_channels=config["discriminator_feature_maps"],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=config["discriminator_feature_maps"],
            out_channels=config["discriminator_feature_maps"] * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(config["discriminator_feature_maps"] * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=config["discriminator_feature_maps"] * 2,
            out_channels=config["discriminator_feature_maps"] * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(config["discriminator_feature_maps"] * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=config["discriminator_feature_maps"] * 4,
            out_channels=config["discriminator_feature_maps"] * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(config["discriminator_feature_maps"] * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=config["discriminator_feature_maps"] * 8,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.Sigmoid(),
    ]

    # Compile layers, send to device, initialize weights, and return module.
    discriminator = nn.Sequential(*discriminator_layers)
    discriminator = discriminator.to(device)
    discriminator.apply(weight_init)

    return discriminator


def weight_init(m: torch.nn.Module) -> None:
    """ Weight initialization function for networks. """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
