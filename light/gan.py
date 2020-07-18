""" Training and generation with GANs. """

import random
from math import sqrt, ceil
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


NUM_FIXED_LATENTS = 64
LABELS = {"fake": 0, "real": 1}


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
    data_loader = torch.utils.data.DataLoader(
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
        discriminator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999),
    )
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999),
    )
    loss_fn = nn.BCELoss()

    # Construct fixed latent vectors to track progress of generator through training.
    fixed_latents = torch.randn(
        NUM_FIXED_LATENTS, config["latent_size"], 1, 1, device=device
    )

    # Training loop.
    training_steps = 0
    generated_imgs = []
    for epoch in range(config["num_epochs"]):
        for batch_index, data in enumerate(data_loader):

            # Update discriminator network. We perform backward passes with one batch of
            # all real images, then one batch of all fake images, then add the
            # gradients. Notice that we call .detach() on ``fake_batch`` before feeding
            # it into the discriminator, so that we are treating the output of the
            # generator as constant during this step.
            discriminator.zero_grad()
            real_batch = data[0].to(device)
            batch_size = real_batch.size(0)
            label = torch.full((batch_size,), LABELS["real"], device=device)
            real_output = discriminator(real_batch).view(-1)
            real_discriminator_loss = loss_fn(real_output, label)
            real_discriminator_loss.backward()
            D_x = real_output.mean().item()

            latent_vectors = torch.randn(
                batch_size, config["latent_size"], 1, 1, device=device
            )
            fake_batch = generator(latent_vectors)
            label.fill_(LABELS["fake"])
            fake_output = discriminator(fake_batch.detach()).view(-1)
            fake_discriminator_loss = loss_fn(fake_output, label)
            fake_discriminator_loss.backward()
            discriminator_loss = real_discriminator_loss + fake_discriminator_loss
            discriminator_optimizer.step()
            D_G_z1 = fake_output.mean().item()

            # Update generator network. Notice that we no longer call .detach() on
            # ``fake_batch``, so that we can train the generator parameters. Here we
            # compute the loss as the binary cross entropy between the discriminator
            # output on a fake batch against the real label, i.e. we want to train the
            # generator so that the discriminator labels fake images as real.
            generator.zero_grad()
            label.fill_(LABELS["real"])
            discriminator_output = discriminator(fake_batch).view(-1)
            generator_loss = loss_fn(discriminator_output, label)
            generator_loss.backward()
            generator_optimizer.step()
            D_G_z2 = discriminator_output.mean().item()

            # Print training metrics.
            if batch_index % config["print_freq"] == 0:
                print(
                    "Epoch: %d / %d, Batch: %d / %d"
                    "\t D loss: %.5f, G loss: %.5f"
                    "\t D(x): %.5f, D(G(z)): %.5f / %.5f"
                    % (
                        epoch,
                        config["num_epochs"],
                        batch_index,
                        len(data_loader),
                        discriminator_loss.item(),
                        generator_loss.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Generate samples from fixed latent vectors to check progress of generator.
            if training_steps % config["generate_freq"] == 0 or (
                epoch == config["num_epochs"] and batch_index == len(data_loader) - 1
            ):
                with torch.no_grad():
                    generated = generator(fixed_latents).detach().cpu()
                generated_imgs.append(
                    vutils.make_grid(generated, padding=2, normalize=True)
                )

            training_steps += 1

    # Animate generated images. This is temporary?
    fig_len = ceil(sqrt(NUM_FIXED_LATENTS))
    fig = plt.figure(figsize=(fig_len, fig_len))
    plt.axis("off")
    imgs = [[plt.imshow(np.transpose(img, (1, 2, 0)), animated=True)] for img in generated_imgs]
    ani = animation.ArtistAnimation(fig, imgs, interval=1000, repeat_delay=1000, blit=True)
    plt.show()


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
