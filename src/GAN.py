import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd as autograd
from src.config import (
    device,
    GAN_LATENT_DIM,
    GAN_EPOCHS,
    GAN_BATCH_SIZE,
    GAN_LR_GENERATOR,
    GAN_LR_DISCRIMINATOR,
    GAN_LAMBDA_GP,
    GAN_N_CRITIC,
    GAN_BETA1,
    GAN_BETA2,
    GAN_TARGET_CLASSES,
    CLASS_NAMES
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
img_shape = (1, 52, 52)

# -------------------------------------------------------------------
# Latent batch helper
# -------------------------------------------------------------------
def get_gaussian_latent_batch(device, batch_size=256):
    return torch.randn((batch_size, GAN_LATENT_DIM), device=device)


# -------------------------------------------------------------------
# Generator
# -------------------------------------------------------------------
class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.init_size = img_shape[1] // 4

        self.fc = nn.Sequential(
            nn.Linear(GAN_LATENT_DIM, 128 * self.init_size ** 2),
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            # Block 1: 13x13 -> 26x26
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 2: 26x26 -> 52x52
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Block 3: sharpen
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.fc(z)
        img = img.view(img.shape[0], 128, self.init_size, self.init_size)
        return self.model(img)


# -------------------------------------------------------------------
# Discriminator
# -------------------------------------------------------------------
class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.init_size = img_shape[1] // 4

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([32, img_shape[1] // 2, img_shape[2] // 2]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([64, img_shape[1] // 4, img_shape[2] // 4]),
            nn.Flatten(),
            nn.Linear(64 * (self.init_size ** 2), 1),
        )

    def forward(self, img):
        return self.model(img)


# -------------------------------------------------------------------
# WGAN-GP Training Class
# -------------------------------------------------------------------
class WGANGPModel:
    def __init__(self, data_loader, num_epochs=GAN_EPOCHS,
                 batch_size=GAN_BATCH_SIZE, lambda_gp=GAN_LAMBDA_GP,
                 n_critic=GAN_N_CRITIC, name="wgan"):
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.name = name
        self.device = device

        self.d_net = DiscriminatorNet().train().to(device)
        self.g_net = GeneratorNet().train().to(device)

        self.d_opt = Adam(self.d_net.parameters(), lr=GAN_LR_DISCRIMINATOR, betas=(GAN_BETA1, GAN_BETA2))
        self.g_opt = Adam(self.g_net.parameters(), lr=GAN_LR_GENERATOR, betas=(GAN_BETA1, GAN_BETA2))

        self.d_losses = []
        self.g_losses = []

    def _compute_gradient_penalty(self, real_images, fake_images):
        alpha = Tensor(np.random.random((real_images.size(0), 1, 1, 1)))
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        d_interpolates = self.d_net(interpolates)
        fake = Tensor(real_images.size(0), 1).fill_(1.0).requires_grad_(False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0

            for batch_idx, (images, _) in enumerate(self.data_loader):
                real_images = images.type(Tensor)

                for _ in range(self.n_critic):
                    self.d_opt.zero_grad()
                    z = get_gaussian_latent_batch(batch_size=images.shape[0], device=self.device)
                    fake_images = self.g_net(z)
                    real_validity = self.d_net(real_images)
                    fake_validity = self.d_net(fake_images.detach())
                    gradient_penalty = self._compute_gradient_penalty(
                        real_images, fake_images.detach()
                    )
                    d_loss = (
                        -torch.mean(real_validity)
                        + torch.mean(fake_validity)
                        + self.lambda_gp * gradient_penalty
                    )
                    d_loss.backward()
                    self.d_opt.step()

                self.g_opt.zero_grad()
                z = get_gaussian_latent_batch(batch_size=images.shape[0], device=self.device)
                fake_images = self.g_net(z)
                fake_validity = self.d_net(fake_images)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                self.g_opt.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                num_batches += 1

            self.d_losses.append(epoch_d_loss / num_batches)
            self.g_losses.append(epoch_g_loss / num_batches)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"D Loss: {epoch_d_loss/num_batches:.4f} "
                  f"G Loss: {epoch_g_loss/num_batches:.4f}")

    def generate(self, num_images):
        self.g_net.eval()
        with torch.no_grad():
            z = get_gaussian_latent_batch(batch_size=num_images, device=self.device)
            generated = self.g_net(z).cpu()
            generated = 0.5 * generated + 0.5
        return generated


# -------------------------------------------------------------------
# Per class dataloaders for minority classes
# -------------------------------------------------------------------
def getClassLoaders(images, labels):
    # Rescale to [-1, 1] for GAN training
    gan_images = (images - 0.5) / 0.5
    gan_image_tensor = torch.tensor(gan_images).float()
    gan_label_indices = torch.tensor(labels.argmax(axis=1)).long()

    class_loaders = {}
    for class_idx in GAN_TARGET_CLASSES:
        mask = gan_label_indices == class_idx
        class_images = gan_image_tensor[mask]
        class_labels = gan_label_indices[mask]

        dataset = TensorDataset(class_images, class_labels)
        loader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True)
        class_loaders[class_idx] = loader
        print(f"Class {class_idx+1}: {len(dataset)} samples")

    return class_loaders


# -------------------------------------------------------------------
# Train one GAN per minority class
# -------------------------------------------------------------------
def trainGANs(class_loaders):
    trained_gans = {}

    for class_idx in GAN_TARGET_CLASSES:
        print(f"\n--- Training GAN for Class {class_idx+1} ---")
        gan_model = WGANGPModel(
            data_loader=class_loaders[class_idx],
            name=f"class_{class_idx+1}"
        )
        gan_model.train()
        trained_gans[class_idx] = gan_model
        print(f"Class {class_idx+1} GAN training complete")

    return trained_gans


# -------------------------------------------------------------------
# Visualize generated images
# -------------------------------------------------------------------
def visualizeGenerated(trained_gans):
    fig, axes = plt.subplots(len(GAN_TARGET_CLASSES), 10, figsize=(20, 6))

    for row, class_idx in enumerate(GAN_TARGET_CLASSES):
        generated = trained_gans[class_idx].generate(10)
        for col in range(10):
            axes[row, col].imshow(generated[col].squeeze(), cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(CLASS_NAMES[class_idx], fontsize=12)

    plt.tight_layout()
    plt.show()