# dcgan.py

import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
from PIL import Image


# Define the DCGAN Generator architecture
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# Utility function: load the generator (on CPU)
def load_generator(model_path='dcgan_generator.pth', nz=100, ngf=64, nc=3, device=torch.device("cpu")):
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    state_dict = torch.load(model_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG


# Option 1: Generate an image from random noise
def generate_random(nz=100, model_path='gan/dcgan_generator.pth'):
    device = torch.device("cpu")
    netG = load_generator(model_path=model_path, nz=nz, device=device)
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = netG(noise).cpu()
    grid = vutils.make_grid(fake_image, normalize=True)
    np_image = np.transpose(grid.numpy(), (1, 2, 0))
    pil_img = Image.fromarray((np_image * 255).astype(np.uint8))
    return pil_img


# Option 2: Generate an image using an uploaded image as seed
def generate_from_image(uploaded_file, nz=100, model_path='gan/dcgan_generator.pth'):
    """
    This function uses the contents of the uploaded image file to seed the noise.
    The seed is derived by hashing the raw bytes of the file so that the output is deterministic.
    """
    device = torch.device("cpu")
    netG = load_generator(model_path=model_path, nz=nz, device=device)

    # Read uploaded file bytes
    image_bytes = uploaded_file.read()
    # Compute a seed from the image bytes (convert to a positive integer)
    seed = abs(hash(image_bytes)) % (2 ** 32)
    # Use the seed to generate a noise vector deterministically
    rng = np.random.default_rng(seed)
    noise_np = rng.standard_normal((1, nz, 1, 1)).astype(np.float32)
    noise = torch.tensor(noise_np, device=device)

    with torch.no_grad():
        fake_image = netG(noise).cpu()

    grid = vutils.make_grid(fake_image, normalize=True)
    np_image = np.transpose(grid.numpy(), (1, 2, 0))
    pil_img = Image.fromarray((np_image * 255).astype(np.uint8))
    return pil_img
