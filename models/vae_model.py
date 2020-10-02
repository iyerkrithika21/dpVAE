"""vae_model.py"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAE_Basic(nn.Module):
    """Basic VAE"""

    def __init__(self, z_dims=10, input_dims=3, sampling_freq=1):
        super(VAE_Basic, self).__init__()
        self.z_dims = z_dims
        self.input_dims = input_dims
        self.sampling_freq = sampling_freq

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            View((-1, 256*1*1)),       # B, 256
            nn.Linear(256, z_dims*2),  # B, z_dims*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 256),  # B, 256
            View((-1, 256, 1, 1)),   # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_dims, 4, 2, 1),
        )

    def reparametrize_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def reparametrize(self, mu, logvar):
        z = self.reparametrize_z(mu, logvar)
        for i in range(1, self.sampling_freq):
            temp = self.reparametrize_z(mu, logvar)
            z += temp
        z = z/self.sampling_freq
        return z

    def forward(self, x):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dims]
        logvar = stats[:, self.z_dims:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z
