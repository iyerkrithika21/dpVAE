"""factor_model.py"""

from models.vae_model import *


class Discriminator(nn.Module):
    def __init__(self, z_dims):
        super(Discriminator, self).__init__()
        self.z_dims = z_dims
        self.net = nn.Sequential(
            nn.Linear(z_dims, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE1(VAE_Basic):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dims=10, input_dims=3):
        super(FactorVAE1, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            View((-1, 256*1*1)),       # B, 256
            nn.Linear(256, z_dims*2),  # B, z_dims*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 256),  # B, 256
            View((-1, 256, 1, 1)),   # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_dims, 4, 2, 1),
        )

    def forward(self, x, no_dec=False):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dims]
        logvar = stats[:, self.z_dims:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decoder(z)
            return x_recon, mu, logvar, z


class FactorVAE2(FactorVAE1):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, z_dims=10, input_dims=3):
        super(FactorVAE2, self).__init__(z_dims)

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

class FactorMNIST(FactorVAE1):
    """Encoder and Decoder architecture for 3D Faces data."""
    def __init__(self, z_dims=10, input_dims=3):
        super(FactorMNIST, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, 32, 4, 2, 1), # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 0), # B,  64, 12, 12
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), # B,  64,  6,  6
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 0), # B,  256,  4,  4
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 1, 0), # B,  256,  1, 1
            nn.ReLU(True),
            View((-1, 256*1*1)),       # B, 256
            nn.Linear(256, z_dims*2),  # B, z_dims*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 256),  # B, 256
            View((-1, 256, 1, 1)),   # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 1, 0), # B,  256,  4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 3, 1, 0), # B,  64,  6,  6
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  32, 12, 12
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 1, 0), # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_dims, 4, 2, 1), # B,  self.input_dims, 28, 28
        )

class Factor_Linear(FactorVAE1):
    """
    Linear version of factor vae for non-image datasets
    """

    def __init__(self, z_dims=10, input_dims=2):
        super(Factor_Linear, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 100),         # B, 100
            nn.LeakyReLU(True),
            nn.Linear(100, 50),                 # B, 50
            nn.LeakyReLU(True),
            nn.Linear(50, z_dims*2),             # B, self.z_dims*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 50),               # B, 50
            nn.LeakyReLU(True),
            nn.Linear(50, 100),                 # B, 100
            nn.LeakyReLU(True),
            nn.Linear(100, input_dims),         # B, input_dims
        )
