"""beta_model.py"""

from models.vae_model import *


class BetaVAE_H(VAE_Basic):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dims=10, input_dims=3):
        super(BetaVAE_H, self).__init__(z_dims, input_dims)
        self.z_dims = z_dims
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dims*2),             # B, z_dims*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_dims, 4, 2, 1),  # B, nc, 64, 64
        )


class BetaVAE_B(VAE_Basic):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dims=10, input_dims=1):
        super(BetaVAE_B, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dims*2),             # B, z_dims*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_dims, 4, 2, 1), # B,  nc, 64, 64
        )


class BetaVAE_Linear(VAE_Basic):
    """
    Linear Model similar to that proposed in understanding
    beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).
    """

    def __init__(self, z_dims=10, input_dims=3):
        super(BetaVAE_Linear, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 100),         # B, 100
            nn.LeakyReLU(True),
            nn.Linear(100, 50),                 # B, 50
            nn.LeakyReLU(True),
            nn.Linear(50, z_dims*2),             # B, z_dims*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 50),               # B, 50
            nn.LeakyReLU(True),
            nn.Linear(50, 100),                 # B, 100
            nn.LeakyReLU(True),
            nn.Linear(100, input_dims),         # B, input_dims
        )


class BetaVAE_MNIST(VAE_Basic):
    """
    CNN Model similar to that proposed in understanding
    beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).
    Assumes image size is 28x28.
    """

    def __init__(self, z_dims=10, input_dims=3):
        super(BetaVAE_MNIST, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_dims, 32, 4, 2, 1),          # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 0),          # B,  32, 12, 12
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  6,  6
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 0),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dims*2),             # B, z_dims*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 3, 1, 0), # B,  32,  6,  6
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 12, 12
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, 1, 0), # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.input_dims, 4, 2, 1), # B,  input_dims, 28, 28
        )


class BetaVAE_SVHN(VAE_Basic):
    """
    CNN Model similar to that proposed in understanding
    beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).
    Assumes image size is 32x32.
    """

    def __init__(self, z_dims=10, input_dims=3):
        super(BetaVAE_SVHN, self).__init__(z_dims, input_dims)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_dims, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 3, 1, 0),          # B,  32,  6,  6
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  6,  6
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 3, 1, 0),          # B,  32,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, z_dims*2),         # B, z_dims*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dims, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4

            nn.ConvTranspose2d(32, 32, 3, 1, 0), # B,  32,  4,  4
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  6,  6
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 3, 1, 0), # B,  32, 8, 8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  8, 8
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),          # B,  32,  16, 16
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 4, 2, 1),     # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, self.input_dims, 3, 1, 1), # B,  input_dims, 32, 32
        )


class BetaVAE_CelebA(VAE_Basic):

    def __init__(self, z_dims=10, input_dims=3):
        super(BetaVAE_CelebA, self).__init__(z_dims, input_dims)

        fil_sz = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, fil_sz, 4, 2, 1),  # B, fil_sz, 32, 32
            nn.BatchNorm2d(fil_sz),
            nn.LeakyReLU(True),

            nn.Conv2d(fil_sz, fil_sz*2, 4, 2, 1),    # B, fil_sz*2, 16, 16
            nn.BatchNorm2d(fil_sz*2),
            nn.LeakyReLU(True),

            nn.Conv2d(fil_sz*2, fil_sz*4, 4, 2, 1),  # B, fil_sz*4, 8, 8
            nn.BatchNorm2d(fil_sz*4),
            nn.LeakyReLU(True),

            nn.Conv2d(fil_sz*4, fil_sz*8, 4, 2, 1),  # B, fil_sz*8, 4, 4
            nn.BatchNorm2d(fil_sz*8),
            nn.LeakyReLU(True),

            nn.Conv2d(fil_sz*8, fil_sz*8, 4, 2, 1),  # B, fil_sz*8, 2, 2
            nn.BatchNorm2d(fil_sz*8),
            nn.LeakyReLU(True),

            View((-1, fil_sz*8*2*2)),                # B, fil_sz*8*2*2
            nn.Linear(fil_sz*8*2*2, z_dims*2),       # B, z_dims*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dims, fil_sz*8*2*2),     # B, fil_sz*8*2*2
            nn.LeakyReLU(True),
            View((-1, fil_sz*8, 2, 2)),          # B, fil_sz*8,  2,  2

            nn.UpsamplingNearest2d(scale_factor=2), # B, fil_sz*8,  4,  4
            nn.ReplicationPad2d(1),
            nn.Conv2d(fil_sz*8, fil_sz*8, 3, 1),    # B, fil_sz*8,  4,  4
            nn.BatchNorm2d(fil_sz*8, 1.e-3),
            nn.LeakyReLU(True),

            nn.UpsamplingNearest2d(scale_factor=2), # B, fil_sz*8,  8,  8
            nn.ReplicationPad2d(1),
            nn.Conv2d(fil_sz*8, fil_sz*4, 3, 1),    # B, fil_sz*4,  8,  8
            nn.BatchNorm2d(fil_sz*4, 1.e-3),
            nn.LeakyReLU(True),

            nn.UpsamplingNearest2d(scale_factor=2), # B, fil_sz*4,  16,  16
            nn.ReplicationPad2d(1),
            nn.Conv2d(fil_sz*4, fil_sz*2, 3, 1),    # B, fil_sz*2,  16,  16
            nn.BatchNorm2d(fil_sz*2, 1.e-3),
            nn.LeakyReLU(True),

            nn.UpsamplingNearest2d(scale_factor=2), # B, fil_sz*2,  32,  32
            nn.ReplicationPad2d(1),
            nn.Conv2d(fil_sz*2, fil_sz, 3, 1),      # B, fil_sz,  32,  32
            nn.BatchNorm2d(fil_sz, 1.e-3),
            nn.LeakyReLU(True),

            nn.UpsamplingNearest2d(scale_factor=2), # B, fil_sz,  64,  64
            nn.ReplicationPad2d(1),
            nn.Conv2d(fil_sz, input_dims, 3, 1),    # B, fil_sz,  64,  64
        )


if __name__ == '__main__':
    pass
