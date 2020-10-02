import numpy as np
import torch
from torch import nn

from models.vae_model import VAE_Basic

class Implicit(VAE_Basic):
    """docstring for Implicit."""

    def __init__(self, prior_model, config):
        super(Implicit, self).__init__(prior_model.z_dims, prior_model.input_dims)
        self.encoder = prior_model.encoder
        self.decoder = prior_model.decoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # mask = torch.from_numpy(np.array([[1, 0], [0, 1]] * 3).astype(np.float32))
        self.z_dims = config['z_dims']
        chk = config["masks"]
        if chk == "half":
            dpt = int(config["mask_percent"]*0.01*self.z_dims)
            a1 = np.zeros(self.z_dims)
            a1[:dpt] = 1
            a2 = np.zeros(self.z_dims)
            a2[dpt:] = 1
            mask = torch.from_numpy(np.array([a1, a2]*3).astype(np.float32))
        else:
            mask  = torch.from_numpy(np.array(config['masks']).astype(np.float32))
        mask = mask.to(self.device)
        self.inMasks = mask


        self.sampling_freq = config['sampling_freq']

        nets = lambda: nn.Sequential(nn.Linear(self.z_dims, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, self.z_dims), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(self.z_dims, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, self.z_dims), nn.Tanh())

        # to the dimension of the hidden unit
        self.mask = nn.Parameter(self.inMasks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(self.inMasks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(self.inMasks))])

    # the forward function going from z_0 to z, z = g_inv(z0)
    def g_inv(self, z0):
        z0 = z0.float()
        z0_ = z0.new_zeros(z0.shape[0]).to(self.device)
        for i in range(len(self.t)):
            z0_ = z0 * self.mask[i]
            s = self.s[i](z0_)*(1 - self.mask[i])
            t = self.t[i](z0_)*(1 - self.mask[i])
            z0 = z0_ + (1 - self.mask[i]) * (z0 * torch.exp(s) + t)
        return z0

    # the backward function going from z to z_0, z0 = g(z)
    def g(self, z):
        z = z.float()
        log_det_J = z.new_zeros(z.shape[0]).to(self.device)
        # import pdb; pdb.set_trace()
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def reparametrize(self, mu, logvar):
        z = super().reparametrize(mu, logvar)
        for i in range(1, self.sampling_freq):
            temp = super().reparametrize(mu, logvar)
            z += temp
        z = z/self.sampling_freq
        return z

    def forward(self, x):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dims]
        logvar = stats[:, self.z_dims:]
        z = self.reparametrize(mu, logvar)
        g_1, g_2 = self.g(z)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z, g_1, g_2
