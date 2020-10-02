from trainers.trainer import *

class InfoTrainer(Trainer):
    """docstring for InfoTrainer."""

    def __init__(self, config):
        super(InfoTrainer, self).__init__(config)

        # train
        self.alpha = config['alpha']
        self.lambd = config['lambda']

    def train_inst(self, x):
        x = x.to(self.device)
        x_recon, mu, logvar, z = self.model(x)
        recon_loss = self.reconstruction_loss(x, x_recon)
        kld = self.kl_divergence(mu, logvar)
        mmd = self.compute_mmd(z)
        loss = recon_loss + (1-self.alpha)*kld + (self.alpha+self.lambd-1)*mmd

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['kld'] = kld.item()
        meta['mmd'] = mmd.item()
        meta['loss'] = loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            # self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), meta

    def sample_qz_info(self):
        N = len(self.data_loader.dataset)
        ks = np.random.random_integers(0, N-1, self.batch_size)
        imps = torch.cat(
          [self.data_loader.dataset[k].unsqueeze(0) for k in ks]
        )
        # imps = self.val_loader.dataset[ks]
        qzs = self.model.encoder(imps.to(self.device))
        mus = qzs[:, :self.z_dims]
        logvars = qzs[:, self.z_dims:]
        z = self.model.reparametrize(mus, logvars)
        return z.detach()

    def implicit_inst(self, x):
        x = x.to(self.device)
        x_recon, mu, logvar, z, g_1, g_2 = self.model(x)
        recon_loss = self.reconstruction_loss(x, x_recon)
        # DKL loss
        ELS = -1*torch.sum(g_2)
        k = torch.mul(g_1, g_1).sum(dim=1)
        DKL = -0.5*torch.sum(logvar.sum(dim=1) - k)
        kld = ELS+DKL
        MD = self.compute_mmd(z)
        # qzs_sample = self.sample_qz_info()
        # g_11, g_21 = self.model.g(qzs_sample)
        # ELS2 = -1*torch.sum(g_21)
        mmd = MD
        loss = recon_loss + (1-self.alpha)*kld + (self.alpha+self.lambd-1)*mmd

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['kld'] = kld.item()
        meta['mmd'] = mmd.item()
        meta['loss'] = loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            # self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), meta

    def compute_mmd(self, z):
        true_z = torch.randn(self.batch_size, self.z_dims, requires_grad = False).to(self.device)
        if self.implicit:
            true_z = self.model.g_inv(true_z)
        true_kernel = self.compute_kernel(true_z, true_z)
        learned_kernel = self.compute_kernel(z, z)
        comb_kernel = self.compute_kernel(true_z, z)
        mmd = true_kernel.mean() + learned_kernel.mean() - 2*comb_kernel.mean()
        return mmd

    def compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)


def main(config):
    trainer = InfoTrainer(config)
    trainer.train()


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    main(config)
