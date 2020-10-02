from trainers.trainer import *
from trainers.loss_helpers import _get_log_pz_qz_prodzi_qzCx, _get_log_pz_qz_prodzi_qzCx_fip

class BetatcTrainer(Trainer):
    """
    docstring for BetatcTrainer.
    See the following github repo for details on loss function.
    https://github.com/YannDubs/disentangling-vae
    """

    def __init__(self, config):
        super(BetatcTrainer, self).__init__(config)

        # train
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']

    def train_inst(self, x):
        x = x.to(self.device)
        x_recon, mu, logvar, z = self.model(x)
        recon_loss = self.reconstruction_loss(x, x_recon)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
          z, (mu, logvar), len(self.data_loader.dataset), is_mss=True
        )
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # kld is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        kld = (log_prod_qzi - log_pz).mean()

        # total loss
        loss = recon_loss + (self.alpha * mi_loss + self.beta * tc_loss + self.gamma * kld)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['mi_loss'] = mi_loss.item()
        meta['tc_loss'] = tc_loss.item()
        meta['kld'] = kld.item()
        meta['loss'] = loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            # self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), meta

    def sample_qz_betatc(self):
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
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx_fip(
          z, g_1, (mu, logvar), len(self.data_loader.dataset), is_mss=True
        )

        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()

        DKL = (log_prod_qzi - log_pz).mean()
        # loss = self.loss(recon_loss, kld)
        # qzs_sample = self.sample_qz_betatc()
        # g_11, g_21 = self.model.g(qzs_sample)
        # ELS = -1*torch.sum(g_21)
        kld = DKL
        loss = recon_loss + (self.alpha * mi_loss + self.beta * tc_loss + self.gamma * kld)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['mi_loss'] = mi_loss.item()
        meta['tc_loss'] = tc_loss.item()
        meta['kld'] = kld.item()
        meta['loss'] = loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            # self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), meta


def main(config):
    trainer = BetatcTrainer(config)
    trainer.train()


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    main(config)
