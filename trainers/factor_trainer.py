from trainers.trainer import *


class FactorTrainer(Trainer):
    """docstring for FactorTrainer."""

    def __init__(self, config):
        super(FactorTrainer, self).__init__(config)

        # model
        from models.factor_model import Discriminator
        self.D = Discriminator(self.z_dims).to(self.device)

        # train
        self.gamma = config['gamma']
        self.lr_D = config['D_lr']
        self.beta1_D = config['D_beta1']
        self.beta2_D = config['D_beta2']
        self.optim_D = optim.Adam(
          self.D.parameters(), lr=self.lr_D, betas=(self.beta1_D, self.beta2_D)
        )
        if self.implicit:
            self.train_inst = self.implicit_inst

        # saving
        if config['cont'] and self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

    def train_inst(self, x):
        x_true1, x_true2 = x
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        x_true1 = x_true1.to(self.device)
        x_recon, mu, logvar, z = self.model(x_true1)
        recon_loss = self.reconstruction_loss(x_true1, x_recon)
        kld = self.kl_divergence(mu, logvar)

        D_z = self.D(z)
        tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = recon_loss + kld + self.gamma*tc_loss

        self.optim.zero_grad()
        vae_loss.backward(retain_graph=True)
        self.optim.step()

        x_true2 = x_true2.to(self.device)
        _, _, _, z_prime = self.model(x_true2)
        z_pperm = self.permute_dims(z_prime).detach()
        D_z_pperm = self.D(z_pperm)
        D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

        self.optim_D.zero_grad()
        D_tc_loss.backward()
        self.optim_D.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['kld'] = kld.item()
        meta['loss'] = vae_loss.item()
        meta['tc_loss'] = tc_loss.item()
        meta['D_tc_loss'] = D_tc_loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, vae_loss)
            self.loop_update = self.loop_update + ' D_loss:{:.3f}'.format(D_tc_loss.item())
            # self.add_logvar(logvar)

        return x_true1, torch.sigmoid(x_recon), meta

    def implicit_inst(self, x):
        x_true1, x_true2 = x
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        x_true1 = x_true1.to(self.device)
        x_recon, mu, logvar, z, g_1, g_2 = self.model(x_true1)
        recon_loss = self.reconstruction_loss(x_true1, x_recon)
        ELS = -1*torch.sum(g_2)
        k = torch.mul(g_1, g_1).sum(dim=1)
        DKL = -0.5*torch.sum(logvar.sum(dim=1) - k)
        kld = ELS+DKL

        D_z = self.D(z)
        tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = recon_loss + kld + self.gamma*tc_loss

        self.optim.zero_grad()
        vae_loss.backward(retain_graph=True)
        self.optim.step()

        x_true2 = x_true2.to(self.device)
        _, _, _, z_prime, _, _ = self.model(x_true2)
        z_pperm = self.permute_dims(z_prime).detach()
        D_z_pperm = self.D(z_pperm)
        D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

        self.optim_D.zero_grad()
        D_tc_loss.backward()
        self.optim_D.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['kld'] = kld.item()
        meta['loss'] = vae_loss.item()
        meta['tc_loss'] = tc_loss.item()
        meta['D_tc_loss'] = D_tc_loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, vae_loss)
            self.loop_update = self.loop_update + ' D_loss:{:.3f}'.format(D_tc_loss.item())
            # self.add_logvar(logvar)

        return x_true1, torch.sigmoid(x_recon), meta

    def permute_dims(self, z):
        assert z.dim() == 2

        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)

    # model save functions

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.model.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        modelD_states = {'net':self.D.state_dict(),}
        optimD_states = {'optim':self.optim_D.state_dict(),}
        states = {'iter':self.global_iter,
                  'meta':self.meta,
                  'model_states':model_states,
                  'optim_states':optim_states,
                  'modelD_states':modelD_states,
                  'optimD_states':optimD_states}

        file_path = osp.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        try:
            file_path = osp.join(self.ckpt_dir, filename)
            if osp.isfile(file_path):
                checkpoint = torch.load(file_path)
                self.global_iter = checkpoint['iter']
                self.meta = checkpoint['meta']
                self.model.load_state_dict(checkpoint['model_states']['net'])
                self.optim.load_state_dict(checkpoint['optim_states']['optim'])
                self.D.load_state_dict(checkpoint['modelD_states']['net'])
                self.optim_D.load_state_dict(checkpoint['optimD_states']['optim'])
                print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
            else:
                print("=> no checkpoint found at '{}'".format(file_path))
        except Exception as e:
            print('model couldnt load')


def main(config):
    trainer = FactorTrainer(config)
    trainer.train()


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    main(config)
