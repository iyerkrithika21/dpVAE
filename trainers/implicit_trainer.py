
from torch.autograd import Variable
from trainers.trainer import *
from dataset import get_dataset


class ImplicitTrainer(Trainer):
    """docstring for ImplicitTrainer."""

    def __init__(self, config):
        super(ImplicitTrainer, self).__init__(config)
        # model
        self.objdef = 'B'
        if self.modef == 'beta_h':
            from models.beta_model import BetaVAE_H
            self.model = BetaVAE_H(self.z_dims, self.input_dims)
            self.loss = self.h_loss
            self.objdef = 'H'
        elif self.modef == 'beta_b':
            from models.beta_model import BetaVAE_B
            self.model = BetaVAE_B(self.z_dims, self.input_dims)
        elif self.modef == 'beta_mnist':
            from models.beta_model import BetaVAE_MNIST
            self.model = BetaVAE_MNIST(z_dims=self.z_dims, input_dims=self.input_dims)
        elif self.modef == 'beta_linear':
            from models.beta_model import BetaVAE_Linear
            self.model = BetaVAE_Linear(self.z_dims, self.input_dims)
        else:
            raise NotImplementedError(
              'unsupported model: {}'.format(self.modef)
            )

        self.model = cuda(self.model, self.use_cuda)

        # train
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.C_max = Variable(
          cuda(torch.FloatTensor([config['C_max']]), self.use_cuda)
        )
        self.C_stop_iter = config['C_stop_iter']
        self.optim = optim.Adam(
          self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        # saving
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        # data
        self.data_loader = get_dataset(config)

    def train_inst(self, x):
        x = Variable(cuda(x, self.use_cuda))
        x_recon, mu, logvar = self.model(x)
        recon_loss = self.reconstruction_loss(x, x_recon)
        kld = self.kl_divergence(mu, logvar)
        loss = self.loss(recon_loss, kld)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.global_iter%self.gather_step == 0:
            self.meta['mus'].append(mu.mean(0).data)
            self.meta['vars'].append(logvar.exp().mean(0).data)
            self.meta['recon_loss'].append(recon_loss.item())
            self.meta['kld'].append(kld.item())
            self.meta['tot_loss'].append(loss.item())

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), loss

    def loss(self, recon_loss, kld):
        C = torch.clamp(
          self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.item()
        )
        return recon_loss + self.gamma*(kld-C).abs()

    def h_loss(self, recon_loss, total_kld):
        return recon_loss + self.beta*total_kld


def main(config):
    trainer = ImplicitTrainer(config)
    trainer.train()


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    main(config)
