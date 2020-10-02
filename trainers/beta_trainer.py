from trainers.trainer import *

class BetaTrainer(Trainer):
    """docstring for BetaTrainer."""

    def __init__(self, config):
        super(BetaTrainer, self).__init__(config)

        # train
        self.objdef = config['objective']
        if self.objdef == 'H':
            self.beta = config['beta']
            self.loss = self.h_loss
        else:
            self.gamma = config['gamma']
            self.C_max = torch.FloatTensor([config['C_max']]).to(self.device)
            self.C_stop_iter = config['C_stop_iter']

    def loss(self, recon_loss, kld):
        C = torch.clamp(
          self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.item()
        )
        return recon_loss + self.gamma*(kld-C).abs()

    def h_loss(self, recon_loss, kld):
        return recon_loss + self.beta*kld


def main(config):
    trainer = BetaTrainer(config)
    trainer.train()


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    main(config)
