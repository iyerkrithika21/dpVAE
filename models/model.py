
import torch

def get_model(config):
    model = None
    if config['model'] == 'basic':
        from models.vae_model import VAE_Basic
        model = VAE_Basic(config['z_dims'], config['input_dims'])
    elif config['model'] == 'beta_h':
        from models.beta_model import BetaVAE_H
        model = BetaVAE_H(config['z_dims'], config['input_dims'])
    elif config['model'] == 'beta_b':
        from models.beta_model import BetaVAE_B
        model = BetaVAE_B(config['z_dims'], config['input_dims'])
    elif config['model'] == 'beta_mnist':
        from models.beta_model import BetaVAE_MNIST
        model = BetaVAE_MNIST(config['z_dims'], config['input_dims'])
    elif config['model'] == 'beta_svhn':
        from models.beta_model import BetaVAE_SVHN
        model = BetaVAE_SVHN(config['z_dims'], config['input_dims'])
    elif config['model'] == 'beta_celeba':
        from models.beta_model import BetaVAE_CelebA
        model = BetaVAE_CelebA(config['z_dims'], config['input_dims'])
    elif config['model'] == 'beta_linear':
        from models.beta_model import BetaVAE_Linear
        model = BetaVAE_Linear(config['z_dims'], config['input_dims'])
    elif config['model'] == 'factor1':
        from models.factor_model import FactorVAE1
        model = FactorVAE1(config['z_dims'], config['input_dims'])
    elif config['model'] == 'factor2':
        from models.factor_model import FactorVAE2
        model = FactorVAE2(config['z_dims'], config['input_dims'])
    elif config['model'] == 'factor_mnist':
        from models.factor_model import FactorMNIST
        model = FactorMNIST(config['z_dims'], config['input_dims'])
    elif config['model'] == 'factor_linear':
        from models.factor_model import Factor_Linear
        model = Factor_Linear(config['z_dims'], config['input_dims'])
    else:
        raise NotImplementedError(
          'unsupported model: {}'.format(config['model'])
        )

    if 'implicit' in config and config['implicit']:
        from models.implicit_model import Implicit
        model = Implicit(model, config)
    if torch.cuda.is_available():
        model = model.cuda()

    return model
