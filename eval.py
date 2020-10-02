
from trainers.trainer import get_trainer
import torch 
import numpy as np

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for images in loader:
        images = images.float()
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def main(config):
    trainer = get_trainer(config)

    # trainer.draw_reconstruction()
    # trainer.draw_generated()
    # trainer.draw_qz()
    # find the training covariance matrix
    # m, sig = online_mean_and_sd(trainer.data_loader)
    # print(m, sig)
    sig  = 0.2891
    ldr = trainer.val_loader
    probVec = np.zeros([1000, 3])
    digVec = np.zeros([1000, 28, 28])
    count = 0
    for x in ldr:
        print(count)
        for i in range(10):
            img = x[i, ...].reshape(1, 28, 28).detach().cpu().numpy()
            [px, poff, pon] = trainer.analyze_zspace(x[i, ...].reshape(1, 1, 28, 28), sig**2 * torch.eye(28*28))
            digVec[i + 10*count, ...] = img
            probVec[i + 10*count, 0] = px.detach().cpu().numpy()
            probVec[i + 10*count, 1] = poff.detach().cpu().numpy()
            probVec[i + 10*count, 2] = pon.detach().cpu().numpy()
        count +=1
    np.save('valdataMNIST.npy', digVec)
    np.save('probVecMNIST.npy', probVec)
    # print('drew qz')
    # trainer.draw_loss()
    # trainer.visualize_traverse()
    # trainer.traversal()
    # print('traversed')
    # trainer.computeGenMetrics()
    # print('gen metrics')
    # trainer.generateSamples(20000)
    # print('generated')
    # trainer.saveZspace()
    # trainer.high_metric(3000)


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    config['cont'] = True
    if len(sys.argv) > 2:
        config['ckpt_name'] = sys.argv[2]
    main(config)
