import os, sys, json
import os.path as osp
from trainers.trainer import get_trainer

# idx = 0
# if len(sys.argv) > 1:
#     idx = int(sys.argv[1])
# condir = 'config'
# configs = [con for con in os.listdir(condir) if '.json' in con]
# configs = configs[idx:]

# idx = 0
# if len(sys.argv) > 1:
#     idx = int(sys.argv[1])
# condir = 'config/factor'
# configs = [osp.join(condir, con) for con in os.listdir(condir) if 'celeba.json' in con]
# # condir = 'config/betatc'
# # configs = configs + [osp.join(condir, con) for con in os.listdir(condir) if 'celeba.json' in con]
# configs = configs[idx:]

# for ci, conf in enumerate(configs):
#     try:
#         print(conf)
#         config = json.load(open(conf))
#         config['cont'] = False
#         trainer = get_trainer(config)
#         trainer.train()
#     except Exception as e:
#         print(e)
#         print('failed on', conf, ',idx:', ci+idx)

# conf = 'config/betatc/ip_celeba.json'
conf = "config/vanilla/mnist.json"
print(conf)
config = json.load(open(conf))
config['cont'] = False
trainer = get_trainer(config)
trainer.train()

# # get first look (only 10000 iterations)
# for ci, conf in enumerate(configs):
#     try:
#         print(conf)
#         config = osp.join(condir, conf)
#         config = json.load(open(config))
#         config['cont'] = False
#         config['max_iter'] = config['display_step']+1
#         trainer = get_trainer(config)
#         trainer.train()
#     except Exception as e:
#         print(e)
#         print('failed on', ci+idx)


# train beta learner on different beta values
# betas = [val for val in range(2, 20, 2)]
# for ci, conf in enumerate(configs):
#     try:
#         config = osp.join(condir, conf)
#         config = json.load(open(config))
#         config['cont'] = False
#         if config['type'] != 'beta':
#             continue
#         for beta in betas:
#             print('type: {}, beta: {}'.format(conf, beta))
#             config['beta'] = beta
#             config['max_iter'] = config['display_step']*5 + 1
#             config['output_dir'] = config['output_dir'] + '_{}'.format(beta)
#             config['ckpt_dir'] = config['ckpt_dir'] + '_{}'.format(beta)
#             trainer = get_trainer(config)
#             trainer.train()
#     except Exception as e:
#         print(e)
#         print('failed on', ci+idx)
