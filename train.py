
from trainers.trainer import get_trainer

def main(config):
    trainer = get_trainer(config)
    trainer.train()


if __name__ == '__main__':
    import json, sys
    config = json.load(open(sys.argv[1]))
    config['cont'] = False
    if len(sys.argv) > 2:
        config['cont'] = True
        config['ckpt_name'] = sys.argv[2]
    main(config)
