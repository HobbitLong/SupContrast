"""Wandb hyperparameter sweep for SupCon training. Trying to minimize the loss."""
import pprint
import wandb
import time

from main_supcon import parse_option, set_loader, set_model, train
from util import adjust_learning_rate
from util import set_optimizer


metric = {"name": "loss", "goal": "minimize"}

parameters_dict = {
    "batch_size": {"values": [16, 32, 64, 128, 256]},
    "learning_rate": {"values": [0.001, 0.01, 0.1, 0.5]},
    # "rand_augment": {"values": [True, False]},
}

parameters_dict.update(
    {
        "epochs": {"value": 100},
        "lr_decay_epochs": {"value": "70, 80, 90"},
        "dataset": {"value": "path"},
        "data_folder": {"value": "./data/input/15objects_lab/samples/train"},
    }
)

N_SWEEPS = 10
SWEEP_CONFIG = {"method": "random"}
SWEEP_CONFIG["metric"] = metric
SWEEP_CONFIG["parameters"] = parameters_dict

# training routine
def tune_train(config=None):
    with wandb.init(config=config):
        opt = parse_option()

        config = wandb.config
        # Add config as a new attributes to opt
        for k, v in config.items():
            setattr(opt, k, v)

        # build data loader
        train_loader = set_loader(opt)

        # build model and criterion
        model, criterion = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt, model)

        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))


def main():
    pprint.pprint(SWEEP_CONFIG)

    # wandb sweep
    wandb.login()
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="supcon_sweep")

    wandb.agent(sweep_id, tune_train, count=N_SWEEPS)


if __name__ == "__main__":
    main()
