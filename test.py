import argparse
import torch
import numpy as np
import torch.optim as otpm
from torch.utils.data import DataLoader
from utils import read_json
from data_loader import MNISTDataset
from data_loader import transforms
from models import CNNModel
from trainer import ClassificationTrainer
import losses as loss_md

# Setting these parameters for re-producing the result
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    # Create test dataloader
    test_ds_cfg = config['test_data_loader']['args']
    test_dataset = MNISTDataset(test_ds_cfg['root'], 
                            test_ds_cfg['csv_file'], 
                            False, transforms=transforms)
    test_loader = DataLoader(test_dataset, test_ds_cfg['batch_size'], 
                            test_ds_cfg['shuffle'], 
                            num_workers=test_ds_cfg['num_workers'])

    # Create classification model
    model = CNNModel()

    # Create criterion (loss function)
    criterion = getattr(loss_md, config['loss'])

    # Create metrics for evaluation
    metrics = [getattr(loss_md, x) for x in config['metrics']]

    # Create optimizer
    optimizer = getattr(otpm, config['optimizer']['name'])(model.parameters(),
                        **config['optimizer']['args'])

    # Create learning rate scheduler
    lr_scheduler = getattr(otpm.lr_scheduler, config['lr_scheduler']['name'])\
                    (optimizer, **config['lr_scheduler']['args'])

    # Create train procedure: classification trainer
    csf_trainer = ClassificationTrainer(config, model, criterion, metrics, 
                                        optimizer, lr_scheduler)

    csf_trainer.test(test_loader)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='MNIST - Digit Recognition')
    args_parser.add_argument('-c', '--config', default=None, type=str, help='Path of config file')
    args_parser.add_argument('-d', '--device', default=None, type=str, help='Indices of GPUs')

    args = args_parser.parse_args()

    config = read_json(args.config)
    main(config)


