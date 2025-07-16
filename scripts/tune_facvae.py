import numpy as np
import os, shutil
import torch
import wandb
import yaml

from facvae.utils import (configsdir, datadir, make_experiment_name, ScatCovDataset, logsdir)
from scripts.facvae_sweeper import FactorialVAETrainer

# Paths to raw Mars waveforms and the scattering covariance thereof.
GALFA_PATH = datadir('galfa_hi')
GALFA_SCAT_COV_PATH = datadir(os.path.join(GALFA_PATH, 'scat_covs_h5'))

# GMVAE training default hyperparameters.
GALFA_CONFIG_FILE = 'facvae_sweep.yaml'

def sweep():
    # Set up wandb for tracking training metrics
    wandb.init(project='scattering-vae-sweep', dir=datadir('logs_sweep'))
    args = wandb.config
    args.scales = ["4096"]
    args.window_size = 4096

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load data from the Mars dataset
    dataset = ScatCovDataset(os.path.join(GALFA_SCAT_COV_PATH,
                                                args.h5_filename),
                                    0.90,
                                    scatcov_datasets=args.scales,
                                    load_to_memory=args.load_to_memory,
                                    normalize_data=args.normalize)

    # Create data loaders for train, validation and test datasets

    if len(dataset.train_idx) < args.batchsize:
        args.batchsize = len(dataset.train_idx)

    train_loader = torch.utils.data.DataLoader(dataset.train_idx,
                                            batch_size=args.batchsize,
                                            shuffle=True,
                                            drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset.val_idx,
                                            batch_size=args.batchsize,
                                            shuffle=True,
                                            drop_last=False)

    # Initialize facvae trainer with the input arguments, dataset, and device
    facvae_trainer = FactorialVAETrainer(args, dataset, device)

    facvae_trainer.train(args, train_loader, val_loader)

    wandb.finish()

if __name__ == "__main__":  
    # Read configuration from the JSON file specified by MARS_CONFIG_FILE.
    os.environ["WANDB_DIR"] = datadir('logs_sweep')
    wandb.require("core")
    with open(os.path.join(configsdir(), GALFA_CONFIG_FILE), 'r') as f:
        sweep_config = yaml.full_load(f)
    
    sweep_id = wandb.sweep(sweep_config, project='scattering-vae-sweep')
    wandb.agent(sweep_id, sweep, project='scattering-vae-sweep', count=20)