import os, argparse
import torch
import numpy as np
from torch.nn import functional as F
from scipy.stats import wasserstein_distance
import scattering as st
from tqdm import tqdm

from scripts.facvae_trainer import FactorialVAETrainer
from scripts.postprocess import PostProcess
from facvae.utils import (configsdir, datadir, read_config, make_experiment_name, ScatCovDataset)

def reconstruction_loss(real, predicted, rec_type='mse'):
        """Reconstruction loss between the true and predicted outputs
        mse = (1/n)*Σ(real - predicted)^2
        bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
        """
        if rec_type == 'mse':
            loss = np.power(real - predicted, 2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise ValueError("invalid loss function... try bce or mse...")
        return loss.mean(axis=0)

def load_model(config_file, epoch=None, dataset='galfa_hi'):
    # Paths to scattering covariance dataset and model configs.
    GALFA_PATH = datadir(dataset)
    GALFA_SCAT_COV_PATH = datadir(os.path.join(GALFA_PATH, 'scat_covs_h5'))
    GALFA_CONFIG_FILE = config_file

    # Read configuration from the JSON file specified by MARS_CONFIG_FILE.
    args = read_config(os.path.join(configsdir(), GALFA_CONFIG_FILE))
    def parse_input_args(args):
        "Use variables in args to create command line input parser."
        parser = argparse.ArgumentParser(description='')
        for key, value in args.items():
            parser.add_argument('--' + key, default=value, type=type(value))
        return parser.parse_args('')
    args = parse_input_args(args)
    args.experiment = make_experiment_name(args)
    if hasattr(args, 'scales'):
        args.scales = args.scales.replace(' ', '').split(',')

    # Load the dataset
    dataset = ScatCovDataset(os.path.join(GALFA_SCAT_COV_PATH,
                                                    args.h5_filename),
                                        0.90,
                                        scatcov_datasets=args.scales,
                                        load_to_memory=args.load_to_memory,
                                        normalize_data=args.normalize)
    train_loader = torch.utils.data.DataLoader(dataset.train_idx,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset.val_idx,
                                                batch_size=args.batchsize,
                                                shuffle=True,
                                                drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset.test_idx,
                                              batch_size=args.batchsize,
                                              shuffle=False,
                                              drop_last=False)

    # Initialize facvae trainer with the input arguments, dataset, and device
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    facvae_trainer = FactorialVAETrainer(args, dataset, device)

    # Load a saved checkpoint for testing.
    if epoch is not None:
        network = facvae_trainer.load_checkpoint(args, epoch)
    else:
        network = facvae_trainer.load_checkpoint(args, args.max_epoch - 1)
    network.gumbel_temp = np.maximum(
                args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
                args.min_temp)
    post = PostProcess(args, network, dataset, device)

    return post, train_loader, val_loader, test_loader

def wass_dist(dset1, dset2):
    "Compute the Wasserstein distance between two datasets."
    N = dset1.shape[1]
    dist_arr = np.zeros(N)
    for i in range(N):
        dist_arr[i] = wasserstein_distance(dset1[:,i], dset2[:,i])
    return dist_arr

def chunk_synthesis(
    target_coef, nchunks, reference_P00=None,
    estimator_name='s_cov_iso', mode='estimator', 
    image_init=None, remove_edge=False, window = None,
    mask=None, threshold_func=None, print_each_step=False, optim_algo='LBFGS',
    M=256, J=7, L=4, seed=None, steps=400, learning_rate=0.2, if_large_batch=False,
):
    "Synthesize an image from scattering coefficients in chunks."
    partition = np.array_split(np.arange(target_coef.shape[0]), nchunks)
    img_syn = []
    for part in tqdm(partition):
        target_coef_part = target_coef[part,...]
        image_init_part = image_init[part,...] if image_init is not None else None
        if reference_P00 is not None:
            reference_P00_part = reference_P00[part,...]
        else:
            reference_P00_part = None
        img_syn_part = st.synthesis(
            estimator_name, mode=mode, M=M, N=M, J=J, L=L, seed=seed, image_init=image_init_part,
            target=target_coef_part, s_cov_func_params=mask, s_cov_func=threshold_func, 
            steps=steps, learning_rate=learning_rate, print_each_step=print_each_step, optim_algorithm=optim_algo,
            remove_edge=remove_edge, weight = window, reference_P00=reference_P00_part, if_large_batch=if_large_batch
        )
        img_syn.append(img_syn_part)
    return np.concatenate(img_syn, axis=0)