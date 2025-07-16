import numpy as np
import os
import shutil
import torch
import wandb
from tqdm import tqdm

from facvae.utils import checkpointsdir, CustomLRScheduler, logsdir
from facvae.vae import FactorialVAE, LossFunctions


class FactorialVAETrainer(object):
    """Class training a Gaussian mixture variational autoencoder model.
    """

    def __init__(self, args, dataset, device):
        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_cat
        self.dataset = dataset
        self.device = device

        # Network architecture.
        self.in_shape = {
            scale: dataset.shape['scat_cov'][scale]
            for scale in args.scales
        }
        print('Multiscale scattering covariance shapes: ', self.in_shape)
        self.network = FactorialVAE(self.in_shape,
                                    args.latent_dim,
                                    args.ncluster,
                                    args.init_temp,
                                    hidden_dim=args.hidden_dim,
                                    nlayer=args.nlayer,
                                    slope=args.relu_slope).to(self.device)

        self.train_log = {
            'rec': [],
            'gauss': [],
            'cat': [],
            'vae': [],
        }
        self.val_log = {key: [] for key in self.train_log}

        # Loss functions.
        self.losses = LossFunctions()

    def compute_loss(self, data, out_net):
        """Loss functions derived from the variational lower bound.

        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the
            network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and
            predictions
        """
        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        rec_loss = 0.0
        gauss_loss = 0.0
        cat_loss = 0.0
        cat_loss_prior = 0.0
        for scale in data.keys():
            # Reconstruction loss.
            rec_loss += self.losses.reconstruction_loss(
                data[scale], data_recon[scale], 'mse') / len(data.keys())

            # Gaussian loss.
            gauss_loss += self.losses.gaussian_loss(
                z[scale], mu[scale], var[scale], y_mu[scale],
                y_var[scale]) / len(data.keys())
            # gauss_loss = self.losses.gaussian_closed_form_loss(
            #     mu[scale], var[scale], y_mu[scale], y_var[scale])

            # Categorical loss (posterior).
            cat_loss -= self.losses.entropy(logits[scale],
                                            prob_cat[scale]) / len(data.keys())

            # Categorical prior.
            pi = torch.ones_like(prob_cat[scale])
            cat_loss_prior += self.losses.entropy(pi, prob_cat[scale]) / len(
                data.keys())

        # Total loss.
        vae_loss = (self.w_rec * rec_loss + self.w_gauss * gauss_loss +
                    self.w_cat * (cat_loss + cat_loss_prior))

        # Obtain predictions.
        clusters = {
            scale: torch.max(logits[scale], dim=1)[1]
            for scale in logits.keys()
        }

        return {
            'vae': vae_loss,
            'rec': rec_loss,
            'gauss': gauss_loss,
            'cat': cat_loss + cat_loss_prior,
            'clusters': clusters
        }
    
    def make_categorical(self, args, num_samples=300):
        # categories for each element
        arr = np.array([])
        for i in range(args.ncluster):
            arr = np.hstack([arr, np.ones(num_samples) * i])
        indices = arr.astype(int).tolist()

        categorical = torch.nn.functional.one_hot(
            torch.tensor(indices), args.ncluster).float().to(self.device)
        categorical = {scale: categorical for scale in args.scales}
        return categorical
    
    def sample_clusters(self, args, categorical, num_samples=300):
        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        gaussian = {scale: mean[scale] + torch.randn_like(var[scale]) * torch.sqrt(var[scale]) for scale in args.scales}

        # generate new samples with the given gaussian
        samples = self.network.generative.pxz(gaussian)['4096']

        return samples.reshape(args.ncluster, num_samples, -1)
        
    def compute_cluster_metrics(self, args, samples):
        mean_var = samples.var(dim=1).mean(dim=1)
        mean_separation = [torch.abs(samples.mean(dim=1)[i] - samples.mean(dim=1)[i-1]).mean() for i in range(args.ncluster)]
        return mean_var, mean_separation

    def train(self, args, train_loader, val_loader):
        """Train the model

        Args:
            train_loader: (DataLoader) corresponding loader containing the
            training data val_loader: (DataLoader) corresponding loader
            containing the validation data

        Returns:
            output: (dict) contains the history of train/val loss
        """
        # Optimizer.
        optim = torch.optim.Adam(self.network.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)

        # Setup the learning rate scheduler.
        scheduler = CustomLRScheduler(optim, args.lr, args.lr_final,
                                      args.max_epoch)

        self.steps_per_epoch = len(train_loader)
        print(f'Number of steps per epoch: {self.steps_per_epoch}')
        
        categorical = self.make_categorical(args)

        # Training loop, run for `args.max_epoch` epochs.
        with tqdm(range(args.max_epoch),
                  unit='epoch',
                  colour='#B5F2A9',
                  dynamic_ncols=True) as pb:
            for epoch in pb:
                self.network.train()
                # iterate over the dataset
                # Update learning rate.
                scheduler.step()
                for i_idx, idx in enumerate(train_loader):
                    # Reset gradient attributes.
                    optim.zero_grad()

                    # Load data batch.
                    x = self.dataset.sample_data(idx, 'scat_cov')
                    x = {
                        scale: x[scale].to(self.device)
                        for scale in self.in_shape.keys()
                    }

                    # Forward call.
                    y = self.network(x)

                    # Compute loss.
                    train_loss = self.compute_loss(x, y)
                    # Compute gradients.
                    train_loss['vae'].backward()

                    for p in self.network.parameters():
                        if p.requires_grad:
                            p.grad.clamp_(-args.clip, args.clip)
                    # Update parameters.
                    optim.step()

                    if i_idx % 10 == 0:
                        # Update progress bar.
                        self.progress_bar(pb, train_loss)

                # Log progress.
                if epoch % 1 == 0:
                    self.network.eval()
                    with torch.no_grad():
                        x_val = self.dataset.sample_data(
                            next(iter(val_loader)), 'scat_cov')
                        x_val = {
                            scale: x_val[scale].to(self.device)
                            for scale in self.in_shape.keys()
                        }

                        y_val = self.network(x_val)
                        val_loss = self.compute_loss(x_val, y_val)

                        samples = self.sample_clusters(args, categorical)
                        mean_var, mean_separation = self.compute_cluster_metrics(args, samples)

                        self.log_progress(args, epoch, train_loss, val_loss, mean_var, mean_separation)

                # Decay gumbel temperature
                if args.temp_decay > 0:
                    self.network.gumbel_temp = np.maximum(
                        args.init_temp * np.exp(-args.temp_decay * epoch),
                        args.min_temp)

                if epoch == args.max_epoch - 1 or (self.steps_per_epoch > 10
                                                   and epoch % 100 == 0):
                    torch.save(
                        {
                            'model_state_dict': self.network.state_dict(),
                            'optim_state_dict': optim.state_dict(),
                            'epoch': epoch,
                            'train_log': self.train_log,
                            'val_log': self.val_log
                        },
                        os.path.join(checkpointsdir(args.experiment),
                                     f'checkpoint_{epoch}.pth'))

    def progress_bar(self, pb, train_loss):
        progress_bar_dict = {}
        for key, item in train_loss.items():
            if key != 'clusters':
                progress_bar_dict[key] = f'{item.item():2.2f}'
        # Progress bar.
        pb.set_postfix(progress_bar_dict)

    def log_progress(self, args, epoch, train_loss, val_loss, mean_var, mean_separation):
        """Log progress of training."""
        # Bookkeeping.
        for key, item in train_loss.items():
            if key != 'clusters':
                self.train_log[key].append(item.item())
        for key, item in val_loss.items():
            if key != 'clusters':
                self.val_log[key].append(item.item())

        for scale in self.in_shape.keys():
            wandb.log(
                {
                    f'classes_train_{scale}_{str(i)}': (train_loss['clusters'][scale]
                             == i).cpu().numpy().astype(float).mean()
                    for i in range(args.ncluster)
                }, commit=False)

            wandb.log(
                {
                    f'classes_val_{scale}_{str(i)}': (val_loss['clusters'][scale]
                             == i).cpu().numpy().astype(float).mean()
                    for i in range(args.ncluster)
                }, commit=False)

        wandb.log({
            'vae_loss_train': train_loss['vae'],
            'vae_loss_val': val_loss['vae']
        }, commit=False)

        wandb.log({
            'rec_loss_train': train_loss['rec'],
            'rec_loss_val': val_loss['rec']
        }, commit=False)

        wandb.log({
            'gauss_loss_train': train_loss['gauss'],
            'gauss_loss_val': val_loss['gauss']
        }, commit=False)

        wandb.log({
            'cat_loss_train': train_loss['cat'],
            'cat_loss_val': val_loss['cat']
        }, commit=False)

        wandb.log({
                f"cluster_variance_{i}": mean_var[i]
                for i in range(args.ncluster)
            }, commit=False)
        
        wandb.log({
                'gumbel_temperature': self.network.gumbel_temp,
            }, commit=False)
        
        wandb.log({
                f"cluster_separation_{i}-{(i-1)%args.ncluster}": mean_separation[i]
                for i in range(args.ncluster)
            }, commit=True)

    def load_checkpoint(self, args, epoch):
        file_to_load = os.path.join(checkpointsdir(args.experiment),
                                    'checkpoint_' + str(epoch) + '.pth')
        if os.path.isfile(file_to_load):
            if self.device == torch.device(type='cpu'):
                checkpoint = torch.load(file_to_load, map_location='cpu')
            else:
                checkpoint = torch.load(file_to_load)

            self.network.load_state_dict(checkpoint['model_state_dict'])

            if not epoch == checkpoint["epoch"]:
                raise ValueError(
                    'Inconsistent filename and loaded checkpoint.')
        else:
            raise ValueError('Checkpoint does not exist.')
        return self.network
