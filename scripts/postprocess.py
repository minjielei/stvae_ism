import os
import numpy as np
import torch
from tqdm import tqdm

# from scripts.scov_synthesis_generic import st_synthesize
from scripts.facvae_trainer import FactorialVAETrainer
from gmvae.utils import (datadir, make_experiment_name, postdir, configsdir, 
                          parse_input_args, read_config, ScatCovDataset)

GALFA_PATH = datadir('galfa_hi')
GALFA_CONFIG_FILE = 'facvae_08_23_test.json'
GALFA_SCAT_COV_PATH = datadir(os.path.join(GALFA_PATH, 'scat_covs_h5'))

class PostProcess(object):
    """Class postprocessing results of a GMVAE training.
    """

    def __init__(self, args, network, dataset, device):
        # Pretrained GMVAE network.
        self.network = network
        self.network.eval()
        # The entire dataset.
        self.dataset = dataset
        # Scales.
        self.scales = args.scales
        self.in_shape = {
            scale: dataset.shape['scat_cov'][scale]
            for scale in self.scales
        }
        # Window size of the dataset.
        self.window_size = args.window_size
        # Device to perform computations on.
        self.device = device

        # (self.cluster_membership, self.cluster_membership_prob,
        #  self.confident_idxs,
        #  self.per_cluster_confident_idxs) = self.evaluate_model(
        #      args, data_loader)
         
        # Colors to be used for visualizing different clusters.
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        self.args = args

        # Create postproecess log directory 
        self.postdir = postdir(args.experiment)

    def get_cat_prob(self, idx):
        x= self.dataset.sample_data(idx, 'scat_cov')
        x = {scale: x[scale].to(self.device) for scale in self.scales}
        output = self.network(x)
        return output['prob_cat']['4096'].cpu().detach().numpy()

    def evaluate_model(self, data_loader):
        """
        Evaluate the trained FACVAE model.

        Here we pass the data through the trained model and for each window and
        each scale, we extract the cluster membership and the probability of
        the cluster membership. We then sort the windows based on the most
        confident cluster membership.

        Args:
            args: (argparse) arguments containing the model information.
            data_loader: (DataLoader) loader containing the data.
        """

        # Placeholder for cluster membership and probablity for all the data.
        cluster_membership = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        dtype=torch.long)
            for scale in self.scales
        }
        cluster_membership_prob = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        dtype=torch.float)
            for scale in self.scales
        }

        # Extract cluster memberships.
        for i_idx, idx in enumerate(data_loader):
            # Load data.
            x = self.dataset.sample_data(idx, 'scat_cov')
            # Move to `device`.
            x = {scale: x[scale].to(self.device) for scale in self.scales}
            # Run the input data through the pretrained GMVAE network.
            with torch.no_grad():
                output = self.network(x)
            # Extract the predicted cluster memberships.
            for scale in self.scales:
                cluster_membership[scale][np.sort(idx), :] = output['logits'][
                    scale].argmax(axis=1).reshape(
                        len(idx),
                        self.dataset.data['scat_cov'][scale].shape[1]).cpu()
                cluster_membership_prob[scale][np.sort(idx), :] = output[
                    'prob_cat'][scale].max(axis=1)[0].reshape(
                        len(idx),
                        self.dataset.data['scat_cov'][scale].shape[1]).cpu()

        # Sort indices based on most confident cluster predictions by the
        # network (increasing). The outcome is a dictionary with a key for each
        # scale, where the window indices are stored.
        confident_idxs = {}
        for scale in self.scales:
            # Flatten cluster_membership_prob into a 1D tensor.
            prob_flat = cluster_membership_prob[scale].flatten()

            # Sort the values in the flattened tensor in descending order and
            # return the indices.
            confident_idxs[scale] = torch.argsort(prob_flat,
                                                  descending=True).numpy()

        per_cluster_confident_idxs = {
            scale: {
                str(i): []
                for i in range(self.args.ncluster)
            }
            for scale in self.scales
        }

        for scale in self.scales:
            for i in range(len(confident_idxs[scale])):
                per_cluster_confident_idxs[scale][str(
                    cluster_membership[scale][confident_idxs[scale]
                                              [i]].item())].append(
                                                  confident_idxs[scale][i])

        return (cluster_membership, cluster_membership_prob, confident_idxs,
                per_cluster_confident_idxs)

    def reconstruct_data(self, sample_size=5):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        # Sample random data from loader
        indices = np.random.choice(np.arange(self.dataset.data['scat_cov']['4096'].shape[0]), size=sample_size, replace=False)
        x = self.dataset.sample_data(indices, 'scat_cov')
        x = {scale: x[scale].to(self.device) for scale in self.scales}
        
        # Obtain reconstructed data.
        with torch.no_grad():
            output = self.network(x)
            x_rec = output['x_rec']

        x = {scale: x[scale].cpu().detach().numpy() for scale in self.scales}
        x_rec = {scale: x_rec[scale].cpu().detach().numpy() for scale in self.scales}

        return x, x_rec

    def reconstruct_data_sample(self, data_loader, sample_size=5):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        # Sample random data from loader
        x = self.dataset.sample_data(next(iter(data_loader)), 'scat_cov')
        indices = np.random.randint(0, x[self.args.scales[0]].shape[0], size=sample_size)
        x = {scale: x[scale][indices, ...].to(self.device) for scale in self.scales}

        # Obtain reconstructed data.
        with torch.no_grad():
            output = self.network(x)
            x_rec = output['x_rec']

        x = {scale: x[scale].cpu().detach().numpy() for scale in self.scales}
        x_rec = {scale: x_rec[scale].cpu().detach().numpy() for scale in self.scales}

        return x, x_rec

    def random_generation(self, num_elements=3):
            """Random generation for each category

            Args:
                num_elements: (int) number of elements to generate

            Returns:
                generated data according to num_elements
            """
            # categories for each element
            arr = np.array([])
            for i in range(self.args.ncluster):
                arr = np.hstack([arr, np.ones(num_elements) * i])
            indices = arr.astype(int).tolist()

            categorical = torch.nn.functional.one_hot(
                torch.tensor(indices), self.args.ncluster).float().to(self.device)
            categorical = {scale: categorical for scale in self.args.scales}
            # infer the gaussian distribution according to the category
            mean, var = self.network.generative.pzy(categorical)

            # gaussian random sample by using the mean and variance
            gaussian = {scale: mean[scale] + torch.randn_like(var[scale]) * torch.sqrt(var[scale]) for scale in self.args.scales}

            # generate new samples with the given gaussian
            samples = self.network.generative.pxz(gaussian)
            samples = {scale: samples[scale].cpu().detach().numpy() for scale in self.scales}

            return samples
        
    def random_generation_sample(self, sample_idx, num_elements=3):
        """Random generation for each category

        Args:
            num_elements: (int) number of elements to generate

        Returns:
            generated data according to num_elements
        """
        # categories for each element
        arr = np.array([])
        for i in range(self.args.ncluster):
            arr = np.hstack([arr, np.ones(num_elements) * i])
        indices = arr.astype(int).tolist()

        categorical = torch.nn.functional.one_hot(
            torch.tensor(indices), self.args.ncluster).float().to(self.device)
        categorical = {scale: categorical for scale in self.args.scales}
        # infer the gaussian distribution according to the category
        sample_idx = np.ones(num_elements*self.args.ncluster) * sample_idx
        x = self.dataset.sample_data(sample_idx, 'scat_cov')
        x = {scale: x[scale].to(self.device) for scale in self.scales}
        qzxy = self.network.inference.qzxy(x, categorical)

        # gaussian random sample by using the mean and variance
        z = {scale: qzxy[scale][2] for scale in qzxy.keys()}
        
        # generate new samples with the given gaussian
        samples = self.network.generative.pxz(z)
        samples = {scale: samples[scale].cpu().detach().numpy() for scale in self.scales}

        return samples
        

    def latent_features(self, data_loader):
        """Obtain latent features learnt by the model

        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not

        Returns:
           features: (array) array containing the features from the data
        """
        N = len(data_loader.dataset)
        features = {
            scale:
            np.zeros([N, self.args.latent_dim])
            for scale in self.scales
        }
        clusters = {
            scale:
            np.zeros([N])
            for scale in self.scales
        }
        counter = 0
        with torch.no_grad():
            for idx in data_loader:
                # Load data batch.
                x = self.dataset.sample_data(idx, 'scat_cov')
                x = {scale: x[scale].to(self.device) for scale in x.keys()}

                # flatten data
                output = self.network.inference(x, self.network.gumbel_temp,
                                                self.network.hard_gumbel)
                for scale in self.scales:
                # cluster_membership[scale][np.sort(idx), :] = output['logits'][
                #     scale].argmax(axis=1).reshape(
                #         len(idx),
                #         self.dataset.data['scat_cov'][scale].shape[1]).cpu()
                # cluster_membership_prob[scale][np.sort(idx), :] = output[
                #     'prob_cat'][scale].max(axis=1)[0].reshape(
                #         len(idx),
                #         self.dataset.data['scat_cov'][scale].shape[1]).cpu()
                    latent_feat = output['mean'][scale]
                    cluster_membership = output['logits'][scale].argmax(axis=1)

                    features[scale][counter:counter +
                            x[scale].shape[0], :] = latent_feat.cpu().detach().numpy()[
                                ...]
                    clusters[scale][counter:counter + x[scale].shape[0]] = cluster_membership.cpu(
                    ).detach().numpy()[...]

                counter += x[scale].shape[0]

        return features, clusters
    
    def generate_scov_coeff(self, num_elements=3):
        """Generate scattering covariance coefficients using the posterior learned by the model

        Args:
                num_elements: (int) number of elements to generate

        Returns:
            save generated coefficients to file
        """
        
        samples = self.random_generation(self.args, num_elements=num_elements)
        samples = torch.from_numpy(samples['4096'])
        s_cov_arr = np.array([samples[i*num_elements:i*num_elements+num_elements, 0].numpy() for i in range(args.ncluster)])
        
        np.save(self.postdir + '/w_rec-' + str(args.w_rec)+'.npy', s_cov_arr)
        
    # def reconstruct_scov_coeff(self, data_loader, sample_size=5):
    #     """Reconstruct scattering covariance coefficients using the posterior learned by the model

    #     Args:
    #             num_elements: (int) number of elements to generate

    #     Returns:
    #         save generated coefficients to file
    #     """
    #     POST_PATH = datadir('postprocess')
        
    #     x, x_rec = self.reconstruct_data(data_loader, sample_size=sample_size)
    #     res = [x['4096'], x_rec['4096']]
        
    #     np.save(self.postdir + '/x_rec-' + str(self.args.w_rec)+'.npy', res)
        
    # def posterior_realization(self, sample_size = 20, nchunks=1):
    #     ## Load index information
    #     idx_info_full = np.load(GALFA_PATH+'/idx_info_J7_L4.npy', allow_pickle=True)
    #     idx_info_iso = np.load(GALFA_PATH+'/idx_info_J7_L4_iso.npy', allow_pickle=True)
    #     cov_type_full, _, _, _, _, _, _ = idx_info_full.T
    #     cov_type_iso, j1, a, b, l1, l2, l3 = idx_info_iso.T

    #     cov_type = np.empty(self.in_shape['4096'][-1], dtype=object)
    #     cov_type[0] = 'mean'
    #     N_P00 = (cov_type_full == 'P00').sum()
    #     cov_type[1:N_P00+1] = 'P00'
    #     cov_type[N_P00+1:] = cov_type_iso[(cov_type_iso != 'P00') & (cov_type_iso != 'mean')]
        
    #     ## Generate scov_coeff samples
    #     samples = self.random_generation(self.args, num_elements=sample_size)
    #     samples = torch.from_numpy(samples['4096'])
    #     samples=self.dataset.unnormalize(samples, type='scat_cov', dset_name='4096').numpy()
    #     s_cov_set = np.array([samples[i*sample_size:i*sample_size+sample_size, 0] for i in range(self.args.ncluster)])

    #     s_cov_arr = np.empty((s_cov_set.shape[0], s_cov_set.shape[1], cov_type_iso.shape[0]))
    #     s_cov_arr[...,cov_type_iso!='P00'] = s_cov_set[...,cov_type!='P00']
    #     s_cov_arr[...,cov_type_iso=='P00'] = s_cov_set[...,cov_type=='P00'].reshape((self.args.ncluster,sample_size,7,-1)).mean(-1)

    #     ## synthesize images from scov_coeff samples
    #     for i in range(self.args.ncluster):
    #         print("Synthesizing {} images for cluster {}".format(sample_size, i))
    #         target_coef = torch.from_numpy(s_cov_arr[i])
    #         P00 = torch.exp(torch.from_numpy(s_cov_set[i][:, cov_type == 'P00'])).reshape((sample_size,7,-1))
    #         image_syn = st_synthesize(
    #             target_coef, P00, nchunks=nchunks
    #         )
    #         np.save(self.postdir+'/img_syn_cluster{}_n{}.npy'.format(i, sample_size), image_syn)
            
if __name__ == '__main__':
    # Read configuration from the JSON file specified by MARS_CONFIG_FILE.
    args = read_config(os.path.join(configsdir(), GALFA_CONFIG_FILE))

    # Parse input arguments from the command line
    args = parse_input_args(args)

    # Random seed.
    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Set experiment name based on input arguments
    args.experiment = make_experiment_name(args)

    # Process filter_key and scales arguments to remove spaces and split by
    # comma
    if hasattr(args, 'filter_key'):
        args.filter_key = args.filter_key.replace(' ', '').split(',')
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
    network = facvae_trainer.load_checkpoint(args, args.max_epoch - 1)
    network.gumbel_temp = np.maximum(
                args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
                args.min_temp)
    post = PostProcess(args, network, dataset, test_loader, device)
    
    # produce posterior image realizations
    post.posterior_realization(sample_size=300, nchunks=30)