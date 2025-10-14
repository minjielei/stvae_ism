"""Script to perform denoising using synthesis of the output coefficients of scattering VAE model

Typical usage example:

"""

import numpy as np
import time
import torch
import scattering as st
from scattering.Scattering2d import Scattering2d, get_scattering_index
import wandb
import matplotlib.pyplot as plt

from gmvae.utils import datadir

class SourceSeparationSynthesis:
    """Separate components from a input image based on scattering vae model output
    """
    def __init__(self, args, J, L, M, N, device='gpu', wavelets='morlet', weight=None, seed=None, log_progress=True):
        self.args = args
        self.device = device
        if not torch.cuda.is_available(): self.device='cpu'
        np.random.seed(seed)
        self.J, self.L, self.M, self.N = J, L, M, N
        self.window = weight
        self.st_calc = Scattering2d(M, N, J, L, device, wavelets, weight=weight)
        self.log_progress = log_progress
        if log_progress:
            # Set up wandb for tracking training metrics
            wandb.require("core")
            wandb.init(
                # Set the project where this run will be logged
                project=args.project_name, 
                # set logs directory
                dir=datadir('source_separation'),
                # Track hyperparameters and run metadata
                config=args)
        
    def source_separation(self, estimator_name, image_original, vae_priors, image_init, 
                          normalization_coeff=None, C11_criteria=None, nchunks=1, mode='denoise',
                          loss_mode='full', w_cross=1, w_wnm=1, w_reg=1, s_cov_func=None, s_cov_func_params=[],
                          optim_algorithm='LBFGS', steps=300, learning_rate=0.2, stopping_patience = 5, stopping_delta = 0.01,
                          if_large_batch=False, print_each_step=False):
        if C11_criteria is None: C11_criteria = 'j2>=j1'
        # if image_init == 'random': np.random.normal(0,1,(image_original.shape[0],self.M,self.N))
        
        # setup normalization factors
        if normalization_coeff is None:
            if estimator_name=='s_cov_iso':
               self.normalization_coeff = torch.exp(vae_priors[...,1:1+self.J]).mean(1).reshape((-1,1,self.J,1))
            else:
                self.normalization_coeff = torch.exp(vae_priors[...,1:1+self.J*self.L]).mean(1).reshape((-1,1,self.J,self.L))
        else:
            self.normalization_coeff = normalization_coeff

        # set up estimator function
        select_and_index = get_scattering_index(self.J, self.L, num_field=2)
        if 's_cov_func' in estimator_name:
            def func_s(image):
                s_cov_set = self.st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, 
                    C11_criteria=C11_criteria
                )
                return s_cov_func(s_cov_set, s_cov_func_params)
            def func_s_2field(target, image):
                result = self.st_calc.scattering_cross_cov(
                    target, image, use_ref=True, normalization='P00'
                )
                N_image = target.shape[0]
                C01 = result['C01'][:,2:,select_and_index['select_2']]
                C11 = result['Corr11'][:,2:,select_and_index['select_3']]
                for_synthesis = torch.cat((
                    # C00.reshape((N_image, -1)).real, 
                    # C00.reshape((N_image, -1)).imag, 
                    C01.reshape((N_image, -1)).real,
                    C01.reshape((N_image, -1)).imag,
                    C11.reshape((N_image, -1)).real,
                    C11.reshape((N_image, -1)).imag
                ), dim=-1)
                return for_synthesis
        if estimator_name=='s_cov_iso':
            func_s = lambda x: self.st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria)['for_synthesis_iso']
            def func_s_2field(target, image):
                result = self.st_calc.scattering_cross_cov(
                    target, image, use_ref=True, normalization='P00'
                )
                N_image = target.shape[0]
                C00 = result['Corr00_iso']
                C01 = result['C01_iso'][:,select_and_index['select_2_iso']]
                C11 = result['Corr11_iso'][:,select_and_index['select_3_iso']]
                for_synthesis_iso = torch.cat((
                    C00.real, 
                    C01.reshape((N_image, -1)),
                    C11.reshape((N_image, -1)),
                ), dim=-1)
                return for_synthesis_iso
        if estimator_name=='s_cov':
            func_s = lambda x: self.st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria)['for_synthesis']
            def func_s_2field(target, image):
                result = self.st_calc.scattering_cov_2fields(
                    target, image, use_ref=True, normalization='P00'
                )
                N_image = target.shape[0]
                C00 = result['Corr00']
                C01 = result['C01'][:,2:,select_and_index['select_2']]
                C11 = result['Corr11'][:,2:,select_and_index['select_3']]
                for_synthesis = torch.cat((
                    # C00.reshape((N_image, -1)).real, 
                    # C00.reshape((N_image, -1)).imag, 
                    C01.reshape((N_image, -1)).real,
                    C01.reshape((N_image, -1)).imag,
                    C11.reshape((N_image, -1)).real,
                    C11.reshape((N_image, -1)).imag
                ), dim=-1)
                return for_synthesis
        
        func = lambda x, n: self.chunk_model(x, func_s, n)
        func_2field = lambda xa, xb, n: self.chunk_model_2fields(xa, xb, func_s_2field, n)
        
        image_syn = self.synthesis(
            image_original, vae_priors, image_init, func, func_2field, nchunks=nchunks,
            mode=mode, optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
            loss_mode=loss_mode, w_cross=w_cross, w_wnm=w_wnm, w_reg=w_reg,
            stopping_patience=stopping_patience, stopping_delta = stopping_delta, print_each_step=print_each_step
        )
        
        return image_syn
    
    def synthesis(
        self, image_original, vae_priors, image_init, 
        estimator_function, estimator_2field, w_cross =1, w_wnm=1, w_reg=1, 
        stopping_patience = 5, stopping_delta = 0.01, nchunks=1, mode='denoise', loss_mode='full', 
        optim_algorithm='LBFGS', steps=300, learning_rate=0.2, print_each_step=False
    ):
        # define parameters
        N_image = image_original.shape[0]
        N_components = vae_priors.shape[0]
        N_realizations = vae_priors.shape[1]
        M = image_original.shape[1]
        N = image_original.shape[2]

        # formating input images
        if type(vae_priors)==np.ndarray:
            vae_priors = torch.from_numpy(vae_priors)
        if type(image_init)==np.ndarray:
            image_init = torch.from_numpy(image_init)
        if type(image_original)==np.ndarray:
            image_original = torch.from_numpy(image_original)
        if type(self.window)==np.ndarray:
            self.window = torch.from_numpy(self.window)
        if self.device=='gpu':
            vae_priors = vae_priors.cuda()
            image_init = image_init.cuda()
            image_original = image_original.cuda()

            self.window = self.window.cuda()
        self.image_original = image_original
        mask = self.window > 0.8

        # Precompute the normalization factors
        estimator_target = []
        cross_estimator_target = []
        for m in range(N_components):
            self.st_calc.add_synthesis_P00(self.normalization_coeff[m,...])
            scov_target = estimator_function(vae_priors[m,...], nchunks)
            estimator_target.append(scov_target)
            if mode == 'denoise' and loss_mode == 'full':
                cross_estimator_target = []
                image_ref = image_original.unsqueeze(1).expand(-1, N_realizations, -1, -1)
                image_prior = vae_priors[[-1]].expand(N_image, -1, -1, -1)
                self.st_calc.add_synthesis_P00_ab(self.normalization_coeff[0,...], self.normalization_coeff[1,...])
                for n in range(N_image):
                    cross_estimator_target.append(estimator_2field(image_ref[n], image_prior[n], nchunks))
                cross_estimator_target = torch.stack(cross_estimator_target, dim=0)

        print('# of estimators: ', estimator_target[0].shape[-1])

        # define optimizable image model
        class OptimizableImage(torch.nn.Module):
            def __init__(self, input_init, Fourier=False):
                # super(OptimizableImage, self).__init__()
                super().__init__()
                self.param = torch.nn.Parameter( input_init )
                
                if Fourier: 
                    self.image = torch.fft.ifftn(
                        self.param[0] + 1j*self.param[1],
                        dim=(-2,-1)).real
                else: self.image = self.param
        
        image_model = OptimizableImage(image_init)    

        # wandb.watch(image_model, log_freq=5)
        
        # print('Image model shape: ', image_model.image.shape)

        # define optimizer
        if optim_algorithm   =='Adam':
            optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='NAdam':
            optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='SGD':
            optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='Adamax':
            optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='LBFGS':
            optimizer = torch.optim.LBFGS(image_model.parameters(), lr=learning_rate, 
                max_iter=1, max_eval=None, 
                tolerance_grad=1e-8, tolerance_change=1e-10, 
                history_size=min(steps//2, 100), line_search_fn=None
            )

        early_stopping = EarlyStopping(image_init, stopping_patience, stopping_delta)
        
        # optimize
        def closure():
            optimizer.zero_grad()
            loss, prior_loss, cross_loss, reg_loss = 0, 0, 0, 0
            # compute loss terms for the physical components
            prior_loss_per_component = []
            image_residual = image_original-image_model.image
            image_syn = torch.stack([image_model.image, image_residual], dim=0)

            # compute loss terms
            if mode == 'denoise':
                components = [0, 1]
            else:
                components = [1]
            for c in components:
                self.st_calc.add_synthesis_P00(self.normalization_coeff[c,...])
                estimator_model = estimator_function(image_syn[c], 1)
                gap = estimator_model.unsqueeze(1)-estimator_target[c].unsqueeze(0)
                norm = estimator_target[c]
                tmp = self.prior_loss(gap, norm)
                if c == 0 and mode == 'denoise': tmp *= w_wnm
                prior_loss_per_component.append(tmp)
                prior_loss += tmp
                if loss_mode == 'full':
                    self.st_calc.add_synthesis_P00_ab(self.normalization_coeff[0,...], self.normalization_coeff[1,...])
                    gap = estimator_2field(image_model.image, image_residual, 1)
                    norm = cross_estimator_target
                    tmp = self.cross_loss(gap, norm, w_cross)
                    cross_loss += tmp
            
            loss = prior_loss + cross_loss

            ## add bright pixel regularization and early stopping if mode is phase decomposition
            eps = 1e-6
            if mode == 'phase_decomp':
                image_fcnm = (image_model.image / (image_original+eps))
                reg_loss += self.image_reg_loss((image_model.image / (image_original+eps)**2), self.window, w_reg)
                loss += reg_loss
                early_stopping(prior_loss)
                # Hacky way to deal with loss blow up for low SNR images
                if prior_loss > 100 * early_stopping.best_loss:
                    early_stopping.early_stop = True
                    image_model.image = torch.nn.Parameter(early_stopping.best_syn)
                    image_residual = image_original-image_model.image
                    image_syn = torch.stack([image_model.image, image_residual], dim=0)
                    image_fcnm = (image_model.image / (image_original+eps))
                    print(f'INFO: Early stopping due to loss blow up, best loss: {early_stopping.best_loss:.4f}')
                else: early_stopping.best_syn = image_model.image.clone().detach()

            if print_each_step:
                if (i%10==0 or i%10==-1):
                    print(f"Step {i}: Total Loss: {loss:.4f}, Prior Loss: {prior_loss:.4f}, Cross Loss: {cross_loss:.4f}, Reg Loss: {reg_loss:.4f}")
                    print("Prior Loss per component: ", [np.round(loss.item(), decimals=4) for loss in prior_loss_per_component])
                    # if loss_mode == 'full':
                    #     print("Cross Loss per component: ", [np.round(loss.item(), decimals=4) for loss in cross_loss_per_component])
            # log metrics to wandb
            if self.log_progress:
                wandb.log({"tot_loss": loss, "prior_loss": prior_loss, "cross_loss": cross_loss, "reg loss": reg_loss}, commit=False)
                if mode == 'denoise':
                    image_total = torch.stack([image_original, image_model.image, image_residual], dim=0)
                    labels = ['Original', 'Data', 'Noise']
                else:
                    image_total = torch.cat([image_original.unsqueeze(0), image_syn, image_fcnm.unsqueeze(0)], dim=0)
                    labels = ['Original', 'CNM', 'WNM','fCNM']
                    for m in range(N_image):
                        fcnm = image_model.image[m][mask].nanmean()/image_original[m][mask].nanmean()
                        wandb.log({f"fcnm_{m}": fcnm}, commit=False)
                self.log_image_syn(image_total.cpu().detach().numpy(), mask.cpu().detach().numpy(), labels=labels, commit=True)
            
            loss.backward()
            
            return loss
                
        # optimize
        t_start = time.time()
        if mode == 'denoise':
            cmin, cmax = 0, None
        else:
            cmin, cmax = image_original-image_original, image_original
        if optim_algorithm =='LBFGS':
            for i in range(steps):
                if early_stopping.early_stop:
                    break
                optimizer.step(closure)
                for p in image_model.parameters():
                    p.data.clamp_(min=cmin, max=cmax)
        else:
            for i in range(steps):
                # print('step: ', i)
                optimizer.step(closure)
                for p in image_model.parameters():
                    p.data.clamp_(min=cmin, max=cmax)
        t_end = time.time()
        print('time used: ', t_end - t_start, 's')

        image_residual = image_original-image_model.image
        image_syn = torch.stack([image_model.image, image_residual], dim=0)
        
        return image_syn.cpu().detach().numpy()
    
    # for large data, scattering computation needs to be chunked to hold on memory
    def chunk_model(self, X, estimator_function, nchunks=1):
        partition = np.array_split(np.arange(X.shape[0]), nchunks)
        res = []
        for part in partition:
            X_part = X[part,...]
            res_part = estimator_function(X_part)
            res.append(res_part)
        return torch.cat(res, dim=0)

    def chunk_model_2fields(self, X, Xb, estimator_function, nchunks=1):
        partition = np.array_split(np.arange(X.shape[0]), nchunks)
        res = []
        for part in partition:
            X_part = X[part,...]
            Xb_part = Xb[part,...]
            res_part = estimator_function(X_part, Xb_part)
            res.append(res_part)
        return torch.cat(res, dim=0)
    
    # define loss function
    def quadratic_loss(self, target, model):
        return ((target - model)**2).mean()*1e8
        
    # define vae source separation prior loss term
    def prior_loss(self, model_target_diff, normalization, no_amplitude=False):
        gap = (model_target_diff**2)
        norm = normalization.var(axis=0)
        res = gap / norm[None, None, ...]
        if no_amplitude:
            return (gap / norm)[...,2:].mean()
        return res.mean()
    
    def prior_cnm_loss(self, model_target_diff, normalization):
        gap = (model_target_diff**2)
        norm = normalization.var(axis=0)
        res = gap / norm[None, None, ...]
        res = self.softmin(res.mean(-1), dim=1, temperature=0.1)
        return res.mean()
    
    def cross_loss(self, cross_coef, normalization, w_cross=1):
        gap = w_cross * (cross_coef**2)
        norm = normalization.var(axis=1)
        return (gap / norm[None, None, :]).mean()
    
    def image_reg_loss(self, image, window, weight):
        res = ((image*window)).nanmean()*weight
        return res
    
    def softmin(self, x, dim=0, temperature=1):
        return torch.sum(x*torch.nn.functional.softmax(-x / temperature, dim=dim), dim=dim)
    
    def softmax(self, x, dim=0, temperature=1):
        return torch.sum(x*torch.nn.functional.softmax(x / temperature, dim=dim), dim=dim)
    
    # plot synthesized image
    def log_image_syn(self, image_syn, window, labels, commit=False):
        n_cluster = image_syn.shape[0]
        n_image = image_syn.shape[1]
        fig, ax = plt.subplots(n_image, n_cluster, constrained_layout=True, squeeze=False)
        for i in range(n_image):
            for j in range(n_cluster):
                img = image_syn[j,i]
                img[window<0.5] = np.nan
                if j == n_cluster-1:
                    ax[i,j].imshow(img, cmap='viridis')
                else:
                    ax[i,j].imshow(img, cmap='viridis')
                ax[i,j].axis('off')
                ax[0,j].set_title(labels[j], fontsize=10)
        wandb.log({f"image_syn": fig}, commit=commit)
        plt.close()

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain iterations.
    """
    def __init__(self, image_init, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_syn = image_init
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif (self.best_loss - loss < self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True