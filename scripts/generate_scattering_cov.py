""" Generate scattering covariance dataset from HI emission images. """
import h5py
import numpy as np
import os
from tqdm import tqdm
import scattering as st
from scattering.angle_transforms_simple import FourierAngle
from scattering.scale_transforms import FourierScale
from turbustat.statistics.apodizing_kernels import TukeyWindow

from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          make_h5_file_name)

# Path to GALFA-HI data directory.
DATA_PATH = datadir('galfa_hi')

# Scattering covariance generation parameters.
SCATCOV_CONFIG_FILE = 'generate_scatcov.json'

angle_operator = FourierAngle()
scale_operator = FourierScale()

def setup_hdf5_file(
    path,
    scat_cov_filename,
    window_size,
    max_img_num,
    scat_cov_size
):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 File.
    file = h5py.File(file_path, 'a')

    # Scattering covariance dataset of size `max_win_num x num_components x
    # scat_cov_size x 2`. The dataset will be resized at the end to reflect
    # the actual number of windows in the data. Chunks are chosen to be
    # efficient for extracting single-component scattering covariances for each
    # window.
    scatcov_group = file.require_group('scat_cov')
    scatcov_group.require_dataset(
        str(window_size),
        (max_img_num, 1, scat_cov_size),
        chunks=(1, 1, scat_cov_size),
        dtype=np.float32)

    file.close()

def update_hdf5_file(
    path,
    scat_cov_filename,
    window_size,
    img_idx,
    img_step,
    scat_covariances
):
    """
    Update the HDF5 file by writing new scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    # Write `scat_covariances` to the HDF5 file.
    file['scat_cov'][str(window_size)][img_idx:img_idx+img_step, 0, ...] = scat_covariances

    file.close()
    
# a function that prepares the argument "s_cov_func" to be given to the "synthesis" function
def harmonic_transform(s_cov_set, mask=None, output_info=False, if_iso=False):
    iso_suffix = '_iso' if if_iso else ''
    # get coefficient vectors and the index vectors
    s_cov = s_cov_set['for_synthesis'+iso_suffix]
    idx_info = st.scale_annotation_a_b(st.to_numpy(s_cov_set['index_for_synthesis'+iso_suffix]).T)

    s_cov, idx_info = angle_operator.fft(s_cov, idx_info, if_isotropic=if_iso)
    # s_cov, idx_info = scale_operator(s_cov, idx_info)

    # output
    if output_info:
        return idx_info, s_cov[:, mask] if mask is not None else s_cov
    else:
        return s_cov[:, mask] if mask is not None else s_cov

def compute_scat_cov(args):
    # Path to directory for creating scattering dataset.
    scat_cov_path = datadir(os.path.join(DATA_PATH, 'scat_covs_h5'))
    
    image_path = args.input_path+'galfa_high_latitude_p{}_s{}_v{}_filtered.h5'.format(args.N, args.S, args.vlim)
    image_data = h5py.File(image_path, 'r')['galfa']['image']
    nchannels = image_data.shape[0]
    npatches = image_data.shape[1]
    max_img_size = nchannels*npatches
    
    # set up scattering calculator
    shape = (args.N, args.N)
    taper = TukeyWindow(alpha=0.3)
    window = taper(shape) 
    st_calc = st.Scattering2d(args.N, args.N, args.J, args.L, device=args.device, weight=window)
    
    print('Computing scattering covariances and save to file...')
    num_images = 0
    for c in tqdm(range(nchannels)):
        input_data = image_data[c]/1e20
        if args.normalize:
            input_data = st.whiten(input_data)
        
        s_cov_set = st.chunk_model(input_data, st_calc, args.nchunks, remove_edge=False)
        if args.scov_idx == "harmonic":
            s_cov = st.to_numpy(harmonic_transform(s_cov_set, mask=None))
        else:
            s_cov = st.to_numpy(s_cov_set[args.scov_idx])
        # s_cov[:,0] = log_data_mean
        scat_cov_size = s_cov.shape[1]
        
        # Setup HDF5 file.
        if c == 0:
            setup_hdf5_file(scat_cov_path, args.scat_cov_filename, args.window_size, max_img_size, scat_cov_size)

        update_hdf5_file(
            scat_cov_path,
            args.scat_cov_filename,
            args.window_size, 
            num_images,
            npatches,
            s_cov
        )
        num_images += npatches

if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), SCATCOV_CONFIG_FILE))
    args = parse_input_args(args)
    args.scat_cov_filename = make_h5_file_name(args)
    print(args.scat_cov_filename)
    
    compute_scat_cov(args)