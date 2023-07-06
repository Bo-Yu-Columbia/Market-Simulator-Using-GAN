# This file is used to run a set of experiments with different models on different datasets.
# The results of the experiments are then saved for further analysis.

import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import download_man_ahl_dataset, download_mit_ecg_dataset
from lib.data import get_data
from lib.plot import savefig, create_summary
from lib.utils import pickle_it


def get_algo_config(dataset, data_params):
    """ 
    Get the configuration parameters for a specific algorithm based on the dataset.
    
    Parameters:
    dataset: The dataset used for the experiment.
    data_params: The parameters of the dataset.
    
    Returns:
    The algorithm configuration parameters.
    """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    elif dataset == 'YIELD':
        key += '_' + '_'.join(data_params['durations'])
    elif dataset == 'EIB':
        key += '_' + '_'.join(data_params['durations'])
    elif dataset == 'EXCHANGE':
        key += '_' + '_'.join(data_params['exchanges'])
    return SIGCWGAN_CONFIGS[key]


def set_seed(seed):
    """ 
    Sets the seed for torch and numpy for reproducibility.
    
    Parameters:
    seed: The seed number.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_algo(algo_id, base_config, dataset, data_params, x_real):
    """
    Retrieves the desired algorithm configuration.
    
    Parameters:
    algo_id: Identifier of the algorithm to use.
    base_config: Basic configuration parameters.
    dataset: The dataset used for the experiment.
    data_params: The parameters of the dataset.
    x_real: The real dataset.
    
    Returns:
    The chosen algorithm with the configured parameters.
    """
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def run(algo_id, base_config, base_dir, dataset, spec, result_dir, data_params={}):
    """ 
    Main function that runs the algorithm on the dataset and saves the results.
    
    Parameters:
    algo_id: Identifier of the algorithm to use.
    base_config: Basic configuration parameters.
    base_dir: The base directory to save the results.
    dataset: The dataset used for the experiment.
    spec: Specification parameters.
    data_params: The parameters of the dataset.
    """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    experiment_directory = pt.join(base_dir, dataset, result_dir, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        os.makedirs(experiment_directory)
    set_seed(base_config.seed)
    x_real = get_data(dataset, base_config.p, base_config.q, **data_params)
    x_real = x_real.to(base_config.device)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:]

    # Initialize the chosen algorithm with the real data and the configurations
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real)

    # Train the algorithm
    algo.fit()

    # After the training, we create a summary of the experiment
    create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_real)

    # Save the summary as an image in the experiment directory
    savefig('summary.png', experiment_directory)

    # Create a long summary and save it as well
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, 8000, x_real, one=True)
    savefig('summary_long.png', experiment_directory)

    # Plot the first 2000 elements of the fake data and save the plot
    plt.plot(x_fake.cpu().numpy()[0, :2000])
    savefig('long_path.png', experiment_directory)

    # Save the real path, generator weights, and training loss for further analysis
    pickle_it(x_real, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))

    # Plot the losses during training and save the plot
    algo.plot_losses()
    savefig('losses.png', experiment_directory)


def get_dataset_configuration(dataset):
    """
    Retrieves the specific configuration for a given dataset.
    
    Parameters:
    dataset: The dataset used for the experiment.
    
    Returns:
    A generator object that yields the specifications and parameters for each configuration of the dataset.
    """
    if dataset == 'ECG':
        generator = [('id=100', dict(filenames=['100']))]
    elif dataset == 'STOCKS':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('SPX',), ('SPX', 'DJI')])
    elif dataset == 'YIELD':
        # generator = (('_'.join(duration), dict(durations=duration)) for duration in [('1Yr',), ('1Yr', '3Yr') , ('1Yr', '3Yr', '10Yr')])
        generator = (('_'.join(duration), dict(durations=duration)) for duration in [('1Yr',)])
    elif dataset == 'EXCHANGE':
        generator = (('_'.join(exchange), dict(exchanges=exchange)) for exchange in [('JPYUSD',), ('JPYUSD', 'EURUSD')])
    elif dataset == 'VAR':
        par1 = itertools.product([1], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)])
        par2 = itertools.product([2], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        par3 = itertools.product([3], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        combinations = itertools.chain(par1, par2, par3)
        generator = (
            ('dim={}_phi={}_sigma={}'.format(dim, phi, sigma), dict(dim=dim, phi=phi, sigma=sigma))
            for dim, (phi, sigma) in combinations
        )
    elif dataset == 'ARCH':
        # if dataset is ARCH, it generates configurations based on different lag values
        generator = (('lag={}'.format(lag), dict(lag=lag)) for lag in [3])
    elif dataset == 'SINE':
        # if dataset is SINE, it generates a single configuration
        generator = [('a', dict())]
    elif dataset == 'EIB':
        generator = (('_'.join(duration), dict(durations=duration))
                     for duration in [('1yr',), ('1yr', '5yr')])
                     # for duration in [('1yr',), ('1yr', '5yr'), ('1yr', '5yr', '10yr')])
    else:
        # if the dataset is not recognized, it raises an exception
        raise Exception('%s not a valid data type.' % dataset)
    return generator

def name_train_script_result_dir(p, q, hidden_dims):
    """
    Creates a directory name for the training script results.

    Parameters:
    p: The length of past path p.
    q: The length of future path q.
    hidden_dims: The hidden dimensions of the generator.

    Returns:
    A string representing the directory name.
    """
    return 'p={}_q={}_hidden_dims={}'.format(str(p), str(q), str(hidden_dims))

def main(args):
    """
    The main function, it orchestrates the entire process of training and evaluating the algorithm based on the arguments provided.

    Parameters:
    args: The command line arguments that control the training and evaluation process.
    """
    # Check if the data directory exists, if not create one
    if not pt.exists('./data'):
        os.mkdir('./data')

    # Check if the Oxford MAN AHL dataset exists, if not download it
    if not pt.exists('./data/oxfordmanrealizedvolatilityindices.csv'):
        print('Downloading Oxford MAN AHL realised library...')
        download_man_ahl_dataset()

    # Check if the MIT-ECG database exists, if not download it
    if not pt.exists('./data/mit_db'):
       print('Downloading MIT-ECG database...')
       download_mit_ecg_dataset()

    print('Start of training. CUDA: %s' % args.use_cuda)
    # Iterate through the chosen datasets
    for dataset in args.datasets:
        # Iterate through the chosen algorithms
        for algo_id in args.algos:
            # Set the seed value for reproducibility
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                # Initialize the base configuration for the algorithm
                base_config = BaseConfig(
                    device='cuda' if args.use_cuda else 'cpu',
                    seed=seed,
                    batch_size=args.batch_size,
                    hidden_dims=args.hidden_dims,
                    p=args.p,
                    q=args.q,
                    total_steps=args.total_steps,
                )
                # Get the dataset configuration
                result_dir = name_train_script_result_dir(args.p, args.q, args.hidden_dims)
                generator = get_dataset_configuration(dataset)
                for spec, data_params in generator:
                    run(
                        algo_id=algo_id,
                        base_config=base_config,
                        data_params=data_params,
                        dataset=dataset,
                        base_dir=args.base_dir,
                        spec=spec,
                        result_dir=result_dir,
                    )


if __name__ == '__main__':
    # Parse the command line arguments
    import argparse

    parser = argparse.ArgumentParser()

    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=1, type=int)
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)

    # The datasets and algos can be rearranged so that the order at which datasets are trained could be different
    # Also, if you are interested in learning about one dataset, you can can just set the default to be that one dataset
    parser.add_argument('-datasets', default=['YIELD', 'EXCHANGE', 'ECG', 'ARCH', 'STOCKS', 'VAR', 'EIB' ], nargs="+")
    parser.add_argument('-algos', default=['SigCWGAN', 'GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN', ], nargs="+")

    # Algo hyperparameters - you can change these and may achieve better result
    parser.add_argument('-batch_size', default=200, type=int)
    parser.add_argument('-p', default=3, type=int) # the length of past path p
    parser.add_argument('-q', default=3, type=int) # the length of future path q
    parser.add_argument('-hidden_dims', default=3 * (50,), type=tuple) # TODO: the hidden dimension of the generator and the discriminator??? by Bo
    parser.add_argument('-total_steps', default=1000, type=int)
    
    # Parsing command-line arguments
    args = parser.parse_args()
    # Call the main function with the parsed arguments
    main(args)



