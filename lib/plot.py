from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from lib.test_metrics import *
from lib.utils import to_numpy


def set_style(ax):
    # Sets the style of the given axes object ax. It hides the right, top, and bottom spines of the plot.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def compare_hists(x_real, x_fake, ax=None, log=False, label=None):
    # compares and plots the histograms of real and fake data. The log parameter controls whether the y-axis should be log-scaled.

    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = 'Historical ' + label
        label_generated = 'Generated ' + label
    else:
        label_historical = 'Historical'
        label_generated = 'Generated'
    bin_edges = ax.hist(x_real.flatten(), bins=80, alpha=0.6, density=True, label=label_historical)[1]
    ax.hist(x_fake.flatten(), bins=bin_edges, alpha=0.6, density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel('log-pdf')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('pdf')
    return ax


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0):
# Computes and plots the autocorrelation function (ACF) of real and fake data. The CI parameter controls 
# whether to plot the confidence interval around the ACF of the fake data.
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[:, i].shape[0]),
                ub[:, i], lb[:, i],
                color='orange',
                alpha=.3
            )
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()
    return ax


def plot_summary(x_fake, x_real, max_lag=None, labels=None):
    # Generates a summary plot for real and fake data. It plots histograms, log-histograms, and ACF for each dimension of the data. 
    # It also computes skewness and kurtosis for each dimension of the data.

    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    from lib.test_metrics import skew_torch, kurtosis_torch
    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 3, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i:i + 1]
        x_fake_i = x_fake[..., i:i + 1]

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 0])

        def text_box(x, height, title):
            textstr = '\n'.join((
                r'%s' % (title,),
                # t'abs_metric=%.2f' % abs_metric
                r'$s=%.2f$' % (skew_torch(x).item(),),
                r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].text(
                0.05, height, textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )

        text_box(x_real_i, 0.95, 'Historical')
        text_box(x_fake_i, 0.70, 'Generated')

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 1], log=True)
        compare_acf(x_real=x_real_i, x_fake=x_fake_i, ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))


def compare_cross_corr(x_real, x_fake):
# Computes and plots the cross-correlation matrices of real and fake data.

    x_real = x_real.reshape(-1, x_real.shape[2])
    x_fake = x_fake.reshape(-1, x_fake.shape[2])
    cc_real = np.corrcoef(to_numpy(x_real).T)
    cc_fake = np.corrcoef(to_numpy(x_fake).T)

    vmin = min(cc_fake.min(), cc_real.min())
    vmax = max(cc_fake.max(), cc_real.max())

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(cc_real, vmin=vmin, vmax=vmax)
    im = axes[1].matshow(cc_fake, vmin=vmin, vmax=vmax)

    axes[0].set_title('Real')
    axes[1].set_title('Generated')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def plot_signature(signature_tensor, alpha=0.2):
    # Plots the signatures of given tensor data.
    plt.plot(to_numpy(signature_tensor).T, alpha=alpha, linestyle='None', marker='o')
    plt.grid()


def savefig(filename, directory):
    # Saves the current figure to the specified directory with the given filename.
    plt.savefig(join(directory, filename))
    plt.close()


# def create_summary(dataset, device, G, lags_past, steps, x_real, one=False):
#     # Generates a summary plot for a given dataset using a generative model G. The function
#     # first generates fake data using the model, and then calls plot_summary to create the
#     # summary plot. The generated fake data is also returned.
#     with torch.no_grad():
#         x_past = x_real[:, :lags_past]
#         if dataset in ['STOCKS', 'ECG']:
#             x_p = x_past.clone().repeat(5, 1, 1)
#         else:
#             x_p = x_past.clone()
#         if one:
#             x_p = x_p[:1]
#         x_fake_future = G.sample(steps, x_p.to(device))
#         plot_summary(x_fake=x_fake_future, x_real=x_real, max_lag=3)
#     return x_fake_future

def create_summary(experiment_directory, dataset, device, G, lags_past, steps, x_real, one=False):
    # Generates a summary plot for a given dataset using a generative model G. The function
    # first generates fake data using the model, and then calls plot_summary to create the
    # summary plot. The generated fake data is also returned.
    with torch.no_grad():
        x_past = x_real[:, :lags_past]
        if dataset in ['STOCKS', 'ECG']:
            x_p = x_past.clone().repeat(5, 1, 1)
        else:
            x_p = x_past.clone()
        if one:
            x_p = x_p[:1]
        x_fake_future = G.sample(steps, x_p.to(device))
        plot_summary(x_fake=x_fake_future, x_real=x_real, max_lag=3)

    import pandas as pd
    import numpy as np

    # Move x_fake_future tensor to CPU
    x_fake_future1 = x_fake_future.cpu().numpy()
    x_real = x_real.cpu().numpy()

    # Reshape x_fake_future to remove the extra dimensions
    x_fake_future1 = x_fake_future1.reshape(x_fake_future.shape[0], -1)

    # Reshape x_fake_future to remove the extra dimensions and transpose
    x_fake_future1 = np.transpose(x_fake_future1, (1, 0)).reshape(-1, 1)

    # Reshape x_real to remove the extra dimensions
    x_real = x_real.reshape(x_real.shape[0], -1)

    # Convert x_fake and x_real to pandas DataFrames
    df_fake = pd.DataFrame(x_fake_future1)
    df_real = pd.DataFrame(x_real)

    # Save DataFrames to Excel file
    #save_path = '/home/tg2885/project_of_EIB'
    save_path = experiment_directory
    df_fake.to_excel(f'{save_path}/1yearyield_only_fake_data.xlsx', index=False)
    df_real.to_excel(f'{save_path}/1yearyield_only_real_data.xlsx', index=False)

    return x_fake_future
