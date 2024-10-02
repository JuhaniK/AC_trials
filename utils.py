"""
Author: Juhani Kivimäki (juhani.kivimaki.at.helsinki.fi)
Disclaimer: https://github.com/JuhaniK/AC_trials/blob/main/Disclaimer

This file contains helper functions for other scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Global variables
RANDOM_SEED = 13
BINS = 20
rng = np.random.default_rng(seed=RANDOM_SEED)


def setstyle():
    """This is used to control the plotting settings"""
    plt.style.use("seaborn-v0_8")

    plt.rc("figure", figsize=(10, 10))
    plt.rc("image", cmap='coolwarm')

    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12) 
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)


def create_dataset(n_easy, n_hard, covariance=1, type='gradient', gamma=np.sqrt(2)):
    """ This is used to create the datasets needed in the experiments as mixtures of Gaussians. 
    Shift amount is controlled by varying the amounts of easy and hard datapoints
    Parameters:
        n_easy (int): Number of easy-to-predict datapoints
        n_hard (int): Number of hard-to-predict datapoints
        covariance (float): A coefficient for controlling the dispersion of the Gaussians
        type (str): One of either 'gradient' or 'circle'
        gamma (float): A label scaling parameter. Bigger values create a steeper decision boundary
    Returns:
        X (NumPy Array): The (two column) feature matrix for all the created datapoints
        y (NumPy Array): The (binary) labels for all the created datapoints
        y_prob (NumPy Array): The probabilities for the positive class used in label generation
    """
    if type == 'gradient':
        # Create data points from a mixture of 6 multivariate Gaussians with diagonal covariance 
        X1_pos = rng.multivariate_normal(mean=(1, -1), cov=covariance*np.eye(2), size=n_hard//2)
        X2_pos = rng.multivariate_normal(mean=(4, 0), cov=covariance*np.eye(2), size=n_easy//4)
        X3_pos = rng.multivariate_normal(mean=(0, -4), cov=covariance*np.eye(2), size=n_easy//4)
        X1_neg = rng.multivariate_normal(mean=(-1, 1), cov=covariance*np.eye(2), size=n_hard//2)
        X2_neg = rng.multivariate_normal(mean=(-4, 0), cov=covariance*np.eye(2), size=n_easy//4)
        X3_neg = rng.multivariate_normal(mean=(0, 4), cov=covariance*np.eye(2), size=n_easy//4)
        
        X = np.concatenate((X1_pos, X2_pos, X3_pos, X1_neg, X2_neg, X3_neg))
    
        # Calculate (signed) distance d from the line y=x for each sampled X
        proj = X[:, 0] - X[:, 1]
        proj_sign = np.sign(proj)
        d = np.sqrt(np.abs(proj)) * proj_sign

        # Map distance d as the probability of positive class using a scaled sigmoid
        y_prob = 1 / (1 + np.exp(-gamma * d))

    elif type == 'circle':
        # Create data points from a mixture of 9 multivariate Gaussians
        X1_pos = rng.multivariate_normal(mean=(0, 0), cov=covariance*2*np.eye(2), size=n_easy//5)
        X2_pos = rng.multivariate_normal(mean=(6, 6), cov=covariance*[[2,-1],[-1,2]], size=n_easy//5)        
        X3_pos = rng.multivariate_normal(mean=(-6, 6), cov=covariance*[[2,1],[1,2]], size=n_easy//5)
        X4_pos = rng.multivariate_normal(mean=(6, -6), cov=covariance*[[2,1],[1,2]], size=n_easy//5)
        X5_pos = rng.multivariate_normal(mean=(-6, -6), cov=covariance*[[2,-1],[-1,2]], size=n_easy//5)
        X1_neg = rng.multivariate_normal(mean=(5, 0), cov=covariance*[[1,0],[0,2]], size=n_hard//4)
        X2_neg = rng.multivariate_normal(mean=(-5, 0), cov=covariance*[[1,0],[0,2]], size=n_hard//4)
        X3_neg = rng.multivariate_normal(mean=(0, 5), cov=covariance*[[2,0],[0,1]], size=n_hard//4)
        X4_neg = rng.multivariate_normal(mean=(0, -5), cov=covariance*[[2,0],[0,1]], size=n_hard//4)
        
        X = np.concatenate((X1_pos, X2_pos, X3_pos, X4_pos, X5_pos, X1_neg, X2_neg, X3_neg, X4_neg))
    
        # Calculate (absolute) distance d from the circle x^2+y^2=5 for each sampled X
        d = np.abs(np.sqrt(X[:, 0]**2 + X[:, 1]**2) - 5)
        
        # Map distance d as the probability of positive class using a scaled sigmoid
        y_prob = np.exp(-gamma * d**2)

    else:
        raise NameError(f"Data scheme '{type}' is not recognized.")

    # Assign a label for each instance by sampling from a Bernoulli distribution
    y = np.array([rng.binomial(1, prob) for prob in y_prob])

    return X, y, y_prob 


def calculate_ese(true_labels, pos_confidences, num_bins=BINS, scheme='equal'):
    """Calculates the calibration error in either in the form of ECE or AdaECE
    Parameters:
        true_labels (NumPy Array): The (binary) labels for all the datapoints used in the calculations
        pos_confidences (NumPy Array): The confidences for the positive class for all the datapoints used in the calculations
        num_bins (int): The number of bins used 
        scheme (srt): Either 'equal' for equiwidth binning (ECE) or 'dynamic' for adaptive binning (AdaECE)
    Returns:
        ece (float): The calibration error     
    """
    # Perform binning
    if scheme == 'equal':
        bins = np.linspace(0.0, 1.0, num_bins + 1)
    elif scheme == 'dynamic':
        borders = np.linspace(0.0, 1.0, num_bins + 1)
        bins = np.array([np.quantile(pos_confidences, q) for q in borders])
        if np.isnan(bins).any():
            print("Revert to equiwidth binning")
            bins = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        raise NameError(f"Binning scheme '{scheme}' is not recognized.")
    bin_indices = np.digitize(pos_confidences, bins, right=True)
    
    # Bin indices should range from 1 to num_bins
    zero_mask = bin_indices[bin_indices == 0]
    zero_amount = zero_mask.sum()
    if zero_amount > 0:
        print(f"{zero_amount} zero indices found. Replace with ones.")
        bin_indices[zero_mask] = 1
        
    # Calculate statistics for each bin
    bin_fraction_of_positives = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(bin_indices == b + 1)[0]
        if len(selected) > 0:
            bin_fraction_of_positives[b] = np.mean(true_labels[selected] == 1)
            bin_confidences[b] = np.mean(pos_confidences[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_fraction_of_positives - bin_confidences)

    # Calculate statistics over all bins
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    return ece


def choose_samples(test_index, size):
    """(Uniformly) randomly chooses a subset of indexes of given size."""
    sampled_indices = rng.choice(test_index, size=size)
    return sampled_indices


def calculate_correlations(rtensor):
    """Used to calculate the Pearson correlation coefficient between calibration error and a set of estimates"""
    correlations = np.zeros(6)
    for i in range(6):       
        if i%2==0:
            # Compare against the ACE for uncalibrated confidence
            cp = pearsonr(rtensor[:,:,1].reshape(-1), np.abs(rtensor[:,:,i+3].reshape(-1)))
            correlations[i] = cp[0]
        else:
            # Compare against the ACE for calibrated confidence
            cp = pearsonr(rtensor[:,:,2].reshape(-1), np.abs(rtensor[:,:,i+3].reshape(-1)))
            correlations[i] = cp[0]
    return correlations

