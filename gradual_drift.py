'''
Author: Juhani Kivimäki (juhani.kivimaki.at.helsinki.fi)
Disclaimer: https://github.com/JuhaniK/AC_trials/blob/main/Disclaimer

This script can be used to reproduce the results seen in Section 5.1.2, Figure 3. 
The results might not match exactly due to randomness involved. 
Requires that the data is already created with "data_simulator.py" and exists in folder ".\monitor\data\simulated_data"
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom
from poibin import PoiBin

# Global variables
THRESHOLD = 0                  # What is the minimum confidence for a prediction to be accepted
WINDOWS = [100, 500]           # Monitoring window sizes
DRIFT = np.linspace(0, 1, 21)  # Degree of drift (as the fraction of samples drawn from the drifted distribution)
TRIALS = 1000                  # How many samples to generate
CI = 0.95

nw = len(WINDOWS)
nd = len(DRIFT)

# Initialize random number generator
rng = np.random.default_rng(seed=43)

# Set plotting parameters
plt.rcParams["figure.titlesize"] = 14  
plt.rcParams["axes.titlesize"] = 12  
plt.rcParams["axes.labelsize"] = 10  
plt.rcParams["xtick.labelsize"] = 8  
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 10

# Read data and filter out low confidence simulated data points
df_orig = pd.read_csv(rf".\monitor\data\simulated_data\simulated_0.csv", index_col="id")
df_shift = pd.read_csv(rf".\monitor\data\simulated_data\simulated_1.csv", index_col="id")
filtered_orig = df_orig[df_orig["confidence"] > THRESHOLD]
filtered_shift = df_shift[df_shift["confidence"] > THRESHOLD]
n_accepted_orig = len(filtered_orig.index)
n_accepted_shift = len(filtered_shift.index)

print("Original instances included =", n_accepted_orig)
print(f"Acceptance ratio = {100*n_accepted_orig/len(df_orig.index):.3f}%")
print("Shifted instances included =", n_accepted_shift)
print(f"Acceptance ratio = {100*n_accepted_shift/len(df_shift.index):.3f}%")

# Filter data according to confidence threshold
original_labels = filtered_orig["true_label"].to_numpy()
shifted_labels = filtered_shift["true_label"].to_numpy()
original_confidences = filtered_orig["confidence"].to_numpy()
shifted_confidences = filtered_shift["confidence"].to_numpy()
original_indices = list(range(n_accepted_orig))
shifted_indices = list(range(n_accepted_orig))

# Collect needed statistics
binom_p = np.mean(original_labels)                  # Binomial parameter (Mean accuracy before drift)
atc = np.quantile(original_confidences, 1-binom_p)  # Confidence threshold needed for the ATC method
print("Accuracy =", binom_p)
print("ATC threshold =", atc)

# Create variables for collecting the observed average percentages to compare against theoretical CIs
true_values = np.zeros(nw)
true_values_binomial = np.zeros(nw)

# Also collect some other metrics
deviations = np.zeros((5, nw, nd*TRIALS))
predicted_variances = np.zeros(nw)
largest_errors = np.zeros(nw)

for j, window in enumerate(WINDOWS):
    largest_error = 0
    x_values = list(range(window + 1))

    # Form the percent point function of the Binomial distribution for the number of errors
    binom_ppf = binom.ppf(CI, window, 1 - binom_p)

    for i, drift in enumerate(DRIFT):

        for trial in range(TRIALS):
            # Sample $window$ random indices for this trial
            sample_indices = rng.choice(original_indices, size=window, replace=True)
            splitpoint = int(window*drift)
            if splitpoint == 0:
                labels = original_labels[sample_indices]
                confidences = original_confidences[sample_indices]     
            elif splitpoint == window:
                labels = shifted_labels[sample_indices]
                confidences = shifted_confidences[sample_indices]                
            else:
                drift_sample_indices, orig_sample_indices = np.split(sample_indices, [splitpoint])

                orig_labels = original_labels[orig_sample_indices]
                orig_confidences = original_confidences[orig_sample_indices]

                drift_labels = shifted_labels[drift_sample_indices]
                drift_confidences = shifted_confidences[drift_sample_indices]

                labels = np.concatenate((orig_labels, drift_labels))
                confidences = np.concatenate((orig_confidences, drift_confidences))

            # Switch to error probabilities
            errors = -1 * (labels - 1)
            error_probs = 1 - confidences

            # Form the pmf of the Poisson binomial distribution for number of errors
            pb = PoiBin(error_probs)
            pmf = pb.pmf(x_values)

            # Calculate the expected and true number of errors. Record relative differences between the two
            mean = np.sum(error_probs)
            truth = np.sum(errors)
            atc_estimate = np.sum(confidences < atc) 

            deviations[0, j, i*TRIALS + trial] = drift                                  # The amount of drift
            deviations[1, j, i*TRIALS + trial] = 100 * (mean - truth) / window          # Estimation error relative to window size (AC)
            deviations[2, j, i*TRIALS + trial] = 100 * (1 - binom_p - (truth / window)) # Estimation error relative to window size (binomial)
            deviations[3, j, i*TRIALS + trial] = 100 * (atc_estimate - truth) / window  # Estimation error relative to window size (ATC)

for j, window in enumerate(WINDOWS):
    plt.subplot(1, 2, j + 1)

    b_dist = ["binomial" for _ in deviations[0, j, :]]
    pb_dist = ["AC" for _ in deviations[0, j, :]]
    atc_dist = ["ATC" for _ in deviations[0, j, :]]
    pb_dataf = pd.DataFrame({"ticks": deviations[0, j, :], "devs": deviations[1, j, :], "Distribution": pb_dist})
    b_dataf = pd.DataFrame({"ticks": deviations[0, j, :], "devs": deviations[2, j, :], "Distribution": b_dist})
    atc_dataf = pd.DataFrame({"ticks": deviations[0, j, :], "devs": deviations[3, j, :], "Distribution": atc_dist})
    dataf = pd.concat((b_dataf, pb_dataf, atc_dataf))
    
    ax = sns.lineplot(data=dataf, x="ticks", y="devs", estimator="mean", hue="Distribution", errorbar=("sd", 1))
    ax.set(xlabel="Drift degree", ylabel="Deviation (% of window size)")
    plt.ylim(-10, 10)
    plt.axhline(y=0, color="r")
    plt.title(f"window size: {window}")

plt.suptitle(f"Deviation of estimated accuracy from true accuracy under gradual shift (in {TRIALS} trials)")
plt.show()
