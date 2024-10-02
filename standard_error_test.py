"""
Author: Juhani Kivimäki (juhani.kivimaki.at.helsinki.fi)
Disclaimer: https://github.com/JuhaniK/AC_trials/blob/main/Disclaimer

This script can be used to reproduce the results seen in Section 5.1.3, Figure 4.
Additionally, one can set the global variable 'PLOT_SAMPLES' to True, to reproduce Figure 1 from Section 4.2.
The results might not match exactly due to randomness involved. 
Requires that the data is already created with "data_simulator.py" and exists in folder ".\monitor\data\simulated_data"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bisect import bisect
from poibin import PoiBin

THRESHOLD = 0                                       # What is the minimum confidence for a prediction to be accepted
CI = 0.95                                           # Confidence interval for the estimated error rate
WINDOWS = np.linspace(100, 500, num=5, dtype=int)   # Monitoring window sizes
TRIALS = 10000                                      # How many trials to run for each window size
PLOT_SAMPLES = False                                # Whether to plot samples of generated Poisson binomial distributions (Figure 1).

# Initialize random number generator
rng = np.random.default_rng(seed=43)

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams["figure.titlesize"] = 16  
plt.rcParams["axes.titlesize"] = 14  
plt.rcParams["axes.labelsize"] = 12  
plt.rcParams["xtick.labelsize"] = 12  
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

nw = len(WINDOWS)
margin = (1-CI)/2


for i in range(2):
    # Read data and filter out low confidence simulated data points
    print("Shift amount:", i)
    df = pd.read_csv(rf".\monitor\data\simulated_data\simulated_{i}.csv", index_col="id")
    filtered = df[df["confidence"] > THRESHOLD]
    n_accepted = len(filtered.index)
    population_accuracy = filtered['true_label'].sum()/n_accepted
    print(f"\tAcceptance ratio = {100*n_accepted/len(df.index):.3f}%")
    print(f"\tFraction of positivies for shift degree {i} is {population_accuracy}")
    print("\tLargest deviation with")

    # Filter data according to confidence threshold
    all_labels = filtered["true_label"].to_numpy()
    all_confidences = filtered["confidence"].to_numpy()
    all_indices = list(range(n_accepted))

    # Binomial parameter and estimated standard error of mean (MSE)
    if i == 0:
        MSEs = np.zeros(nw)
        universal_std = np.std(all_labels)
        
    # Collect observed average percentages to compare against theoretical CIs
    true_values_cdf = np.zeros(nw)
    true_values_batch = np.zeros(nw)
    true_values_universal = np.zeros(nw)

    for k, window in enumerate(WINDOWS):
        hits_cdf = 0
        hits_batch = 0
        hits_universal = 0
        largest_deviation = 0
        denominator = np.sqrt(window)

        for trial in range(TRIALS):
            # Sample $window$ random indices for this trial
            sample_indices = rng.choice(all_indices, size=window, replace=True)
            labels = all_labels[sample_indices]
            confidences = all_confidences[sample_indices]
            expected_mean_accuracy = np.mean(confidences)

            # Form the pmf of the Poisson-Binomial distribution for number of correct predictions
            pb = PoiBin(confidences)
            x_values = list(range(window + 1))
            pmf = pb.pmf(x_values)

            # Form the cdf of the Poisson binomial distribution and calculate the confidence interval bound
            cdf = pb.cdf(x_values)
            cdf_lower_bound = bisect(cdf, margin) / window
            cdf_upper_bound = bisect(cdf, CI + margin) / window

            # Estimate standard error with reference data / sample confidences
            batch_mse = np.std(confidences) / denominator
            universal_mse = universal_std / denominator
            batch_lower_bound = expected_mean_accuracy - 1.96 * batch_mse
            batch_upper_bound = expected_mean_accuracy + 1.96 * batch_mse
            universal_lower_bound = expected_mean_accuracy - 1.96 * universal_mse
            universal_upper_bound = expected_mean_accuracy + 1.96 * universal_mse

            # Check if the accuracy falls within the predicted CI. Also keep track of the largest error.
            true_accuracy = np.mean(labels)
            if true_accuracy > cdf_lower_bound and true_accuracy <= cdf_upper_bound:
                hits_cdf += 1
            if true_accuracy > batch_lower_bound and true_accuracy <= batch_upper_bound:
                hits_batch += 1
            if true_accuracy > universal_lower_bound and true_accuracy <= universal_upper_bound:
                hits_universal += 1
            deviation = true_accuracy - expected_mean_accuracy
            if deviation > largest_deviation:
                largest_deviation = deviation

            # This plots 6 samples from the process:
            if PLOT_SAMPLES is True:
                if i==0 and window == 500 and trial<6:
                    n_preds = window
                    bin_pos = np.array(x_values)/n_preds
                    pa = population_accuracy
                    ta = true_accuracy
                    pe = expected_mean_accuracy
                    lb = cdf_lower_bound
                    ub = cdf_upper_bound
        
                    plt.subplot(2, 3, trial+1)
                    plt.hist(bin_pos, bins=n_preds, weights=pmf, color="#0099CC", alpha=0.5, label = 'Poisson binomial distribution')
                    plt.axvline(x=pa, color='m', linestyle='dashed' , label = f'expected accuracy = {100*pa:.1f}%')
                    plt.axvline(x=ta, color='g', linestyle='dashed' , label = f'batch accuracy = {100*ta:.1f}%')
                    plt.axvline(x=pe, color='b', linestyle='dashed' , label = f'point estimate = {100*pe:.1f}%')
                    plt.axvline(x=lb, color='r', linestyle='dashed', label = f'95% conf. lower bound = {100*lb:.1f}%')
                    plt.axvline(x=ub, color='r', linestyle='dashed', label = f'95% conf. upper bound = {100*ub:.1f}%')
                    plt.xlim([lb-0.025, ub+0.08])
                    plt.xlabel("Predictive accuracy")
                    plt.ylabel("Probability")
                    plt.title(f"Trial {trial+1}")
                    plt.legend()           

        true_values_cdf[k] = 100 * hits_cdf / TRIALS
        true_values_batch[k] = 100 * hits_batch / TRIALS
        true_values_universal[k] = 100 * hits_universal / TRIALS

        # Print largest deviation:
        print(f"\t\twindow size {window} is {100*largest_deviation:.2f} % units.")

        if PLOT_SAMPLES is True and i==0 and window==500:
            plt.suptitle("Distribution of estimated accuracy for 6 batches of 500 predictions")
            plt.show()
            plt.close()
    
    plt.subplot(1, 2, i + 1)
    plt.plot(WINDOWS, true_values_cdf, label="Poisson binomial")
    plt.plot(WINDOWS, true_values_universal, label="SEM")
    plt.axhline(y=100 * CI, color="black", label="target conf. bound")
    if i == 0:
        plt.title("Original data")
    else:
        plt.title(f"Shifted data")
    plt.xlabel("Monitoring window size")
    plt.ylabel("% of trials")
    plt.xticks(WINDOWS)
    plt.ylim(90, 100)
    plt.legend(loc='lower left')

plt.suptitle(f"Percentage of trials with estimated accuracy within the predicted {100*CI}% Confidence Interval")
plt.show()