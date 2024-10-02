"""
Author: Juhani Kivimäki (juhani.kivimaki.at.helsinki.fi)
Disclaimer: https://github.com/JuhaniK/AC_trials/blob/main/Disclaimer

This script can be used to reproduce the results seen in Section 5.1.1, Figure 1. 
The results might not match exactly due to randomness involved. 

It creates a simulated test set with confidences drawn from a mixture of Beta distributions
and labels assigned randomly according to confidence. This creates a dataset, which is perfectly 
calibrated at the limit of the number of data points.

The created datasets are stored in the folder ".\monitor\data\simulated_data"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOTAL = 20000
N = np.array([[18000, 400, 1600],    # The weights of the components
              [16000, 1000, 3000]])  
ALPHA = [20, 1, 2]
BETA = [1, 20, 2]

rng = np.random.default_rng(seed=43)

plt.style.use("seaborn-v0_8")

# Uncomment lines below for poster style graph
plt.rcParams["figure.titlesize"] = 20  
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["axes.labelsize"] = 12  
plt.rcParams["xtick.labelsize"] = 10 
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

confs = np.zeros(TOTAL)

for j in range(N.shape[0]):
    cum_N = np.cumsum(N[j, :])

    # Generate confidence scores according to mixture weights
    for i in range(N.shape[1]):
        plt.subplot(1, 2, j + 1)
        if i == 0:
            confs[: cum_N[i]] = rng.beta(ALPHA[i], BETA[i], N[j, i])
        else:
            confs[cum_N[i - 1] : cum_N[i]] = rng.beta(ALPHA[i], BETA[i], N[j, i])

    plt.hist(confs, bins=50, density=True)
    if j == 0:
        plt.title("Original data")
    else:
        plt.title(f"Shifted data")
    plt.xlabel("confidence")
    plt.ylabel("density")
    plt.ylim([0, 16])

    # Generate labels according to confidence scores generated
    rands = rng.uniform(size=TOTAL)
    labels = np.array([1 if rands[i] < confs[i] else 0 for i in range(TOTAL)])
    print(np.mean(labels))

    df = pd.DataFrame(
        {"id": list(range(TOTAL)), "confidence": confs, "true_label": labels}
    )
    path = rf".\monitor\data\simulated_data\simulated_{j}.csv"
    df.to_csv(path, index_label='id')
    print(
        f"Fraction of positivies for shift degree {j} is {df['true_label'].sum()/TOTAL}"
    )

plt.suptitle("Simulated datasets")
plt.tight_layout()
plt.show()
