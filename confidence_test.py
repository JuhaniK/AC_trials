"""
Author: Juhani Kivimäki (juhani.kivimaki.at.helsinki.fi)
Disclaimer: https://github.com/JuhaniK/AC_trials/blob/main/Disclaimer

This script can be used to reproduce the results seen in Section 5.2 and in Appendix A.
The results might not match exactly due to randomness involved. 
"""

import os
from utils import calculate_ese, choose_samples, create_dataset, setstyle, calculate_correlations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from betacal import BetaCalibration

# Global variables
PATH = os.getcwd()
TYPE = 'gradient'  # Which of the two datasets to create. Either 'gradient' or 'circle'
COV = 1            # The covariance coefficient used in data creation
TRIALS = 1000      # Number of times the estimate is calculated for each monitoring window
WINDOW = 500       # Sample size of each monitoring window
CMAP = 'isotonic'  # Calibration mapping used. Either 'isotonic' or 'beta'
ECE = 'dynamic'    # Set to 'equal' for equiwidth binning (ECE) or 'dynamic' for equisize (adaptive, ACE) binning 
BINS = 20          # The number of bins to use when calculating the calibration error

# Whether to create plots:
PLOT_DATA_DISTRIBUTIONS = True
PLOT_DECISION_BOUNDARY = True

# Set the label scaling parameter gamma:
if TYPE == 'gradient':
    GAMMA = np.sqrt(2)  
else:
    GAMMA = np.log(np.sqrt(2))  


def draw_true_decisicion_boundary(gamma=GAMMA, type=TYPE):
    """ Draws the true decision boundary (where p(y|x) = 0.5) as a dashed line on top of existing ax.
    Parameters
        gamma (float): A hyperparameter controlling the steepness of p(y|x) at the decision boundary 
        type (str): One of either 'gradient' or 'circle'
    Returns
        None
    """
    if type == 'gradient':
        plt.axline((0,0), (1,1), c='black', alpha=0.8, ls='--', label="True decision boundary")
    elif type =='circle':
        angle = np.linspace( 0 , 2 * np.pi , 150 ) 
        delta = np.sqrt(-np.log(0.5) / gamma) 
        r1 = 5 - delta
        r2 = 5 + delta

        x1 = r1 * np.cos( angle ) 
        y1 = r1 * np.sin( angle )       
        x2 = r2 * np.cos( angle ) 
        y2 = r2 * np.sin( angle )       

        plt.plot(x1, y1, c='black', alpha=0.8, ls='--', label="True decision boundary")
        plt.plot(x2, y2, c='black', alpha=0.8, ls='--')
    else:
        raise NameError(f"Data scheme '{type}' is not recognized.")


# Generate training data
X_train, y_train, y_prob_train = create_dataset(80000, 20000, covariance=1, type=TYPE, gamma=GAMMA)
X_valid, y_valid, y_prob_valid = create_dataset(20000, 5000, covariance=1, type=TYPE, gamma=GAMMA)
X_test, y_test, y_prob_test = create_dataset(20000, 5000, covariance=1, type=TYPE, gamma=GAMMA)

# Control the amount of easy and hard to predict samples respectively
test_portions = [
    (20000, 5000),  # This division contains no covariate shift
    (15000, 10000),  # The rest represent shifted distributions with increasing portion of hard samples
    (12500, 12500),
    (10000, 15000)
]

# Initialize models
models = [GaussianNB(), 
          LogisticRegression(), 
          KNeighborsClassifier(),
          SVC(probability=True), 
          RandomForestClassifier(),
          XGBClassifier(verbosity=0, use_label_encoder=False), 
          LGBMClassifier(verbosity=-1)]
model_names =["Naive Bayes", "Logistic Regression", "K-nearest neighbors", "Support Vector Machine", "Random Forest", "XGBoost", "LightGBM"]
model_abbs =["NB", "LR", "KNN", "SVM", "RF", "XGB", "LGBM"]
n_models = len(models)

# Containers for data used in calculations
calibration_mappings = []
calibs = np.zeros(n_models)
og_accs = np.zeros(n_models)
uc_og_confs = np.zeros(n_models)
c_og_confs = np.zeros(n_models)
uatc_thresholds = np.zeros(n_models)
catc_thresholds = np.zeros(n_models)

# Containers for temporary data storage for the repeated trials
sample_accs = np.zeros(TRIALS)
sample_uconfs = np.zeros(TRIALS)
sample_cconfs = np.zeros(TRIALS)
sample_udocs = np.zeros(TRIALS)
sample_cdocs = np.zeros(TRIALS)
sample_uact = np.zeros(TRIALS)
sample_cact = np.zeros(TRIALS)

# Containers for final results
bayes_tensor = np.zeros((len(test_portions), 2))
results_tensor = np.zeros((len(test_portions), len(models), 9, 2))

# Set plotting style:
setstyle()

# To draw the decicion boundaries for each classifier
if PLOT_DECISION_BOUNDARY is True:
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    # Set color map
    cm = plt.cm.bwr
    cm_bright = ListedColormap(["#0000FF", "#FF0000"])
    ax = plt.subplot(2, 4, 1)
    
    # Plot the data points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.03, edgecolors="k")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Input data")
    draw_true_decisicion_boundary()

# Train models and calibration mappings:
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    og_accs[i] = model.score(X_test, y_test)
    
    # Calculate (uncalibrated) probabilities for the positive class
    p_valid = model.predict_proba(X_valid)        
    p_test = model.predict_proba(X_test)
    probas_valid = p_valid[:, 1]
    probas_test = p_test[:, 1]

    # Store confidence as the maximum of predicted probabilities for each sample
    uc_confs = np.max(p_test, axis=1)
    uc_og_confs[i] = np.mean(uc_confs)  
    print(f"\nFor model {model_names[i]}")
    print(f"- Uncalibrated AC is {uc_og_confs[i]}")

    # Train a calibration mapping
    if CMAP == 'isotonic':
        c_map = IsotonicRegression(out_of_bounds='clip')
    elif CMAP == 'beta':
        c_map = BetaCalibration(parameters='abm')
    else:
        print("Unknown calibration mapping suggested. Reverting to Isotonic Regression.")     
        c_map = IsotonicRegression(out_of_bounds='clip')

    try:
        c_map.fit(probas_valid, y_valid)
    except:
        c_map.fit(probas_valid.reshape(-1, 1), y_valid)
    calibration_mappings.append(c_map)        

    # Calculate AC, and ATC_u and ATC_c thresholds (based on error rate and maximum confidence) for the original data set.
    calib = c_map.predict(probas_test)
    c_confs = [x if x>0.5 else 1-x for x in calib]
    c_og_confs[i] = np.mean(c_confs)
    print(f"- Calibrated AC is {c_og_confs[i]}")
    uatc_thresholds[i] = np.quantile(uc_confs, 1-og_accs[i])
    print(f"- ATC_u threshold is {uatc_thresholds[i]}")
    catc_thresholds[i] = np.quantile(c_confs, 1-og_accs[i])
    print(f"- ATC_c threshold is {catc_thresholds[i]}")

    # To draw the decicion boundaries for each classifier
    if PLOT_DECISION_BOUNDARY is True:
        ax = plt.subplot(2, 4, i+2)
        DecisionBoundaryDisplay.from_estimator(model, 
                                               X_train,
                                               grid_resolution=1000,
                                               plot_method="pcolormesh",
                                               cmap=cm,
                                               alpha=0.8,
                                               ax=ax, eps=0.5
                                              )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())                
        ax.set_title(model_names[i])
        draw_true_decisicion_boundary()

if PLOT_DECISION_BOUNDARY is True:
    # Add colorbar and supertitle
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes((0.85, 0.1, 0.02, 0.78))
    plt.title(r"$p(y=1|\boldsymbol{x})$", pad=20.0)
    plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cm), cax=cax, location="right", fraction=0.05, ticks=np.linspace(0.0, 1.0, 11))
    plt.suptitle("Decision boundaries for the models used.")
    plt.show()

print()
# Loop through all degrees of shift
for j, portion in enumerate(test_portions):
    n, m = portion
    X_test, y_test, y_prob_test = create_dataset(n, m, covariance=COV, type=TYPE, gamma=GAMMA)
    print(f"The fraction of positive labels with portions {portion} is {np.mean(y_test)}")
    
    # Plot a visualization of the data distributions during shift
    if PLOT_DATA_DISTRIBUTIONS is True:
        positive = y_test==1
        plt.subplot(1, len(test_portions), j+1)
        draw_true_decisicion_boundary() 
        plt.scatter(x=X_test[positive[0], 0], y=X_test[positive[0], 1], c='r', alpha=0.5, label="1")
        plt.scatter(x=X_test[~positive[0], 0], y=X_test[~positive[0], 1], c='b', alpha=0.5, label="0")
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.03, edgecolors="k")     

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(loc="upper left")
        if TYPE == 'gradient':
            plt.scatter([1, -1], [-1, 1], c='k', marker='*', label='hard')
            plt.scatter([0, -4, 0, 4], [4, 0, -4, 0], c='k', marker='s', label='easy')
        else:
            plt.scatter([0, 0, -5, 5], [5, -5, 0, 0], c='k', marker='*', label='hard')
            plt.scatter([0, 6, 6, -6, -6], [0, 6, -6, 6, -6], c='k', marker='s', label='easy')
        if j == 0:
            plt.title(f"Easy={n}, Hard={m} (no shift)")
        else:
            plt.title(f"Easy={n}, Hard={m}")
    
    # Calculate the accuracy and calibration error of the Bayes optimal classifier
    if TYPE == 'gradient':
        y_hat_bayes = (X_test[:,0] > X_test[:,1]).astype(int)
    elif TYPE == 'circle':
        distances = np.linalg.norm(X_test, axis=1)
        delta = np.sqrt(-np.log(0.5) / GAMMA) 
        r1 = 5 - delta
        r2 = 5 + delta
        y_hat_bayes = ((distances > r1) & (distances < r2))
    else:
        raise NameError(f"Data scheme '{type}' is not recognized.")
    bayes_tensor[j, 0] = np.mean(y_hat_bayes == y_test)
    bayes_tensor[j, 1] = calculate_ese(y_test, y_prob_test)

    # Calculate calibration error for each model        
    for i, model in enumerate(models): 
        c_map = calibration_mappings[i]

        # Create probabilistic predictions 
        p_test = model.predict_proba(X_test)
        probas_test = p_test[:, 1]

        # Use probability of the positive class to calculate adaECE
        ece_uncalib = calculate_ese(y_test, probas_test, scheme=ECE)
        calib = c_map.predict(probas_test)
        ece_calib = calculate_ese(y_test, calib, scheme=ECE)

        # Run $TRIALS trials and estimate accuracy using each of the 6 methods
        test_index = range(len(y_test))
        for k in range(TRIALS):
            sampled_indices = choose_samples(test_index, size=WINDOW)
            sample_X = X_test[sampled_indices]
            sample_y = y_test[sampled_indices]

            p_test = model.predict_proba(sample_X)
            probas_test = p_test[:, 1]
            calib = c_map.predict(probas_test)
            
            uconfs = np.max(p_test, axis=1)
            cconfs = [x if x>0.5 else 1-x for x in calib]

            # Calculate sample accuracy and error of each estimate (deviation from true sample accuracy) for each trial k
            sample_acc = model.score(sample_X, sample_y)
            sample_accs[k] = sample_acc
            sample_uconfs[k] = np.mean(uconfs) - sample_acc
            sample_cconfs[k] = np.mean(cconfs) - sample_acc
            sample_udocs[k] = og_accs[i] + sample_uconfs[k] - uc_og_confs[i]
            sample_cdocs[k] = og_accs[i] + sample_cconfs[k] - c_og_confs[i]
            sample_uact[k] = np.mean(uconfs > uatc_thresholds[i]) - sample_acc
            sample_cact[k] = np.mean(cconfs > catc_thresholds[i]) - sample_acc

        # Collect the true accuracy and calibration errors (for the whole dataset) 
        acc = model.score(X_test, y_test)
        results_tensor[j,i,0,0] = acc        
        results_tensor[j,i,1,0] = ece_uncalib
        results_tensor[j,i,2,0] = ece_calib

        # Record the average the estimation error results over all trials 
        results_tensor[j,i,3,0] = np.mean(sample_uconfs) 
        results_tensor[j,i,4,0] = np.mean(sample_cconfs) 
        results_tensor[j,i,5,0] = np.mean(sample_udocs) 
        results_tensor[j,i,6,0] = np.mean(sample_cdocs) 
        results_tensor[j,i,7,0] = np.mean(sample_uact)
        results_tensor[j,i,8,0] = np.mean(sample_cact)
        
        # Also record the standard error of the estimation error means
        results_tensor[j,i,0,1] = 2*np.std(sample_accs)
        results_tensor[j,i,3,1] = 2*np.std(sample_uconfs)
        results_tensor[j,i,4,1] = 2*np.std(sample_cconfs)
        results_tensor[j,i,5,1] = 2*np.std(sample_udocs)
        results_tensor[j,i,6,1] = 2*np.std(sample_cdocs)
        results_tensor[j,i,7,1] = 2*np.std(sample_uact)
        results_tensor[j,i,8,1] = 2*np.std(sample_cact)        
                
if PLOT_DATA_DISTRIBUTIONS is True:
    plt.suptitle("Visualization of covariate shift within the experiment.")
    plt.show()

# Collect all results into dataframes and persist them
cols = ["Accuracy", "ACE$_u$", "ACE$_c$", "AC$_u$", "AC$_c$", "DOC-Feat$_u$", "DOC-Feat$_c$", "ATC$_u$", "ATC$_c$"]
datapath = os.path.join(PATH, 'data')

correlations = calculate_correlations(results_tensor[:,:,:,0])  
print("\nThe Pearson correlation coefficients with calibration error are:")  
for i in range(6):
    print(f"{cols[i+3]}: {correlations[i]}")

for i, portion in enumerate(test_portions):        
    # print("Results (mean) for the model", name)
    # means_df = pd.DataFrame(results_tensor[:,:,i,0], columns=cols)
    print("\nResults (mean) for portion", portion)
    means_reshaped = results_tensor[i,:,:,0]
    means_df = pd.DataFrame(means_reshaped, columns=cols, index=model_names)
    print(means_df)
    
    stds_reshaped = results_tensor[i,:,:,1]

    all_data = np.zeros(means_reshaped.shape).astype(object)
    for n in range(all_data.shape[0]):
        for m in range(all_data.shape[1]):
            if m==1 or  m==2:
                all_data[n, m] = f"{100*means_reshaped[n,m]:.1f}\%"
            else:
                all_data[n, m] = f"{100*means_reshaped[n,m]:.1f}\%$\pm${100*stds_reshaped[n,m]:.1f}\%"
    data_df = pd.DataFrame(all_data, columns=cols, index=model_abbs)

    filepath = os.path.join(datapath, f"{TYPE}_shift_{i}.csv")
    data_df.to_csv(filepath)

    if i == 0:
        all_data_df = data_df.copy()
    else:
        all_data_df = pd.concat((all_data_df, data_df))

all_data_df.reset_index(inplace=True)
all_data_df.rename(columns={'index': 'Model'}, inplace=True)
filepath = os.path.join(datapath, f"{TYPE}_shift_all.csv")
all_data_df.to_csv(filepath)

print("\nResults for the Bayes optimal classifier")
bayes_data = np.zeros(bayes_tensor.shape).astype(object)
for n in range(bayes_data.shape[0]):
    for m in range(bayes_data.shape[1]):
        bayes_data[n, m] = f"{100*bayes_tensor[n,m]:.1f}\%"        
bayes_df = pd.DataFrame(bayes_data, columns=['Accuracy', 'ACE'])
print(bayes_df)
filepath = os.path.join(datapath, f"bayes_{TYPE}.csv")
bayes_df.to_csv(filepath)
