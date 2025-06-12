#!/usr/bin/env python

"""
This script is adjusted from Matt's (2025) regression and
classification analysis, and instead performs it on available
energy.
"""

import numpy as np
import h5py
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

# fix the seed
np.random.seed(0)


###############################################################################
# Load the dataset
###############################################################################

# fixed
AE_fixed_gradients = np.load("data_processed/fixed/AEs.npy")
Q_fixed_gradients = np.load("data_processed/fixed/Qs.npy")
nfps_fixed_gradients = np.load("data_processed/fixed/nfps.npy")

# varied
AE_random_gradients = np.load("data_processed/random/AEs.npy")
Q_random_gradients = np.load("data_processed/random/Qs.npy")
nfps_random_gradients = np.load("data_processed/random/nfps.npy")


# gradients
a_over_LT = np.load("data_processed/random/w_Ts.npy")
a_over_Ln = np.load("data_processed/random/w_ns.npy")


# tokamasks 
tokamask_fixed = nfps_fixed_gradients == 0 
tokamask_random = nfps_random_gradients == 0

# now split data into stell and tok subsets
# fixed
AE_fixed_tok = AE_fixed_gradients[tokamask_fixed]
AE_fixed_stel = AE_fixed_gradients[~tokamask_fixed]
Q_fixed_tok = Q_fixed_gradients[tokamask_fixed]
Q_fixed_stel = Q_fixed_gradients[~tokamask_fixed]
AE_random_tok = AE_random_gradients[tokamask_random]
AE_random_stel = AE_random_gradients[~tokamask_random]
Q_random_tok = Q_random_gradients[tokamask_random]
Q_random_stel = Q_random_gradients[~tokamask_random]
a_over_LT_tok = a_over_LT[tokamask_random]
a_over_LT_stel = a_over_LT[~tokamask_random]
a_over_Ln_tok = a_over_Ln[tokamask_random]
a_over_Ln_stel = a_over_Ln[~tokamask_random]

# settings of cv
cv = KFold(n_splits=5, shuffle=True)
score_measure='accuracy' #'neg_log_loss' #'accuracy'

###############################################################################
# Evaluate Spearman correlation with the heat flux for fixed gradients
###############################################################################


Q_fixed_stel_new = np.where(Q_fixed_stel < 0.1, 0.0, Q_fixed_stel)
Spearman_correlation = spearmanr(AE_fixed_stel, Q_fixed_stel_new, nan_policy='omit')[0]
print("Spearman correlation with the heat flux for fixed gradients (stellarators): {:.4g}".format(Spearman_correlation))

Q_fixed_tok_new = np.where(Q_fixed_tok < 0.1, 0.0, Q_fixed_tok)
Spearman_correlation = spearmanr(AE_fixed_tok, Q_fixed_tok_new, nan_policy='omit')[0]
print("Spearman correlation with the heat flux for fixed gradients (tokamaks): {:.4g}".format(Spearman_correlation))



###############################################################################
# Evaluate Spearman correlation with the heat flux for random gradients
###############################################################################

Q_random_stel_new = np.where(Q_random_stel < 0.1, 0.0, Q_random_stel)
Spearman_correlation = spearmanr(AE_random_stel, Q_random_stel_new, nan_policy='omit')[0]
print("Spearman correlation with the heat flux for random gradients (stellarators): {:.4g}".format(Spearman_correlation))

Q_random_tok_new = np.where(Q_random_tok < 0.1, 0.0, Q_random_tok)
Spearman_correlation = spearmanr(AE_random_tok, Q_random_tok_new, nan_policy='omit')[0]
print("Spearman correlation with the heat flux for random gradients (tokamaks): {:.4g}".format(Spearman_correlation))


###############################################################################
# XGBoost regression using only AE:
###############################################################################
X_regression = np.column_stack(
    (
        AE_fixed_stel,
    )
)


# Only attempt regression where there is instability:
mask = Q_fixed_stel > 0.1

estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor())
Y_regression = np.log(Q_fixed_stel[mask])
X_regression = X_regression[mask, :]
cv_scores = cross_val_score(estimator, X_regression, Y_regression, cv=cv, verbose=0)
print("R^2 for regression on ln Q for fixed gradients (stellarators): {:.4g}".format(np.mean(cv_scores)))


X_regression = np.column_stack(
    (
        AE_fixed_tok,
    )
)


# Only attempt regression where there is instability:
mask = Q_fixed_tok > 0.1

estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor())
Y_regression = np.log(Q_fixed_tok[mask])
X_regression = X_regression[mask, :]
cv_scores = cross_val_score(estimator, X_regression, Y_regression, cv=cv, verbose=0)
print("R^2 for regression on ln Q for fixed gradients (tokamaks): {:.4g}".format(np.mean(cv_scores)))

###############################################################################
# XGBoost regression using a/LT, a/Ln, and the optimized feature:
###############################################################################

X_regression = np.column_stack(
    (
        a_over_LT_stel,
        a_over_Ln_stel,
        AE_random_stel,
    )
)


# Only attempt regression where there is instability:
mask = Q_random_stel > 0.1

estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor())
Y_regression = np.log(Q_random_stel[mask])
X_regression = X_regression[mask, :]
cv_scores = cross_val_score(estimator, X_regression, Y_regression, cv=cv, verbose=0)
print("R^2 for regression on ln Q for random gradients (stellarators): {:.4g}".format(np.mean(cv_scores)))


X_regression = np.column_stack(
    (
        a_over_LT_tok,
        a_over_Ln_tok,
        AE_random_tok,
    )
)


# Only attempt regression where there is instability:
mask = Q_random_tok > 0.1

estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor())
Y_regression = np.log(Q_random_tok[mask])
X_regression = X_regression[mask, :]
cv_scores = cross_val_score(estimator, X_regression, Y_regression, cv=cv, verbose=0)
print("R^2 for regression on ln Q for random gradients (tokamaks): {:.4g}".format(np.mean(cv_scores)))


# ###############################################################################
# # XGBoost classification (stability vs instability) using AE
# ###############################################################################

mask_stel = Q_fixed_stel > 0.1

X_classification = np.column_stack(
    (
        AE_fixed_stel,
    )
)

estimator = make_pipeline(StandardScaler(), xgb.XGBClassifier())
cv_scores = cross_val_score(
    estimator, X_classification, mask_stel, cv=cv, verbose=0, scoring=score_measure
)
print("Accuracy for fixed-gradient classification (stellarators): {:.4g}".format(np.mean(cv_scores)))

mask_tok = Q_fixed_tok > 0.1

X_classification = np.column_stack(
    (
        AE_fixed_tok,
    )
)

estimator = make_pipeline(StandardScaler(), xgb.XGBClassifier())
cv_scores = cross_val_score(
    estimator, X_classification, mask_tok, cv=cv, verbose=0, scoring=score_measure
)
print("Accuracy for fixed-gradient classification (tokamaks): {:.4g}".format(np.mean(cv_scores)))


# ###############################################################################
# # XGBoost classification (stability vs instability) using a/LT, a/Ln, and AE
# ###############################################################################

mask_stel = Q_random_stel > 0.1

X_classification = np.column_stack(
    (
        a_over_LT_stel,
        a_over_Ln_stel,
        AE_random_stel,
    )
)

estimator = make_pipeline(StandardScaler(), xgb.XGBClassifier())
cv_scores = cross_val_score(
    estimator, X_classification, mask_stel, cv=cv, verbose=0, scoring=score_measure
)
print("Accuracy for random-gradient classification (stellarators): {:.4g}".format(np.mean(cv_scores)))

mask_tok = Q_random_tok > 0.1

X_classification = np.column_stack(
    (
        a_over_LT_tok,
        a_over_Ln_tok,
        AE_random_tok,
    )
)

estimator = make_pipeline(StandardScaler(), xgb.XGBClassifier())
cv_scores = cross_val_score(
    estimator, X_classification, mask_tok, cv=cv, verbose=0, scoring=score_measure
)
print("Accuracy for random-gradient classification (tokamaks): {:.4g}".format(np.mean(cv_scores)))
# ###############################################################################
