#!/usr/bin/env python

"""
This file shows how to do some basic regression and classification tasks
with the dataset of nonlinear gyrokinetic simulations in
Landreman et al, (2025), "How does ion temperature gradient turbulence
depend on magnetic geometry? Insights from data and machine learning"
"""

import numpy as np
import h5py
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

###############################################################################
# Load the dataset
###############################################################################

filename = "20250102-01_GX_stellarator_dataset.h5"

with h5py.File(filename, "r") as f:
    raw_feature_tensor = f["/raw_feature_tensor"][()]
    a_over_LT = f["/varied_gradient_simulations/a_over_LT"][()]
    a_over_Ln = f["/varied_gradient_simulations/a_over_Ln"][()]
    Q_varied_gradients = f["/varied_gradient_simulations/Q_avgs"][()]
    Q_fixed_gradients = f["/fixed_gradient_simulations/Q_avgs"][()]
    z_functions_GX = f["/z_functions_GX"][()]

###############################################################################
# Make some aliases for a few of the raw geometric features
###############################################################################

index = 0
assert z_functions_GX[index] == b"bmag"
bmag = raw_feature_tensor[:, :, index]

index = 1
assert z_functions_GX[index] == b"gbdrift"
gbdrift = raw_feature_tensor[:, :, index]

index = 2
assert z_functions_GX[index] == b"cvdrift"
cvdrift = raw_feature_tensor[:, :, index]

index = 6
assert z_functions_GX[index] == b"gds22_over_shat_squared"
grad_x_squared = raw_feature_tensor[:, :, index]

###############################################################################
# Compute the optimized features from the paper
###############################################################################

best_regression_feature = np.mean(
    (np.heaviside(cvdrift, 0.0) + 0.2) * grad_x_squared**1.5 / bmag, axis=1
)

best_classification_feature = np.mean(
    (np.heaviside(gbdrift, 0.0) + 0.4) * np.sqrt(grad_x_squared / bmag), axis=1
)

###############################################################################
# Evaluate Spearman correlation with the heat flux for fixed gradients
###############################################################################

Spearman_correlation = spearmanr(Q_fixed_gradients, best_regression_feature)[0]
print(
    "Spearman correlation with the heat flux for fixed gradients:", Spearman_correlation
)

###############################################################################
# XGBoost regression using a/LT, a/Ln, and the optimized feature:
###############################################################################

cv = KFold(n_splits=5, shuffle=True)

X_regression = np.column_stack(
    (
        a_over_LT,
        a_over_Ln,
        best_regression_feature,
    )
)

# Only attempt regression where there is instability:
mask = Q_varied_gradients > 0.1

estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor())
Y_regression = np.log(Q_varied_gradients[mask])
X_regression = X_regression[mask, :]
cv_scores = cross_val_score(estimator, X_regression, Y_regression, cv=cv, verbose=2)
print("R^2 for regression on ln Q for varied gradients:", np.mean(cv_scores))

###############################################################################
# XGBoost classification (stability vs instability) using a/LT, a/Ln, and the optimized feature:
###############################################################################

X_classification = np.column_stack(
    (
        a_over_LT,
        a_over_Ln,
        best_classification_feature,
    )
)

estimator = make_pipeline(StandardScaler(), xgb.XGBClassifier())
cv_scores = cross_val_score(
    estimator, X_classification, mask, cv=cv, verbose=2, scoring="accuracy"
)
print("Accuracy for classification (stability vs instability):", np.mean(cv_scores))
