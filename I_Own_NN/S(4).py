import numpy as np
import statsmodels.api as sm

# Generate some example data
np.random.seed(123)
n_obs = 100
n_features = 2
X = np.random.randn(n_obs, n_features)
y = np.random.randn(n_obs)

# Create the state-space model
model = sm.tsa.statespace.MLEModel(
    endog=y,
    exog=X,
    k_states=2,  # Number of states
    k_posdef=1,  # Number of variance-covariance matrices
    initialization='stationary',  # Initialization method
    loglikelihood_burn=5  # Burn-in period for likelihood
)

# Fit the model
results = model.fit(X,y)

# Print summary of the model
print(results.summary())
