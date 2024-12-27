import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class WeightedPLSRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        """
        Weighted Partial Least Squares (PLS) Regression.
        X and y are standardized before fitting.

        Parameters
        ----------
        n_components : int
            Number of components for PLS.
        """
        self.n_components = n_components

    def fit(self, X, Y, sample_weight=None):
        """
        Fit the model to the given data.
        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, then each sample is given equal weight.
        """
        
        X = np.array(X)
        Y = np.array(Y)
        Y = np.reshape(Y, (len(Y), 1)) if Y.ndim == 1 else Y

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.array(sample_weight).flatten()
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("Sample weights must have the same length as the number of samples.")

        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Normalize sample weights
        W = np.diag(sample_weight / sample_weight.sum())

        # Weighted means and standard deviations
        self.x_mean_ = np.average(X, axis=0, weights=sample_weight)
        self.x_std_ = np.sqrt(np.average((X - self.x_mean_)**2, axis=0, weights=sample_weight))
        self.y_mean_ = np.average(Y, axis=0, weights=sample_weight)
        self.y_std_ = np.sqrt(np.average((Y - self.y_mean_)**2, axis=0, weights=sample_weight))

        X_scaled = (X - self.x_mean_) / self.x_std_
        Y_scaled = (Y - self.y_mean_) / self.y_std_

        # Initialize storage for PLS components
        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        self.x_rotations_ = np.zeros((n_features, self.n_components))
        self.y_rotations_ = np.zeros((n_targets, self.n_components))

        self.t_list_ = []

        X_residual = X_scaled
        Y_residual = Y_scaled

        for lv in range(self.n_components):
            # Weighted computation of weights
            w_k = np.dot(X_residual.T, W @ Y_residual)[:, 0]
            w_k /= np.linalg.norm(w_k)
            self.x_weights_[:, lv] = w_k

            c_k = np.dot(Y_residual.T, W @ (X_residual @ w_k)).flatten()
            c_k /= np.linalg.norm(c_k)
            self.y_weights_[:, lv] = c_k

            # Weighted scores
            t_k = np.dot(X_residual, w_k)
            u_k = np.dot(Y_residual, c_k)
            self.x_scores_[:, lv] = t_k
            self.y_scores_[:, lv] = u_k

            # Weighted loadings
            p_k = np.dot(X_residual.T, W @ t_k) / np.dot(t_k.T, W @ t_k)
            q_k = np.dot(Y_residual.T, W @ t_k) / np.dot(t_k.T, W @ t_k)
            self.x_loadings_[:, lv] = p_k
            self.y_loadings_[:, lv] = q_k

            self.x_rotations_[:, lv] = w_k
            self.y_rotations_[:, lv] = c_k

            # Update residuals
            X_residual -= np.outer(t_k, p_k)
            Y_residual -= np.outer(t_k, q_k)

            self.t_list_.append(w_k)

        X_weights = self.x_weights_
        X_loadings = self.x_loadings_
        Y_loadings = self.y_loadings_

        # Calculate coefficients and intercept
        inv_mat = np.linalg.inv(np.dot(X_loadings.T, X_weights))
        B = np.dot(X_weights, inv_mat)  
        B = np.dot(B, Y_loadings.T)   
        self.coef_ = B * (self.y_std_ / self.x_std_).reshape(-1, 1) 
        self.intercept_ = self.y_mean_

        return self

    def predict(self, X):

        X = np.array(X)
        X -= self.x_mean_
        Y_pred = np.dot(X, self.coef_) + self.intercept_

        return Y_pred

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"n_components": self.n_components}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self