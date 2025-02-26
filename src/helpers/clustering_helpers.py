from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


def find_best_dbscan_params(features,eps_range,min_samples_range):
    """
    Find the best parameters for DBSCAN clustering using grid search.

    This function performs a grid search over the specified ranges of eps and
    min_samples parameters for DBSCAN clustering. It evaluates the clustering
    performance using silhouette score and returns the best parameters found.

    Args:
        features (array-like): The input features to cluster.
        eps_range (list): A list of float values to try for the eps parameter.
        min_samples_range (list): A list of integer values to try for the min_samples parameter.

    Returns:
        tuple: A tuple containing:
            - dict: The best parameters found (keys: 'eps', 'min_samples').
            - array: The cluster labels for the best clustering.
            - list: A list of dictionaries containing grid search results.

    Raises:
        ValueError: If features is empty or if eps_range or min_samples_range are empty.
    """
    best_score = -np.inf
    best_params = {'eps': eps_range[0], 'min_samples': min_samples_range[0]}
    best_labels = None
    grid_search_results = []
    
    # Define parameter combinations for grid search
    param_combinations = list(itertools.product(eps_range, min_samples_range))
    
    for eps, min_samples in param_combinations:
        # Train DBSCAN with current parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features)
        
        # Calculate metrics if there are at least 2 clusters (excluding noise)
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        
        metrics_results = {}
        current_score = -np.inf
        
        if n_clusters >= 2:
            # Filter out noise points for metric calculation
            noise_mask = cluster_labels != -1
            features_no_noise = features[noise_mask]
            labels_no_noise = cluster_labels[noise_mask]
            
            # Only calculate metrics if we have enough samples without noise
            if len(features_no_noise) > n_clusters:
                try:
                    sil_score = metrics.silhouette_score(features_no_noise, labels_no_noise)
                    ch_score = metrics.calinski_harabasz_score(features_no_noise, labels_no_noise)
                    db_score = metrics.davies_bouldin_score(features_no_noise, labels_no_noise)
                    
                    
                    metrics_results = {
                        'silhouette_score': float(sil_score),
                        'calinski_harabasz_score': float(ch_score),
                        'davies_bouldin_score': float(db_score)
                    }
                    
                    # Use silhouette score as the primary metric for parameter selection
                    current_score = sil_score
                except Exception as e:
                    metrics_results = {'error': str(e)}
        
        # Track results for this parameter combination
        combo_result = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': int(n_clusters),
            'n_noise_points': int(np.sum(cluster_labels == -1)),
            'metrics': metrics_results
        }
        
        grid_search_results.append(combo_result)
        
        # Update best parameters if we found a better score
        if current_score > best_score:
            best_score = current_score
            best_params = {'eps': eps, 'min_samples': min_samples}
            best_labels = cluster_labels
    
    # If no valid clustering was found (e.g., all noise), use the parameters with the most clusters
    if best_labels is None or np.all(best_labels == -1):
        most_clusters = max(grid_search_results, 
                          key=lambda x: (x['n_clusters'], -x['n_noise_points']))
        best_params = {'eps': most_clusters['eps'], 'min_samples': most_clusters['min_samples']}
        
        # Rerun DBSCAN with the selected parameters
        dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
        best_labels = dbscan.fit_predict(features)
    
    return best_params, grid_search_results, best_labels, best_score


class CustomMFA(BaseEstimator, TransformerMixin):
    """
    A custom transformer that implements a simple Multiple Factor Analysis (MFA) procedure.
    
    The procedure is:
      1. For each group, run PCA and obtain the first eigenvalue (lambda1).
      2. Normalize the group by multiplying by weight = 1/sqrt(lambda1).
      3. Concatenate the normalized groups.
      4. Run a final global PCA on the concatenated data.
    """
    def __init__(self, groups, n_components_global=2, n_components_per_group=None):
        """
        Parameters:
        - groups: list of arrays/lists of column indices for each group.
        - n_components_global: number of global components to extract.
        - n_components_per_group: number of PCA components to use for each group (if None, uses full group dimension).
        """
        self.groups = groups
        self.n_components_global = n_components_global
        self.n_components_per_group = n_components_per_group

    def fit(self, X, y=None):
        """
        Fit the CustomMFA model on the dataset X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data to fit.
        - y: Ignored. Present for compatibility with scikit-learn pipelines.

        Returns:
        - self: object
            Returns the instance itself.
        """
        X = np.asarray(X)
        self.group_weights_ = []
        self.pca_per_group_ = []
        self.group_normalized_data_ = []
        # Process each group separately.
        for group in self.groups:
            X_group = X[:, group]
            n_comp = self.n_components_per_group if self.n_components_per_group is not None else X_group.shape[1]
            pca = PCA(n_components=n_comp)
            pca.fit(X_group)
            # Get the first eigenvalue from PCA (explained variance of the first component)
            lambda1 = pca.explained_variance_[0]
            weight = 1.0 / np.sqrt(lambda1)
            self.group_weights_.append(weight)
            self.pca_per_group_.append(pca)
            # Normalize the group data
            X_group_norm = X_group * weight
            self.group_normalized_data_.append(X_group_norm)
        # Concatenate all normalized groups horizontally
        X_norm_concat = np.hstack(self.group_normalized_data_)
        # Fit a global PCA on the concatenated normalized data
        self.global_pca_ = PCA(n_components=self.n_components_global)
        self.global_pca_.fit(X_norm_concat)
        return self

    def transform(self, X):
        """
        Transform the dataset X using the fitted CustomMFA model.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns:
        - X_transformed: array-like of shape (n_samples, n_components_global)
            The transformed data.
        """
        X = np.asarray(X)
        group_norm = []
        # Normalize each group using the learned weights
        for group, weight in zip(self.groups, self.group_weights_):
            X_group = X[:, group]
            group_norm.append(X_group * weight)
        X_norm_concat = np.hstack(group_norm)
        # Transform using the global PCA
        return self.global_pca_.transform(X_norm_concat)