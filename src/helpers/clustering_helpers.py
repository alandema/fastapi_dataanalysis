from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
import itertools
#test

def find_best_dbscan_params(features,eps_range,min_samples_range):
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