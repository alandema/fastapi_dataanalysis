from sklearn.cluster import HDBSCAN
import numpy as np
from sklearn.metrics import silhouette_score


def cluster_dataframe(df, features=None):
    if features is not None:
        df = df[features]

    eps_values = np.linspace(0.1, 1.0, 10)  # could be increased
    min_samples_values = range(3, 10)  # could be increased

    best_score = -1
    best_params = {}

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = HDBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(df, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}

    dbscan = HDBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    dbscan.fit(df)

    cluster_labels = dbscan.labels_

    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = list(dbscan.labels_).count(-1)
    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')

    return cluster_labels
