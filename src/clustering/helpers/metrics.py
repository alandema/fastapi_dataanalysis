from sklearn import metrics


def get_metrics(labels_true, labels, feature_data):
    return {
        "Homogeneity": metrics.homogeneity_score(labels_true, labels),
        "Completeness": metrics.completeness_score(labels_true, labels),
        "V-measure": metrics.v_measure_score(labels_true, labels),
        "Silhouette Coefficient": metrics.silhouette_score(feature_data, labels),
        "Adjusted Rand Index": metrics.adjusted_rand_score(labels_true, labels),
        "Adjusted Mutual Information": metrics.adjusted_mutual_info_score(labels_true, labels),
        "Silhouette Coefficient": metrics.silhouette_score(feature_data, labels)
    }
