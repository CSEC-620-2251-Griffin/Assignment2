import numpy as np
import matplotlib as plt
from typing import Tuple
    
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(k, data: np.ndarray) -> np.ndarray:
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))
    
    # Randomly put centroids
    for i in range(k):
        centroids[i] = data[np.random.choice(n_samples)]
    
    return centroids

def assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n_samples = data.shape[0]
    cluster_assignments = np.zeros(n_samples)
    
    # Find closest centroid
    for i, point in enumerate(data):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_assignments[i] = np.argmin(distances)
    
    return cluster_assignments

def update_centroids(k, data: np.ndarray, cluster_assignments: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n_features = data.shape[1]
    new_centroids = np.zeros((k, n_features))
    
    for i in range(k):
        cluster_points = data[cluster_assignments == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # If no points assigned to cluster, keep previous centroid
            new_centroids[i] = centroids[i]
    
    return new_centroids