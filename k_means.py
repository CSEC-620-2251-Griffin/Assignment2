import numpy as np
    
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(k, data: np.ndarray) -> np.ndarray:
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        centroids[i] = data[np.random.choice(n_samples)]
    
    return centroids