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

def fit(data: np.ndarray, centroids: np.ndarray, max_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    initialize_centroids(data)
    
    for iteration in range(max_iterations):
        # Assign points to clusters
        cluster_assignments = assign_clusters(data, centroids)
        
        # Update centroids
        k = centroids.shape[0]
        new_centroids = update_centroids(k, data, cluster_assignments, centroids)
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break
        
        centroids = new_centroids
    return cluster_assignments, centroids

def predict(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Try to identify the next iteration
    return assign_clusters(data, centroids)

def visualize_clusters(data: np.ndarray, cluster_assignments: np.ndarray, centroids: np.ndarray):
    """Visualize the clustering results"""
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown']
    
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    for i in range(len(np.unique(cluster_assignments))):
        cluster_data = data[cluster_assignments == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i}')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.title('K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()