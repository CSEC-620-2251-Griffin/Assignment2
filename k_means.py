import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
    
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(k: int, data: np.ndarray) -> np.ndarray:
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

def update_centroids(k: int, data: np.ndarray, cluster_assignments: np.ndarray, centroids: np.ndarray) -> np.ndarray:
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

def detect_anomalies(data: np.ndarray, centroids: np.ndarray, cluster_assignments: np.ndarray, threshold: float) -> np.ndarray:
    samples = data.shape[0]
    anomalies = np.zeros(samples, dtype=bool)
    
    for i, point in enumerate(data):
        assigned_cluster = int(cluster_assignments[i])
        distance = euclidean_distance(point, centroids[assigned_cluster])
        anomalies[i] = distance > threshold
    
    return anomalies

def fit(k: int, data: np.ndarray, centroids: np.ndarray, max_iterations: int, threshold: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Remove the initialize_centroids call since centroids are passed as parameter
    
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
    
    # Detect anomalies if threshold is provided
    anomalies = np.zeros(len(data), dtype=bool)
    if threshold is not None:
        anomalies = detect_anomalies(data, centroids, cluster_assignments, threshold)
        print(f"Detected {np.sum(anomalies)} anomalous points")
    
    return cluster_assignments, centroids, anomalies

def predict(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Try to identify the next iteration
    return assign_clusters(data, centroids)

def visualize_clusters(data: np.ndarray, cluster_assignments: np.ndarray, centroids: np.ndarray, anomalies: np.ndarray = None) -> None:
    plt.figure(figsize=(12, 8))
    
    # Get unique clusters and create color map
    unique_clusters = np.unique(cluster_assignments)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot normal data points colored by cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_assignments == cluster
        if anomalies is not None:
            # Only plot non-anomalous points in cluster colors
            normal_mask = cluster_mask & ~anomalies
            cluster_points = data[normal_mask]
        else:
            cluster_points = data[cluster_mask]
        
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], label=f'Cluster {int(cluster)}', alpha=0.7, s=50)
    
    # Plot anomalous points if detected
    if anomalies is not None and np.any(anomalies):
        anomalous_points = data[anomalies]
        plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], 
                   c='orange', marker='^', s=100, label='Anomalies', alpha=0.8, edgecolors='black')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_hyperparameters(k_values: list, threshold_values: list, k_variances: list, t_variances: list) -> None:
    # Plot training results of k
    plt.plot(k_values, k_variances)
    plt.xlabel('Values of k')
    plt.ylabel('Variances')
    plt.title('Training hyperparameter k')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot training results of t
    plt.plot(threshold_values, t_variances)
    plt.xlabel('Values of t')
    plt.ylabel('Variances')
    plt.title('Training hyperparameter t')
    plt.grid(True, alpha=0.3)
    plt.show()    

def intercluster_variance(data: np.ndarray, cluster_assignments: np.ndarray, centroids: np.ndarray) -> float:
    variance = 0.0
    
    for i, point in enumerate(data):
        assigned_cluster = int(cluster_assignments[i])
        variance += euclidean_distance(point, centroids[assigned_cluster]) ** 2.0
    
    return variance

def train_hyperparameters(max_k: int, max_threshold: int, iterations: int, data: np.ndarray) -> None:
    k_values = list()
    threshold_values = list()
    k_variances = list()
    t_variances = list()
    
    # Train k value with fixed threshold
    for k in range(1, max_k):
        centroids = initialize_centroids(k, data=data)
        clusters, centroids, _ = fit(k, data, centroids, iterations, 5.0)
        k_values.append(k)
        k_variances.append(intercluster_variance(data, clusters, centroids))

    # Train distance threshold with fixed k value
    for t in range(1, max_threshold):
        centroids = initialize_centroids(5, data=data)
        clusters, centroids, _ = fit(5, data, centroids, iterations, t)
        threshold_values.append(t)
        t_variances.append(intercluster_variance(data, clusters, centroids))

    plot_hyperparameters(k_values, threshold_values, k_variances, t_variances)

def test():
    import Loading
    
    max_k = 19
    max_threshold = 7
    iterations = 50
    data = Loading.load_data(0)[1]
    train_hyperparameters(max_k, max_threshold, iterations, data)

test()