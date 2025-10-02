import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def dbscan_core(X: np.array, eps, min_pts):
    X = np.array(X, dtype=float)
    D = euclidean_distances(X)
    neighbors = (D <= eps).astype(int)
    neighbors_count = neighbors.sum(axis=1)
    core_mask = neighbors_count >= min_pts
    core_points = X[core_mask]
    return core_points, core_mask, neighbors_count

def dbscan_cluster(core_points, X, eps):
    X = np.array(X, dtype=float)
    if core_points.size == 0:
        return np.ones(X.shape[0], dtype=int)
    D = euclidean_distances(X, core_points)
    normal_mask = (D <= eps).any(axis=1)
    return normal_mask

def do_dbscan(train_pca, eps, min_pts):
    core_points, core_mask, neighbors_count = dbscan_core(train_pca, eps, min_pts)
    normal_mask = dbscan_cluster(core_points, train_pca, eps)

    #plot the clusters
    plt.figure(figsize=(8,8))
    plt.scatter(train_pca[normal_mask,0], train_pca[normal_mask,1], s=6, alpha=0.6, label='Normal', edgecolors='none')
    plt.scatter(train_pca[~normal_mask,0], train_pca[~normal_mask,1], s=6, alpha=0.6, label='Attack', edgecolors='none')
    plt.title('DBSCAN Clusters')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def dbscan_matrix(train_pca, normal_pca, attack_pca, eps, min_pts):
    core_points, core_mask, neighbors_count = dbscan_core(train_pca, eps, min_pts)

    X_test = np.vstack((normal_pca, attack_pca))
    normal_mask = np.asarray(dbscan_cluster(core_points, X_test, eps), dtype=bool) 

    n_normal = normal_pca.shape[0]
    n_attack = attack_pca.shape[0]

    tn = int(np.sum(normal_mask[:n_normal]))       
    fp = n_normal - tn                              
    fn = int(np.sum(normal_mask[n_normal:]))        
    tp = n_attack - fn                               

    acc = accuracy(tp, tn, fp, fn)
    tpr_v = tpr(tp, fn)
    fpr_v = fpr(fp, tn)
    f1_v  = f1_score(tp, fp, fn)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": acc, "TPR": tpr_v, "FPR": fpr_v, "F1": f1_v}

