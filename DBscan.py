import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def accuracy(true_pos, true_neg, false_pos, false_neg):
    #Calculates the accuracy given the parameters
    #All parameters are integers
    return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

def tpr(true_pos, false_neg):
    #Calulates the true positive rate given the parameters
    #All parameters are integers
    return true_pos / (true_pos + false_neg)

def fpr(false_pos, true_neg):
    #Calulates the false positive rate given the parameters
    #All parameters are integers
    return false_pos / (false_pos + true_neg)

def f1_score(true_pos, false_pos, false_neg):
    #Calulates the F1 score given the parameters
    #All parameters are integers
    return (2 * true_pos) / (2 * true_pos + false_pos + false_neg)

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

def tune_dbscan(train_pca, normal_pca, attack_pca, eps_values=None, min_pts_values=None, score='f1'):

    def score_val(metrics, which):
        s = which.lower()
        if s == 'f1':
            return metrics['F1']
        if s == 'accuracy':
            return metrics['Accuracy']
        if s == 'tpr':
            return metrics['TPR']
        if s == 'fpr':
            return 1.0 - metrics['FPR']  # lower FPR is better
        raise ValueError("score must be one of: 'f1','accuracy','tpr','fpr'")

    results = []
    best = None

    for eps in eps_values:
        for m in min_pts_values:
            metrics = dbscan_matrix(train_pca, normal_pca, attack_pca, eps=float(eps), min_pts=int(m))
            sv = score_val(metrics, score)
            rec = {'eps': float(eps), 'min_pts': int(m), 'score': sv, **metrics}
            results.append(rec)
            if best is None or sv > best['score']:
                best = rec

    results.sort(key=lambda r: r['score'], reverse=True)
    return best, results
