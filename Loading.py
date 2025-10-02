import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from DBscan import dbscan_core, dbscan_cluster, dbscan_matrix, do_dbscan, tune_dbscan

seed = 0
rng = np.random.RandomState(seed)

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

def load_data(seed=0):
    #This functions loads the data from the A2_materials/KDD99 folder
    #Then scales the data properly
    #Before fitting the traing data to pca, while transforming all of them
    #Input: seed - for reproducibility of PCA itself
    #Output: train_pca, attack_pca, normal_pca - all in PCA space 2D

    # Load the data
    data_attack = np.load('/home/t/Downloads/Code/Assignment2/A2_materials/data/testing_attack.npy')   # shape: [n_attack, 41]
    data_normal = np.load('/home/t/Downloads/Code/Assignment2/A2_materials/data/testing_normal.npy')   # shape: [n_normal, 41]
    train_normal = np.load('/home/t/Downloads/Code/Assignment2/A2_materials/data/training_normal.npy') # shape: [n_train_norm, 41]


    # Scale all of the data presently
    scaler = StandardScaler().fit(train_normal) #Scalar helps!
    train_normal_s = scaler.transform(train_normal)
    data_attack_s  = scaler.transform(data_attack)
    data_normal_s  = scaler.transform(data_normal)

    #We only fit on train_normal as stated in part 2
    #Transform all of them
    pca = PCA(n_components=2, random_state=seed)
    train_pca = pca.fit_transform(train_normal_s) 
    attack_pca = pca.transform(data_attack_s)
    normal_pca = pca.transform(data_normal_s)

    #Return all PCA data
    return train_pca, attack_pca, normal_pca

def plot_data(train_pca, attack_pca, normal_pca):
    # Plotting for Part 2 specifically!
    max_points = 4000
    idx_attack = rng.choice(attack_pca.shape[0], min(max_points, attack_pca.shape[0]), replace=False)
    idx_normal = rng.choice(normal_pca.shape[0], min(max_points, normal_pca.shape[0]), replace=False)

    plt.figure(figsize=(8,8))
    plt.scatter(normal_pca[idx_normal,0], normal_pca[idx_normal,1], s=6, alpha=0.6, label='Test Normal', edgecolors='none')
    plt.scatter(attack_pca[idx_attack,0], attack_pca[idx_attack,1], s=6, alpha=0.6, label='Test Attack', edgecolors='none')

    tn_idx = rng.choice(train_pca.shape[0], min(2000, train_pca.shape[0]), replace=False)
    plt.scatter(train_pca[tn_idx,0], train_pca[tn_idx,1], s=6, alpha=0.3, label='Train Normal', edgecolors='none')

    plt.title('KDD99 – PCA (fit on train_normal only)')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


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


def plot_dbscan_results(best, results, metric='F1'):
    ### I used ai for this because its just plotting the results and i didnt want to write it myself

    """
    Plot a heatmap of a chosen metric from tune_dbscan outputs.
    - best: dict (from tune_dbscan) containing 'eps', 'min_pts' and metrics
    - results: list[dict] (from tune_dbscan), each with 'eps', 'min_pts', and metrics
    - metric: one of keys in results (e.g., 'F1', 'Accuracy', 'TPR', 'FPR', or 'score')

    Returns (fig, ax) so you can further tweak or save.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect unique sorted grids
    eps_vals = sorted({float(r['eps']) for r in results})
    min_vals = sorted({int(r['min_pts']) for r in results})

    # Build metric grid (rows = min_pts, cols = eps)
    Z = np.full((len(min_vals), len(eps_vals)), np.nan, dtype=float)
    for r in results:
        e = float(r['eps']); m = int(r['min_pts'])
        val = r.get(metric, r.get('score', np.nan))
        i = min_vals.index(m)
        j = eps_vals.index(e)
        Z[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, origin='lower', aspect='auto')

    # Ticks/labels
    ax.set_xticks(range(len(eps_vals)))
    ax.set_yticks(range(len(min_vals)))
    ax.set_xticklabels([f"{e:.2f}" for e in eps_vals], rotation=45, ha='right')
    ax.set_yticklabels([str(m) for m in min_vals])
    ax.set_xlabel('eps')
    ax.set_ylabel('min_pts')
    ax.set_title(f'DBSCAN grid — {metric}')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    # Mark best
    be, bm = float(best['eps']), int(best['min_pts'])
    j_best = int(np.argmin(np.abs(np.array(eps_vals) - be)))
    i_best = int(np.argmin(np.abs(np.array(min_vals) - bm)))
    ax.scatter([j_best], [i_best], marker='x', s=100)

    # Small textbox with best metrics
    info = f"best: eps={be:.2f}, min_pts={bm}\n"
    # Try to show common metrics if present
    for k in ('Accuracy', 'TPR', 'FPR', 'F1', 'score'):
        if k in best:
            v = best[k]
            info += f"{k}={v:.3f}\n"
    ax.text(1.02, 0.5, info.rstrip(), transform=ax.transAxes, va='center', fontsize=9)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    train_pca, attack_pca, normal_pca = load_data(seed=0)
    #plot_data(train_pca, attack_pca, normal_pca)
    do_dbscan(train_pca, 0.5, 4)
    eps_list = np.linspace(0.3, 1.5, 9)
    min_pts_list = np.arange(2, 11)
    best_result, all_results = tune_dbscan(train_pca, normal_pca, attack_pca, eps_list, min_pts_list)
    print(best_result)
    plot_dbscan_results(best_result, all_results, metric='F1')
    plot_dbscan_results(best_result, all_results, metric='Accuracy')
    plot_dbscan_results(best_result, all_results, metric='TPR')
    plot_dbscan_results(best_result, all_results, metric='FPR')
    plt.show()

    
