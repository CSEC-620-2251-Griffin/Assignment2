
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

seed=0
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
    data_attack = np.load('A2_materials/KDD99/testing_attack.npy')   # shape: [n_attack, 41]
    data_normal = np.load('A2_materials/KDD99/testing_normal.npy')   # shape: [n_normal, 41]
    train_normal = np.load('A2_materials/KDD99/training_normal.npy') # shape: [n_train_norm, 41]


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

if __name__ == "__main__":
    train_pca, attack_pca, normal_pca = load_data(seed=0)
    plot_data(train_pca, attack_pca, normal_pca)
