#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load 
data_attack = np.load('/home/t/Downloads/Code/Assignment2/A2_materials/data/testing_attack.npy')   # shape: [n_attack, 41]
data_normal = np.load('/home/t/Downloads/Code/Assignment2/A2_materials/data/testing_normal.npy')   # shape: [n_normal, 41]
train_normal = np.load('/home/t/Downloads/Code/Assignment2/A2_materials/data/training_normal.npy') # shape: [n_train_norm, 41]

seed = 0
rng = np.random.RandomState(seed)

# Scale fit on train_normal only
scaler = StandardScaler().fit(train_normal)
train_normal_s = scaler.transform(train_normal)
data_attack_s  = scaler.transform(data_attack)
data_normal_s  = scaler.transform(data_normal)

# PCA fit on train_normal only
pca = PCA(n_components=2, random_state=seed)
train_pca = pca.fit_transform(train_normal_s)
attack_pca = pca.transform(data_attack_s)
normal_pca = pca.transform(data_normal_s)

# Plot
max_points = 2000
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
