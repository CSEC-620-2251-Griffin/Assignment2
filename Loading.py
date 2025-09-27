import Numpy as np
import sklearn.decomposition as skd
import matplotlib.pyplot as plt

data_attack = np.load('A2_materials/KDD99/testing_attack.npy')
data_normal = np.load('A2_materials/KDD99/testing_normal.npy')
train_normal = np.load('A2_materials/KDD99/training_normal.npy')

seed = 0

attack_pca = skd.PCA(2,random_state=seed)
attack_pca_reduce = attack_pca.transform(data_attack)
normal_pca = skd.PCA(2,random_state=seed)
normal_pca_reduce = normal_pca.transform(data_normal)
train_pca = skd.PCA(2,random_state=seed)
train_pca_reduce = train_pca.fit_transform(train_normal)

plt.figure(figsize=(10,10))
plt.scatter(train_pca_reduce[:,0],train_pca_reduce[:,1],)
