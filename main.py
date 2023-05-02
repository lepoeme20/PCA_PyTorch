import torch
import numpy as np

from pca import TorchPCA
from sklearn.decomposition import PCA

# 1. create a data
data = np.random.randn(20, 10)

# 2. sklearn PCA
sk_pca = PCA(n_components=4)
sk_pca.fit(data)
sk_proj = sk_pca.transform(data)
sk_reconst = sk_pca.inverse_transform(sk_proj)

# 3. custom torch PCA
torch.set_printoptions(precision=8)
data_tensor = torch.from_numpy(data)
torch_pca = TorchPCA(n_components=4)
torch_pca.fit(data_tensor)
torch_proj = torch_pca.transform(data_tensor)
torch_reconst = torch_pca.inverse_transform(torch_proj)

# 4. compare the results
print(" Compare the Results ")
print(" #1. Principal Components ")
# print("sklearn PCA")
# print(sk_pca.components_)

# print("custom torch PCA")
# print(torch_pca.components_)

print("Check if almost close or not")
print(np.isclose(sk_pca.components_, torch_pca.components_).all())

print("-" * 100)

print(" #2. Projection Results ")
# print("sklearn PCA")
# print(sk_proj)

# print("custom torch PCA")
# print(torch_proj)

print("Check if almost close or not")
print(np.isclose(sk_proj, torch_proj).all())

print("-" * 100)

print( "#3. Inverse trnasform (reconsturction) ")
# print("sklearn PCA")
# print(sk_reconst)

# print("custom torch PCA")
# print(torch_reconst)

print("Check if almost close or not")
print(np.isclose(sk_reconst, torch_reconst).all())

print("-" * 100)