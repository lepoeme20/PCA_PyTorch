# PyTorch PCA

This repository contains code for performing Principal Component Analysis (PCA) using PyTorch, a popular deep learning framework. PCA is a widely used dimensionality reduction technique that finds the principal components of a dataset, which are orthogonal directions that capture the largest amount of variance in the data.

## Dependencies

To run the code in this repository, you will need:

- Python 3
- PyTorch
- NumPy

You can install these dependencies using pip:

```
pip install torch numpy
```

## Usage

The PCA code is contained in the file `pca.py`. `main.py` shows how to use custom PyTorch PCA. You can use this code to perform PCA on a dataset by calling the `pca` function:

```python
import torch
from pca import TorchPCA

# Create a dataset of random data
data = torch.randn(100, 10)

# Perform PCA on the dataset
pca = TorchPCA(n_components=4)
pca.fit(data)
proj = pca.transform(data)
reconst = pca.inverse_transform(proj)

# Print the results
print(pca.components_)
print(proj)
print(reconst)
```

The `pca` function takes a PyTorch tensor as input and have methods same as PCA module in sklean:

 - pca.components_
 - pca.mean_
 - pca.n_samples
 - pca.explained_variance_
 - pca.explained_variance_ratio_
 - pca.singular_values