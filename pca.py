import torch
from typing import Tuple
from torch.linalg import svd

class TorchPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, data: torch.Tensor):
        assert len(data.size()) == 2, "Data should be matrix"
        min_value = min(data.size(0), data.size(1))
        assert self.n_components <= min_value, \
            f"n_components={self.n_components} must be between 0 and min(n_samples, n_features)={min_value}"
        self._fit(data)
        return self

    def _fit(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples, n_features = data.size()

        self.mean_ = torch.mean(data, dim=0)
        self.scale_ = torch.std(data, dim=0)

        # Singular value decomposition of a matrix
        data -= self.mean_
        U, S, Vh = svd(data, full_matrices=False)
        max_abs_idx = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[max_abs_idx, range(U.size(1))])
        U *= signs
        Vh *= signs[:, None]

        components_ = Vh

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (n_samples)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.clone().detach()

        if self.n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[self.n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        self.n_samples = n_samples
        self.components_ = components_[:self.n_components]
        self.explained_variance_ = explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        self.singular_values = singular_values_[:self.n_components]

        return U, S, Vh

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return data @ self.components_.T

    def inverse_transform(self, proj: torch.Tensor) -> torch.Tensor:
        reconstruction = proj @ self.components_ + self.mean_

        return reconstruction