import torch
import numpy as np

class Residuals:
    def __init__(self, n_comp: int, device: torch.device):
        self.n_comp = n_comp
        self.device = device
        self.gram = None
        self.q = None
        self.input_dim = None

    def update(self, X: np.ndarray, Y: np.ndarray):
        '''Y is passed for consistency (not used)'''
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
            self.gram = torch.zeros((self.input_dim, self.input_dim), device=self.device)

        self.gram += X.T @ X
        eigvals, eigvecs = torch.linalg.eigh(self.gram)
        self.q = eigvecs[:, :self.n_comp]  # shape: (D, n_comp)

    def score(self, X: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        proj = X @ self.q
        residual = torch.norm(proj, dim=1)
        return -residual.cpu().numpy()
