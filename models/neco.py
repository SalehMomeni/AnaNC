import torch
import numpy as np
import pickle

class NECO:
    def __init__(self, fc_path: str, n_comp: int, device: torch.device):
        self.fc_path = fc_path
        self.n_comp = n_comp
        self.device = device
        self.gram = None
        self.P = None
        self.input_dim = None

        if fc_path is not None:
            with open(fc_path, 'rb') as f:
                w, b = pickle.load(f)
            self.fc_weight = torch.tensor(w, dtype=torch.float32, device=self.device)  # (C, D)
            self.fc_bias = torch.tensor(b, dtype=torch.float32, device=self.device)    # (C,)

    def update(self, X: np.ndarray, Y: np.ndarray):
        '''Y is passed for consistency (not used)'''
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
            self.gram = torch.zeros((self.input_dim, self.input_dim), device=self.device)

        self.gram += X.T @ X
        eigvals, eigvecs = torch.linalg.eigh(self.gram)
        self.P = eigvecs[:, self.n_comp:]  # shape: (D, D - n_comp)

    def score(self, X: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        proj = X @ self.P
        top_norm = torch.norm(proj, dim=1) # (N,)
        input_norm = torch.norm(X, dim=1)  # (N,)
        neco = top_norm / (input_norm + 1e-8)

        if self.fc_path is not None:
            logits = X @ self.fc_weight.T + self.fc_bias  # (N, C)
            max_logits, _ = torch.max(logits, dim=1)      # (N,)
            neco = neco * max_logits # scale by max logit

        return neco.cpu().numpy()
