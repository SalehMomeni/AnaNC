import torch
import numpy as np
from collections import defaultdict

class MD:
    def __init__(self, reg: float, device: torch.device):
        self.device = device
        self.reg = reg

        self.num_features = None
        self.class_means = {}
        self.class_counts = defaultdict(int)
        self.sigma = None
        self.total_count = 0

        self.sigma_inv = None
        self.is_updated = False

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.int64, device=self.device)

        if self.num_features is None:
            self.num_features = X.shape[1]
            self.sigma = torch.zeros((self.num_features, self.num_features), device=self.device)

        for cls in Y.unique():
            cls_mask = (Y == cls)
            X_cls = X[cls_mask]
            n_new = X_cls.size(0)
            cls_mean_new = X_cls.mean(dim=0)

            n_old = self.class_counts[cls.item()]
            if n_old == 0:
                self.class_means[cls.item()] = cls_mean_new
            else:
                cls_mean_old = self.class_means[cls.item()]
                updated_mean = (n_old * cls_mean_old + n_new * cls_mean_new) / (n_old + n_new)
                self.class_means[cls.item()] = updated_mean

            self.class_counts[cls.item()] += n_new

            centered = X_cls - self.class_means[cls.item()]
            self.sigma += centered.T @ centered
            self.total_count += n_new

        self.is_updated = False

    def _update_sigma_inv(self):
        sigma = self.sigma + self.reg * torch.eye(self.num_features, device=self.device)
        self.sigma_inv = torch.inverse(sigma)
        self.is_updated = True

    def score(self, X: np.ndarray, batch_size = 1024):
        X = torch.tensor(X, dtype=torch.float32)
        
        if not self.is_updated:
            self._update_sigma_inv()

        class_labels = list(self.class_means.keys())
        means = torch.stack([self.class_means[cls] for cls in class_labels])  # (C, D)
        means_exp = means.unsqueeze(0)  # (1, C, D)

        scores = []
        for i in range(0, X.size(0), batch_size):
            x_batch = X[i:i+batch_size].to(self.device)
            x_exp = x_batch.unsqueeze(1)  # (B, 1, D)
            delta = x_exp - means_exp  # (B, C, D)
            dists = torch.einsum("bcd,df,bcf->bc", delta, self.sigma_inv, delta)  # (B, C)
            min_dists, _ = torch.min(dists, dim=1)
            scores.append(-min_dists.cpu())

        return torch.cat(scores).numpy()  # ID Score

    def predict(self, X: np.ndarray, batch_size = 1024):
        X = torch.tensor(X, dtype=torch.float32)
        
        if not self.is_updated:
            self._update_sigma_inv()

        class_labels = list(self.class_means.keys())
        means = torch.stack([self.class_means[cls] for cls in class_labels])  # (C, D)
        means_exp = means.unsqueeze(0)  # (1, C, D)

        preds = []
        for i in range(0, X.size(0), batch_size):
            x_batch = X[i:i+batch_size].to(self.device)
            x_exp = x_batch.unsqueeze(1)  # (B, 1, D)
            delta = x_exp - means_exp  # (B, C, D)
            dists = torch.einsum("bcd,df,bcf->bc", delta, self.sigma_inv, delta)  # (B, C)
            min_indices = torch.argmin(dists, dim=1)
            batch_preds = [class_labels[idx.item()] for idx in min_indices]
            preds.extend(batch_preds)

        return np.array(preds)
