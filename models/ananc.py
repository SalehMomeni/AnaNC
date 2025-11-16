import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

class FeedForward:
    def __init__(self, D, input_dim, device, reg):
        self.D = D
        self.device = device
        self.W = torch.randn(D, input_dim, device=device)
        self.A = torch.zeros(D, D, device=device)
        self.Z_means = defaultdict(lambda: torch.zeros(D, device=self.device))
        self.counts = defaultdict(int)
        self.reg = reg
        self.theta = None

    def _normalize(self, X):
        return F.normalize(X, p=2, dim=1)

    def update_stats(self, X, Y):
        X = self._normalize(X)
        Z = F.gelu(X @ self.W.T)
        self.A += Z.T @ Z
        for c in torch.unique(Y):
            mask = (Y == c)
            c = c.item()
            z_mean = Z[mask].mean(dim=0)
            n = mask.sum().item()
            total = self.counts[c] + n
            self.Z_means[c] = (self.Z_means[c] * self.counts[c] + z_mean * n) / total
            self.counts[c] = total

    def solve_theta(self, target_means):
        B = torch.zeros((self.D, target_means.shape[1]), device=self.device)
        for c, z_mean in self.Z_means.items():
            B += self.counts[c] * torch.outer(z_mean, target_means[c])
        self.theta = torch.linalg.solve(self.A + self.reg * torch.eye(self.D, device=self.device), B)

    def forward(self, X):
        X = self._normalize(X)
        Z = F.gelu(X @ self.W.T)
        return Z @ self.theta


class AnaNC:
    def __init__(self, D:int, reg:float, seed:int, device:torch.device):
        self.device = device
        self.D = D
        self.reg = reg
        self.seed = seed
        torch.manual_seed(self.seed)

        self.input_dim = None
        self.num_classes = 0
        self.class_counts = defaultdict(int)
        self.class_means = None
        self.class_map = {}
        self.inverse_map = {}
        self.num_updates = 0

    def _update_class_map(self, Y):
        new_labels = sorted(set(Y.tolist()))
        for label in new_labels:
            if label not in self.class_map:
                mapped_id = self.num_classes
                self.class_map[label] = mapped_id
                self.inverse_map[mapped_id] = label
                self.num_classes += 1

    def _ETF_proj(self, M):
        C, d = M.shape
        # S = I - (1/C) * 11^T
        S = torch.eye(C, device=self.device) - (1.0 / C) * torch.ones(C, C, device=self.device)

        eigvals, eigvecs = torch.linalg.eigh(S)  # eigvecs: (C, C), eigvals: (C,)
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0.0))
        A = eigvecs @ torch.diag(sqrt_eigvals)  # (C, C)

        # Solve Procrustes: find Q to minimize || A Q^T - M ||_F^2
        U, _, Vt = torch.linalg.svd(M.T @ A, full_matrices=False)
        Q = U @ Vt  # (D, C)
        M_ETF = A @ Q.T  # shape: (C, D)
        return M_ETF
    
    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.long, device=self.device)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
            self.feedforward = FeedForward(self.D, self.input_dim, self.device, self.reg)
            self.class_means = defaultdict(lambda: torch.zeros(self.input_dim, device=self.device))

        self._update_class_map(Y)
        Y_mapped = torch.tensor([self.class_map[y.item()] for y in Y], device=self.device)

        for c in torch.unique(Y_mapped):
            c_int = c.item()
            x_mean = X[Y_mapped == c].mean(dim=0)
            n = (Y_mapped == c).sum().item()
            total = self.class_counts[c_int] + n
            self.class_means[c_int] = (self.class_means[c_int] * self.class_counts[c_int] + x_mean * n) / total
            self.class_counts[c_int] = total

        stacked_means = torch.stack([self.class_means[c] for c in range(self.num_classes)])
        if self.num_updates == 0:
            target_means = stacked_means
        else:
            target_means = self._ETF_proj(stacked_means)
        self.num_updates += 1

        self.feedforward.update_stats(X, Y_mapped)
        self.feedforward.solve_theta(target_means)

        # Compute class prototypes
        self.class_prototypes = self.feedforward.forward(stacked_means)

    def score(self, X: np.ndarray, batch_size: int = 1024):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        scores = []
        for i in range(0, X.size(0), batch_size):
            x_batch = X[i:i+batch_size]
            outs = self.feedforward.forward(x_batch)
            sims = F.cosine_similarity(outs.unsqueeze(1), self.class_prototypes.unsqueeze(0), dim=-1)
            max_sim, _ = sims.max(dim=1)
            scores.append(max_sim)
        scores = torch.cat(scores)
        return scores.cpu().numpy()  # ID Score

    def predict(self, X: np.ndarray, batch_size = 1024):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        for i in range(0, X.size(0), batch_size):
            x_batch = X[i:i+batch_size]
            outs = self.feedforward.forward(x_batch)
            sims = F.cosine_similarity(outs.unsqueeze(1), self.class_prototypes.unsqueeze(0), dim=-1)
            pred_classes = sims.argmax(dim=1)
            preds.append(pred_classes)
        preds = torch.cat(preds)
        return np.array([self.inverse_map[p.item()] for p in preds])
