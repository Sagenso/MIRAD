from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class Digitizer:

    def __init__(self, max_bins: int):
        self.max_bins = max_bins
        self.min_weight = (1 / max_bins) / 2
        self.cutoffs_ = None
        self.weights_ = None

    def fit(self, X):
        X = np.array(X)
        self.cutoffs_, self.weights_ = map(list, zip(*(
            self._get_cutoffs_and_weights_for_feature(X[:, col_ix])
            for col_ix in range(X.shape[1])
        )))
        return self

    def transform(self, X):
        X = np.array(X)
        digitized = np.empty_like(X)
        for col_ix, cutoffs in enumerate(self.cutoffs_):
            digitized[:, col_ix] = np.digitize(X[:, col_ix], cutoffs, right=True)
        return digitized

    def _get_cutoffs_and_weights_for_feature(self, x):
        initial_cutoffs = np.unique([
            np.quantile(x, q, method='lower')
            for q in np.arange(1, self.max_bins) / self.max_bins
        ])
        final_cutoffs, weights = self._merge_narrow_bins(x, initial_cutoffs)
        return final_cutoffs, weights

    def _merge_narrow_bins(self, x, cutoffs):
        weights = self._get_weights_from_cutoffs(x, cutoffs)
        # this implementation favors legibility over efficiency
        while np.min(weights) < self.min_weight:
            ix = np.argmin(weights)
            neighbor_weight_and_border_cutoff_ix_pairs = []
            if 0 < ix:
                neighbor_weight_and_border_cutoff_ix_pairs.append((weights[ix - 1], ix - 1))
            if ix + 1 < len(weights):
                neighbor_weight_and_border_cutoff_ix_pairs.append((weights[ix + 1], ix))
            assert neighbor_weight_and_border_cutoff_ix_pairs
            smallest_neighbor_weight, cutoff_to_delete = min(neighbor_weight_and_border_cutoff_ix_pairs)
            cutoffs = np.delete(cutoffs, cutoff_to_delete)
            weights = self._get_weights_from_cutoffs(x, cutoffs)
        return cutoffs, weights

    @staticmethod
    def _get_weights_from_cutoffs(x, cutoffs):
        return np.diff([0, *(np.mean(x <= cutoff) for cutoff in cutoffs), 1])


class EmbeddingSumModule(nn.Module):

    def __init__(self, values_weights: list[list[float]], free_term, dtype=torch.float32):
        super().__init__()
        self.values_weights = [
            torch.tensor(w, dtype=dtype, requires_grad=False)
            for w in values_weights
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=len(w),
                embedding_dim=1,
                _weight=torch.zeros((len(w), 1), dtype=dtype, requires_grad=True),
            )
            for w in values_weights
        ])
        self.free_term = nn.Parameter(
            data=free_term.clone().detach(),
            requires_grad=True,
        )
        self.dtype = dtype

    def forward(self, X):
        result = self.free_term.expand(len(X))
        assert X.shape[1] == len(self.embeddings)
        for i, emb in enumerate(self.embeddings):
            result = result + emb(X[:, i]).flatten()
        return result

    def mean_square_step(self):
        return torch.concat([
            (emb.weight[1:] - emb.weight[:-1]).flatten()
            for emb in self.embeddings
        ]).pow(2).mean()

    def mean_square_embedding_sum(self):
        return torch.concat([
            ((w @ emb.weight).sum() / w.sum()).view(1)
            for w, emb in zip(self.values_weights, self.embeddings)
        ]).pow(2).mean()

    @torch.no_grad()
    def feature_importance(self, i: int | None = None):
        if i is None:
            return torch.tensor([self.feature_importance(i) for i in range(len(self.embeddings))])
        else:
            w = self.values_weights[i] / self.values_weights[i].sum()
            emb = self.embeddings[i].weight.view(w.shape)
            mean = (w @ emb).sum()
            return (w @ (emb - mean).abs()).sum()


class EmbeddingSumClassifier:
    """ Scikit-learn-compatible classifier """

    def __init__(
            self,
            max_bins: int,
            max_epochs: int,
            lr: float,
            step_loss_weight: float,
            embedding_sum_loss_weight: float,
    ):
        super().__init__()
        self.digitizer = Digitizer(max_bins=max_bins)
        self.max_epochs = max_epochs
        self.lr = lr
        self.step_loss_weight = step_loss_weight
        self.embedding_sum_loss_weight = embedding_sum_loss_weight

        self.classes_ = np.array([0, 1])  # sklearn compatibility
        self.module_: EmbeddingSumModule | None = None
        self.training_history_: list[dict[str, Any]] | None = None

    def fit(self, X, y, weight=None):  # sklearn compatibility
        self.digitizer.fit(X)
        X_tensor = torch.tensor(self.digitizer.transform(X), dtype=torch.int32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
        weight_tensor = None if weight is None else torch.tensor(np.array(weight), dtype=torch.float32)
        self.module_ = EmbeddingSumModule(
            values_weights=self.digitizer.weights_,
            free_term=torch.logit(torch.mean(y_tensor)),
        )
        self.module_.train()
        with torch.enable_grad():
            self.train(X_tensor, y_tensor, weight=weight_tensor)
        self.module_.eval()
        return self

    def predict_proba(self, X):  # sklearn compatibility
        X_tensor = torch.tensor(self.digitizer.transform(X), dtype=torch.int32)
        with torch.no_grad():
            y_pred = F.sigmoid(self.module_(X_tensor)).numpy()
        result = np.empty(shape=(X.shape[0], 2))
        result[:, 0] = 1 - y_pred
        result[:, 1] = y_pred
        return result

    def train(self, X, y_true, weight=None):
        m = self.module_
        optimizer = torch.optim.SGD(params=m.parameters(), lr=self.lr)
        self.training_history_ = []
        for epoch in range(self.max_epochs):
            m.zero_grad()
            y_pred = m(X)
            train_clf_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight)
            train_step_loss = m.mean_square_step()
            train_embedding_sum_loss = m.mean_square_embedding_sum()
            train_loss = (
                    train_clf_loss
                    + train_step_loss * self.step_loss_weight
                    + train_embedding_sum_loss * self.embedding_sum_loss_weight
            )
            train_loss.backward()
            optimizer.step()
            self.training_history_.append({
                'epoch': epoch,
                'train_loss': train_loss.detach().item(),
                'train_clf_loss': train_clf_loss.detach().item(),
                'train_step_loss': train_step_loss.detach().item(),
                'train_embedding_sum_loss': train_embedding_sum_loss.detach().item(),
            })

    def plot_training_history(self):
        (
            pd.DataFrame(self.training_history_)
            .assign(
                train_step_loss_weighted=lambda df: df['train_step_loss'] * self.step_loss_weight,
                train_embedding_sum_loss_weighted=lambda df: (
                    df['train_embedding_sum_loss'] * self.embedding_sum_loss_weight
                ),
            )
            .set_index('epoch')
            .plot()
        )

    def visualize(self, feature_names: list[str] | None = None, subset: list[str] = None):
        if feature_names is None:
            feature_names = list(map('feature {}'.format, range(len(self.module_.embeddings))))
        assert len(feature_names) == len(self.module_.embeddings)
        max_embedding_abs = max(
            max(np.abs(emb.weight.data.detach().numpy()[:, 0]))
            for emb in self.module_.embeddings
        )
        for i, feature_name in enumerate(feature_names):
            if subset is not None and feature_name not in subset:
                continue

            # data
            embedding_values = self.module_.embeddings[i].weight.data.detach().numpy()[:, 0]
            cutoffs = self.digitizer.cutoffs_[i]
            widths = self.digitizer.weights_[i]
            xs = np.concatenate([[0], widths.cumsum()])

            fig, ax = plt.subplots()
            ax.set_title(feature_name)
            for x in xs[1:-1]:
                ax.axvline(x, color='grey', linestyle=':', linewidth=1)
            ax.set_xlim((xs[0], xs[-1]))
            ax.bar(
                x=xs[:-1],
                width=widths,
                height=embedding_values,
                align='edge',
                color=np.where(embedding_values > 0, 'red', 'green')
            )
            ax.axhline(0, color='grey', linewidth=1)
            ax.set_ylim((-max_embedding_abs - .1, max_embedding_abs + .1))
            ax.set_xticks([])
            ax.set_ylabel('embedding value')

            ax2 = ax.twinx()
            ax2.scatter(x=xs[1:-1], y=cutoffs, color='royalblue')
            ax2.tick_params(axis='y', labelcolor='royalblue')
            ax2.set_ylabel('cutoff', color='royalblue')

    def feature_importance(self, feature_names: list[str] | None = None) -> pd.Series:
        if feature_names is None:
            feature_names = list(map('feature {}'.format, range(len(self.module_.embeddings))))
        assert len(feature_names) == len(self.module_.embeddings)
        return pd.Series(
            self.module_.feature_importance().numpy(),
            index=feature_names,
            name='importance'
        )
