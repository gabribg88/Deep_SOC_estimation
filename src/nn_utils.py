import os
import random
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
import torch
import torch.nn as nn


def create_folds(train, features, target, num_folds, shuffle=True, seed=42, groups='Curve_id'):
    del num_folds, shuffle, seed
    folds = []
    splitter = LeaveOneGroupOut()
    for train_fold_idx, valid_fold_idx in splitter.split(
        X=train[features], y=train[target], groups=train[groups]
    ):
        folds.append((train_fold_idx, valid_fold_idx))
    return folds


class FeedForwardNN(nn.Module):
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size):
        super().__init__()
        layers = [nn.Linear(nfeatures, hidden_size), nn.ReLU()]
        for _ in range(1, nlayers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, ntargets))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualFeedForwardNN(nn.Module):
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size):
        super().__init__()
        self.first_layer = nn.Sequential(nn.Linear(nfeatures, hidden_size), nn.ReLU())
        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(nlayers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, ntargets)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x) + x
        return self.output_layer(x)


class Engine:
    def __init__(self, model, optimizer, loss_fn=nn.MSELoss()):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def process_batch(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def train(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        loss, outputs = self.process_batch(inputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), outputs

    def evaluate(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            loss, outputs = self.process_batch(inputs, targets)
        return loss.item(), outputs


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', save_model=True, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.path = path
        self.save_model = save_model
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_model:
                self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            if val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                if self.save_model:
                    self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))


def report_metrics(true, preds, model_name, res_df=None):
    if res_df is None:
        res_df = pd.DataFrame(data=0.0, columns=['MAE', 'MSE', 'MAPE'])
    res_df.loc[model_name, 'MAE'] = mean_absolute_error(true, preds)
    res_df.loc[model_name, 'MSE'] = mean_squared_error(true, preds)
    res_df.loc[model_name, 'MAPE'] = mean_absolute_percentage_error(true, preds) * 100
    return res_df


def plot_diagnostic(preds_df, target, preds_column, bins=50, percentiles=(0.05, 0.95), figsize=(16, 5)):
    df = preds_df.copy()
    df['residual'] = df[target] - df[preds_column]
    grouped = df.groupby(target)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.errorbar(
        grouped.groups.keys(),
        grouped[preds_column].median().values,
        yerr=[
            grouped[preds_column].median() - grouped[preds_column].quantile(percentiles[0]),
            grouped[preds_column].quantile(percentiles[1]) - grouped[preds_column].median(),
        ],
        marker='.',
        linestyle='--',
        markersize=18,
        capsize=5,
        color='b',
    )
    bounds = min([ax1.get_xlim()[0], ax1.get_ylim()[0]]), max([ax1.get_xlim()[1], ax1.get_ylim()[1]])
    ax1.set_xlim(bounds)
    ax1.set_ylim(bounds)
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(bounds, bounds, lw=2, ls='--', color='k', alpha=0.5)
    ax1.set_xlabel(r'$q(V_{min})$')
    ax1.set_ylabel(r'$\hat{q}(V_{min})$')

    ax2.errorbar(
        grouped.groups.keys(),
        grouped['residual'].median().values,
        yerr=[
            grouped['residual'].median() - grouped['residual'].quantile(percentiles[0]),
            grouped['residual'].quantile(percentiles[1]) - grouped['residual'].median(),
        ],
        marker='.',
        linestyle='--',
        markersize=18,
        capsize=5,
        color='b',
    )
    ax2.axhline(0, lw=2, ls='--', color='k', alpha=0.5)
    ax2.set_xlabel(r'$q(V_{min})$')
    ax2.set_ylabel(r'$q(V_{min}) - \hat{q}(V_{min})$')

    df['AE'] = (df[target] - df[preds_column]).abs()
    df['AE'].hist(bins=bins, ax=ax3, color='b')
    ax3.set_xlabel('Absolute error [Ah]')
    ax3.axvline(df['AE'].median(), linestyle='--', color='k', label=f'Median AE = {df["AE"].median().round(3)}', lw=2, alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    plt.show()


DEFAULT_RANDOM_SEED = 42


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)


def save_fig(
    fig: matplotlib.figure.Figure,
    fig_name: str,
    fig_dir: str,
    fig_fmt: str,
    fig_size: Tuple[float, float] = [6.4, 4],
    save: bool = True,
    dpi: int = 300,
    transparent_png: bool = True,
):
    if not save:
        return

    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(fig_dir, f'{fig_name}.{fig_fmt}')
    if fig_fmt == 'pdf':
        metadata = {'Creator': '', 'Producer': '', 'CreationDate': None}
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(pth, bbox_inches='tight', dpi=dpi)
