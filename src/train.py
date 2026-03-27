from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from nn_utils import DEFAULT_RANDOM_SEED, EarlyStopping, Engine, FeedForwardNN, seed_everything


def eval_metric(targets, outputs):
    targets = 100 * (1 - (targets / targets[-1]))
    outputs = 100 * (1 - (outputs / outputs[-1]))
    return mean_absolute_error(targets, outputs)


def run_training(train, features, target, fold, config):
    seed = config['training_params'].get('seed', DEFAULT_RANDOM_SEED)
    device = config['training_params'].get('device', 'cpu')
    epochs = config['training_params'].get('epochs', 1000)
    batch_size = config['training_params'].get('batch_size', 16)
    verbose = config['training_params'].get('verbose', True)
    use_early_stopping = config['training_params'].get('use_early_stopping', True)
    fold_column = config['training_params'].get('fold_column', 'fold')

    seed_everything(seed=seed)
    train = train.copy()
    train_fold = train[train[fold_column] != fold].copy()
    valid_fold = train[train[fold_column] == fold].copy()

    train_fold_x = train_fold[features].to_numpy()
    valid_fold_x = valid_fold[features].to_numpy()
    train_fold_y = train_fold[target].to_numpy().reshape(-1, 1)
    valid_fold_y = valid_fold[target].to_numpy().reshape(-1, 1)

    scaler_fold_x, scaler_fold_y = StandardScaler(), StandardScaler()
    train_fold_x = scaler_fold_x.fit_transform(train_fold_x)
    train_fold_y = scaler_fold_y.fit_transform(train_fold_y)
    valid_fold_x = scaler_fold_x.transform(valid_fold_x)
    valid_fold_y = scaler_fold_y.transform(valid_fold_y)

    train_fold_x = torch.from_numpy(train_fold_x).type(torch.Tensor).to(device)
    valid_fold_x = torch.from_numpy(valid_fold_x).type(torch.Tensor).to(device)
    train_fold_y = torch.from_numpy(train_fold_y).type(torch.Tensor).to(device)
    valid_fold_y = torch.from_numpy(valid_fold_y).type(torch.Tensor).to(device)

    train_dataset = TensorDataset(train_fold_x, train_fold_y)
    valid_dataset = TensorDataset(valid_fold_x, valid_fold_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False, drop_last=False)

    model = FeedForwardNN(**config['model_params'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer_params'])
    eng = Engine(model=model, optimizer=optimizer, loss_fn=nn.MSELoss())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler_params'])

    if use_early_stopping:
        early_stopping = EarlyStopping(path=f'checkpoint_fold{fold}.pt', **config['early_stopping_params'])

    train_losses = []
    valid_losses = []
    train_eval_metrics = []
    valid_eval_metrics = []
    stopping_round = None

    for epoch in range(epochs):
        train_loss = 0.0
        train_samples = 0
        for train_fold_x_batch, train_fold_y_batch in train_loader:
            train_loss_batch, _ = eng.train(train_fold_x_batch, train_fold_y_batch)
            train_loss += train_loss_batch * train_fold_x_batch.size(0)
            train_samples += train_fold_x_batch.size(0)
        train_loss /= train_samples

        valid_loss = 0.0
        valid_samples = 0
        for valid_fold_x_batch, valid_fold_y_batch in valid_loader:
            valid_loss_batch, _ = eng.evaluate(valid_fold_x_batch, valid_fold_y_batch)
            valid_loss += valid_loss_batch * valid_fold_x_batch.size(0)
            valid_samples += valid_fold_x_batch.size(0)
        valid_loss /= valid_samples

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        model.eval()
        with torch.no_grad():
            train_fold_preds = scaler_fold_y.inverse_transform(model(train_fold_x).cpu().numpy())
            valid_fold_preds = scaler_fold_y.inverse_transform(model(valid_fold_x).cpu().numpy())
            train_eval_metric = eval_metric(train_fold[target].to_numpy(), train_fold_preds)
            valid_eval_metric = eval_metric(valid_fold[target].to_numpy(), valid_fold_preds)

        train_eval_metrics.append(train_eval_metric)
        valid_eval_metrics.append(valid_eval_metric)

        if verbose:
            print(
                f'Fold {fold} Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                f'Valid Loss: {valid_loss:.4f}, Train Eval: {train_eval_metric:.4f}, '
                f'Valid Eval: {valid_eval_metric:.4f}'
            )

        if use_early_stopping:
            early_stopping(valid_eval_metric, model)
            if early_stopping.early_stop:
                print('Early stopping triggered')
                stopping_round = epoch - early_stopping.patience
                break

        scheduler.step(valid_eval_metric)

    if use_early_stopping and early_stopping.early_stop:
        early_stopping.load_checkpoint(model)

    model.eval()
    with torch.no_grad():
        train_preds = scaler_fold_y.inverse_transform(model(train_fold_x).cpu().numpy())
        valid_preds = scaler_fold_y.inverse_transform(model(valid_fold_x).cpu().numpy())

    return {
        'best_model': model,
        'best_loss': early_stopping.best_loss if use_early_stopping else valid_eval_metric,
        'train_predictions': train_preds,
        'valid_predictions': valid_preds,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_eval_metrics': train_eval_metrics,
        'valid_eval_metrics': valid_eval_metrics,
        'stopping_round': stopping_round if use_early_stopping else epoch,
    }
