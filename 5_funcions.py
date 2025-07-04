
import torch
torch.manual_seed(0)
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import random
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


def balancejar_dataset(dataset):
    etiquetes = np.array([data.y.item() for data in dataset])
    indexs = np.arange(len(dataset))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(sss.split(indexs, etiquetes))

    temp_labels = etiquetes[temp_idx]
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss_val_test.split(temp_idx, temp_labels))

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset   = [dataset[i] for i in temp_idx[val_idx]]
    test_dataset  = [dataset[i] for i in temp_idx[test_idx]]
    return train_dataset, val_dataset, test_dataset


def guardar_resultats(train_losses_all, test_metrics_all, path):
    models = list(train_losses_all.keys())
    num_epochs = len(next(iter(train_losses_all.values())))
    epochs = list(range(1, num_epochs + 1))
    df_train = pd.DataFrame({'epoch': epochs})
    for model in models:
        df_train[f'{model}_train_loss'] = train_losses_all[model]

    test_data = {'epoch': 'TEST'}
    for model in models:
        metrics = test_metrics_all[model]
        test_data[f'{model}_train_loss'] = metrics['test_loss'] 
        test_data[f'{model}_test_acc'] = metrics['test_acc']
        test_data[f'{model}_test_rec_pos'] = metrics['test_rec_pos']
        test_data[f'{model}_test_rec_neg'] = metrics['test_rec_neg']
    
    df_test = pd.DataFrame([test_data])
    for col in df_test.columns:
        if col not in df_train.columns:
            df_train[col] = None
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all.to_csv(path, index=False)
