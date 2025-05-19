
import torch
torch.manual_seed(0)
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import random
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


class MyGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.', None, None, None)
        self.data, self.slices = self.collate(data_list)
    
    def get(self, idx):
        return super().get(idx)
    

def smote_graph_level(data_list, target_class=0, target_total=200):
    class_data = [d for d in data_list if d.y.item() == target_class]
    current_num = len(class_data)
    needed = target_total - current_num
    new_graphs = []
    for _ in range(needed):
        g1, g2 = random.sample(class_data, 2)
        assert g1.x.size() == g2.x.size()
        alpha = random.uniform(0, 1)
        new_x = g1.x * (1 - alpha) + g2.x * alpha
        new_edge_index = g1.edge_index  
        new_y = torch.tensor([target_class], dtype=torch.long)
        new_data = Data(x=new_x, edge_index=new_edge_index, y=new_y)
        new_graphs.append(new_data)
    return data_list + new_graphs


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


def guardar_resultats(train_losses_all, test_metrics_all):
    max_epochs = max(len(losses) for losses in train_losses_all.values())
    df_losses = pd.DataFrame({'Epoch': list(range(1, max_epochs + 1))})

    for model_name, losses in train_losses_all.items():
        padded_losses = losses + [None] * (max_epochs - len(losses))
        df_losses[f'{model_name} Train Loss'] = padded_losses

    test_data = []
    for model_name, metrics in test_metrics_all.items():
        row = {'Model': model_name}
        row.update(metrics)
        test_data.append(row)
    df_test = pd.DataFrame(test_data)

    with pd.ExcelWriter('/Users/aina/Desktop/TFG/codi/resultats') as writer:
        df_losses.to_excel(writer, sheet_name='Train Losses', index=False)
        df_test.to_excel(writer, sheet_name='Test Results', index=False)

    print(f"\n✅ Resultats guardats a: {'/Users/aina/Desktop/TFG/codi/resultats'}")
