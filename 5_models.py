
import torch
torch.manual_seed(0)
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import funcions as funcions
import GNN as GNN


#--------------------------------------------------------------------------------------------------------------------------------------------------  
adj_all = np.load('/Users/aina/Desktop/TFG/codi/xarxes/multilayer.npy')
print(f"Mida de adj_all: {adj_all.shape}")  # (270, 152, 152)

x_all = np.load('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multiplex.npy')
print(f"Mida de x_all: {x_all.shape}")  # (270, 76, 7)
x_concatenat = np.concatenate([x_all, x_all], axis=1)
print(f"Mida de x: {x_concatenat.shape}")  # (270, 152, 7)

y_all = np.load("/Users/aina/Desktop/TFG/codi/etiquetes/etiquetes.npy")
print(f"Mida de y_all: {y_all.shape}")  # (270,)

#--------------------------------------------------------------------------------------------------------------------------------------------------  
data_list = []
for i in range(adj_all.shape[0]):
    A = adj_all[i] 
    x = torch.tensor(x_concatenat[i], dtype=torch.float)  
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    y = torch.tensor([y_all[i]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)

data_list_balanced = funcions.smote_graph_level(data_list, target_class=0, target_total=200)
data_list_balanced = [d for d in data_list_balanced if d.y.item() == 0] + [d for d in data_list_balanced if d.y.item() == 1]
dataset = funcions.MyGraphDataset(data_list_balanced)

print(f'\nDataset: {dataset}')
print('-----------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {dataset[0].x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

#--------------------------------------------------------------------------------------------------------------------------------------------------  
train_dataset, val_dataset, test_dataset = funcions.balancejar_dataset(dataset)

print(f'\nTraining set   = {len(train_dataset)} graphs')
print(f'Validation set = {len(val_dataset)} graphs')
print(f'Test set       = {len(test_dataset)} graphs')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=True)

print('\nTrain loader:')
for i, batch in enumerate(train_loader):
    print(f' - Batch {i}: {batch}')

print('\nValidation loader:')
for i, batch in enumerate(val_loader):
    print(f' - Batch {i}: {batch}')

print('\nTest loader:')
for i, batch in enumerate(test_loader):
    print(f' - Batch {i}: {batch}\n')


#--------------------------------------------------------------------------------------------------------------------------------------------------  

gcn = GNN.GCN(dataset, dim_h = 256)
print('GCN:')
gcn, gcn_losses = GNN.train(gcn, train_loader, val_loader, epochs=100)
gcn_test_loss, gcn_test_acc, gcn_test_rec_pos, gcn_test_rec_neg = GNN.test(gcn, test_loader)
print(f'Test Loss: {gcn_test_loss:.2f} | Test Acc: {gcn_test_acc*100:.2f}% | '
      f'Test Recall Pos: {gcn_test_rec_pos*100:.2f}% | Test Recall Neg: {gcn_test_rec_neg*100:.2f}%\n')


gin = GNN.GIN(dataset, dim_h=256)
print('GIN:')
gin, gin_losses = GNN.train(gin, train_loader, val_loader, epochs=100)
gin_test_loss, gin_test_acc, gin_test_rec_pos, gin_test_rec_neg = GNN.test(gin, test_loader)
print(f'Test Loss: {gin_test_loss:.2f} | Test Acc: {gin_test_acc*100:.2f}% | '
      f'Test Recall Pos: {gin_test_rec_pos*100:.2f}% | Test Recall Neg: {gin_test_rec_neg*100:.2f}%\n')


gat = GNN.GAT(dataset, dim_in=7, dim_h=256, dim_out=2)
print('GAT:')
gat, gat_losses = GNN.train(gat, train_loader, val_loader, epochs=100)
gat_test_loss, gat_test_acc, gat_test_rec_pos, gat_test_rec_neg = GNN.test(gat, test_loader)
print(f'Test Loss: {gat_test_loss:.2f} | Test Acc: {gat_test_acc*100:.2f}% | '
      f'Test Recall Pos: {gat_test_rec_pos*100:.2f}% | Test Recall Neg: {gat_test_rec_neg*100:.2f}%\n')

#--------------------------------------------------------------------------------------------------------------------------------------------------

train_losses_all = {
    'GCN': gcn_losses,
    'GIN': gin_losses,
    'GAT': gat_losses
}

test_metrics_all = {
    'GCN': {
        'test_loss': gcn_test_loss,
        'test_acc': gcn_test_acc,
        'test_rec_pos': gcn_test_rec_pos,
        'test_rec_neg': gcn_test_rec_neg
    },
    'GIN': {
        'test_loss': gin_test_loss,
        'test_acc': gin_test_acc,
        'test_rec_pos': gin_test_rec_pos,
        'test_rec_neg': gin_test_rec_neg
    },
    'GAT': {
        'test_loss': gat_test_loss,
        'test_acc': gat_test_acc,
        'test_rec_pos': gat_test_rec_pos,
        'test_rec_neg': gat_test_rec_neg
    }
}

funcions.guardar_resultats_complets(train_losses_all, test_metrics_all)

#--------------------------------------------------------------------------------------------------------------------------------------------------  


'''
plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in train_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/plots/informe/loss/gcn_1capes_100epoques_penalitzat.png', dpi=300)
plt.show()
'''
# El recall positiu mesura la capacitat del model per identificar correctament les instàncies positives, 
# El recall negatiu mesura la capacitat del model per identificar correctament les instàncies negatives.

#source /Users/aina/Desktop/TFG/codi/tfg_env/bin/activate
