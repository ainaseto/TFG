
import torch
torch.manual_seed(0)
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import funcions as funcions
import GNN as GNN
import time

#--------------------------------------------------------------------------------------------------------------------------------------------------  
adj_all_real = np.load('/Users/aina/Desktop/TFG/codi/xarxes/multilayer.npy')
print(f"Mida de adj_all_real: {adj_all_real.shape}")  # (270, 152, 152)
adj_all_sintetic = np.load('/Users/aina/Desktop/TFG/codi/xarxes/smote/multilayer.npy')
print(f"Mida de adj_all_sintetic: {adj_all_sintetic.shape}")  # (128, 152, 152)
adj_all = np.concatenate([adj_all_real, adj_all_sintetic], axis=0)
print(f"Mida de adj_all: {adj_all.shape}")  # (398, 76, 7)

x_all_real = np.load('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multilayer.npy')
print(f"\nMida de x_all_real: {x_all_real.shape}")  # (270, 76, 7)
x_all_sintetic = np.load('/Users/aina/Desktop/TFG/codi/embeddings/smote/embeddings_multilayer.npy')
print(f"Mida de x_all_sintetic: {x_all_sintetic.shape}")  # (128, 76, 7)
x_all = np.concatenate([x_all_real, x_all_sintetic], axis=0)
print(f"Mida de x_all_total: {x_all.shape}")  # (398, 76, 7)

x_concatenat = np.concatenate([x_all, x_all], axis=1)
print(f"Mida de x: {x_concatenat.shape}")  # (398, 152, 7)

y_all_real = np.load("/Users/aina/Desktop/TFG/codi/etiquetes/etiquetes.npy")
print(f"\nMida de y_all_real: {y_all_real.shape}")  # (270,)
y_all_smote = np.load("/Users/aina/Desktop/TFG/codi/etiquetes/etiquetes_smote.npy")
print(f"Mida de y_all_smote: {y_all_smote.shape}")  # (128,)
y_all = np.concatenate([y_all_real, y_all_smote], axis=0)
print(f"Mida de y_all: {y_all.shape}")  # (270,)

#--------------------------------------------------------------------------------------------------------------------------------------------------  
data_list = []

for k in range(adj_all.shape[0]):
    edge_index = []
    edge_weight = []
    for i in range(adj_all.shape[1]):
        for j in range(adj_all.shape[1]):
            if i != j:
                if adj_all[k, i, j] > 0.0:
                    edge_index.append([i, j])
                    edge_weight.append(adj_all[k, i, j])
    y = torch.tensor(y_all[k], dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor(x_concatenat[k], dtype=torch.float)  
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weight=edge_weight, y=y)
    data_list.append(data)

#--------------------------------------------------------------------------------------------------------------------------------------------------  
train_dataset, val_dataset, test_dataset = funcions.balancejar_dataset(data_list)
from collections import Counter

def comptar_0s_i_1s(dataset, nom):
    y_values = [data.y.item() for data in dataset]
    comptador = Counter(y_values)
    print(f"\n{nom}:")
    print(f"  Classe 0: {comptador.get(0, 0)}")
    print(f"  Classe 1: {comptador.get(1, 0)}")

comptar_0s_i_1s(train_dataset, "Train")
comptar_0s_i_1s(val_dataset, "Validation")
comptar_0s_i_1s(test_dataset, "Test")

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

t0 = time.time()
gcn = GNN.GCN(dim_h = 128)    
print('GCN:')
gcn, gcn_losses = GNN.train(gcn, train_loader, val_loader, epochs=100)
gcn_test_loss, gcn_test_acc, gcn_test_rec_pos, gcn_test_rec_neg = GNN.test(gcn, test_loader)
print(f'Test Loss: {gcn_test_loss:.2f} | Test Acc: {gcn_test_acc*100:.2f}% | 'f'Test Recall Pos: {gcn_test_rec_pos*100:.2f}% | Test Recall Neg: {gcn_test_rec_neg*100:.2f}%\n')
t1 = time.time()
print(t1-t0)
torch.save(gcn.state_dict(), f'/Users/aina/Desktop/TFG/codi/resultats/multilayer/gcn.pt')

plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in gcn_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/resultats/multilayer/gcn_loss.png', dpi=300, bbox_inches='tight')
plt.show()

t0 = time.time()
gin = GNN.GIN(dim_h=128)
print('GIN:')
gin, gin_losses = GNN.train(gin, train_loader, val_loader, epochs=100)
gin_test_loss, gin_test_acc, gin_test_rec_pos, gin_test_rec_neg = GNN.test(gin, test_loader)
print(f'Test Loss: {gin_test_loss:.2f} | Test Acc: {gin_test_acc*100:.2f}% | 'f'Test Recall Pos: {gin_test_rec_pos*100:.2f}% | Test Recall Neg: {gin_test_rec_neg*100:.2f}%\n')
t1 = time.time()
print(t1-t0)
torch.save(gin.state_dict(), f'/Users/aina/Desktop/TFG/codi/resultats/multilayer/gin.pt')

plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in gin_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/resultats/multilayer/gin_loss.png', dpi=300, bbox_inches='tight')
plt.show()


t0 = time.time()
gat = GNN.GAT(dim_in=7, dim_h=128, dim_out=2)
print('GAT:')
gat, gat_losses = GNN.train(gat, train_loader, val_loader, epochs=100)
gat_test_loss, gat_test_acc, gat_test_rec_pos, gat_test_rec_neg = GNN.test(gat, test_loader)
print(f'Test Loss: {gat_test_loss:.2f} | Test Acc: {gat_test_acc*100:.2f}% | 'f'Test Recall Pos: {gat_test_rec_pos*100:.2f}% | Test Recall Neg: {gat_test_rec_neg*100:.2f}%\n')
t1 = time.time()
print(t1-t0)
torch.save(gat.state_dict(), f'/Users/aina/Desktop/TFG/codi/resultats/multilayer/gat.pt')

plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in gat_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/resultats/multilayer/gat_loss.png', dpi=300, bbox_inches='tight')
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------

train_losses_all = {
    'GCN': [round(t.item(), 3) for t in gcn_losses],
    'GIN': [round(t.item(), 3) for t in gin_losses],
    'GAT': [round(t.item(), 3) for t in gat_losses]
}

test_metrics_all = {
    'GCN': {
        'test_loss': round(gcn_test_loss.item(), 3),
        'test_acc': gcn_test_acc,
        'test_rec_pos': gcn_test_rec_pos,
        'test_rec_neg': gcn_test_rec_neg
    },
    'GIN': {
        'test_loss': round(gin_test_loss.item(), 3),
        'test_acc': gin_test_acc,
        'test_rec_pos': gin_test_rec_pos,
        'test_rec_neg': gin_test_rec_neg
    },
    'GAT': {
        'test_loss': round(gat_test_loss.item(), 3),
        'test_acc': gat_test_acc,
        'test_rec_pos': gat_test_rec_pos,
        'test_rec_neg': gat_test_rec_neg
    }
}

funcions.guardar_resultats(train_losses_all, test_metrics_all, '/Users/aina/Desktop/TFG/codi/resultats/multilayer/resultats_multilayer.csv')


