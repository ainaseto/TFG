
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
adj_all = np.load('/Users/aina/Desktop/TFG/codi/xarxes/multiplex.npy')
print(f"Mida de adj_all: {adj_all.shape}")  # (270, 152, 152)

x_all = np.load('/Users/aina/Desktop/TFG/codi/embeddings/embeddings_multiplex.npy')
print(f"Mida de x_all: {x_all.shape}")  # (270, 76, 7)
x_conc = np.concatenate([x_all, x_all], axis=1)
x_concatenat = np.concatenate([x_conc, x_all], axis=1)
print(f"Mida de x: {x_concatenat.shape}")  # (270, 228, 7)

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

dataset = funcions.smote_graph_level(data_list, target_class=0, target_total=199)

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

t0 = time.time() 
gcn = GNN.GCN(dataset, dim_h = 128)    
print('GCN:')
gcn, gcn_losses = GNN.train(gcn, train_loader, val_loader, epochs=100)
gcn_test_loss, gcn_test_acc, gcn_test_rec_pos, gcn_test_rec_neg = GNN.test(gcn, test_loader)
print(f'Test Loss: {gcn_test_loss:.2f} | Test Acc: {gcn_test_acc*100:.2f}% | '
      f'Test Recall Pos: {gcn_test_rec_pos*100:.2f}% | Test Recall Neg: {gcn_test_rec_neg*100:.2f}%\n') 
t1 = time.time()
print(t1-t0)
torch.save(gcn.state_dict(), f'/Users/aina/Desktop/TFG/codi/resultats/multiplex/gcn.pt')

plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in gcn_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/resultats/multiplex/gcn_loss.png', dpi=300, bbox_inches='tight')
plt.show()

t0 = time.time() 
gin = GNN.GIN(dataset, dim_h=128)
print('GIN:')
gin, gin_losses = GNN.train(gin, train_loader, val_loader, epochs=100)
gin_test_loss, gin_test_acc, gin_test_rec_pos, gin_test_rec_neg = GNN.test(gin, test_loader)
print(f'Test Loss: {gin_test_loss:.2f} | Test Acc: {gin_test_acc*100:.2f}% | '
      f'Test Recall Pos: {gin_test_rec_pos*100:.2f}% | Test Recall Neg: {gin_test_rec_neg*100:.2f}%\n') 
t1 = time.time()
print(t1-t0)
torch.save(gcn.state_dict(), f'/Users/aina/Desktop/TFG/codi/resultats/multiplex/gin.pt')

plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in gin_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/resultats/multiplex/gin_loss.png', dpi=300, bbox_inches='tight')
plt.show()
 
t0 = time.time() 
gat = GNN.GAT(dataset, dim_in=7, dim_h=128, dim_out=2)
print('GAT:')
gat, gat_losses = GNN.train(gat, train_loader, val_loader, epochs=100)
gat_test_loss, gat_test_acc, gat_test_rec_pos, gat_test_rec_neg = GNN.test(gat, test_loader)
print(f'Test Loss: {gat_test_loss:.2f} | Test Acc: {gat_test_acc*100:.2f}% | '
      f'Test Recall Pos: {gat_test_rec_pos*100:.2f}% | Test Recall Neg: {gat_test_rec_neg*100:.2f}%\n') 
t1 = time.time()
print(t1-t0)
torch.save(gcn.state_dict(), f'/Users/aina/Desktop/TFG/codi/resultats/multiplex/gat.pt')

plt.figure(figsize=(8, 5))
plt.plot([loss.detach().item() for loss in gat_losses], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/aina/Desktop/TFG/codi/resultats/multiplex/gat_loss.png', dpi=300, bbox_inches='tight')
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

funcions.guardar_resultats(train_losses_all, test_metrics_all, '/Users/aina/Desktop/TFG/codi/resultats/multiplex/resultats_multiplex.csv')

