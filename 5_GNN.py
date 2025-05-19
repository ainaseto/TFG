
import torch
torch.manual_seed(0)
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool
from sklearn.metrics import confusion_matrix

#--------------------------------------------------------------------------------------------------------------------------------------------------  

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dataset, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = h.relu()
        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return F.log_softmax(h, dim=1)

#--------------------------------------------------------------------------------------------------------------------------------------------------  

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dataset, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)

#--------------------------------------------------------------------------------------------------------------------------------------------------  

class GAT(torch.nn.Module):
    def __init__(self, dataset, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_h, heads=1)
        self.lin = Linear(dim_h, dim_out)

    def forward(self, x, edge_index, batch):
        # Capes GAT
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)

        # Pooling a nivell de graf
        hG = global_mean_pool(h, batch)

        # Classificador
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)

        return F.log_softmax(h, dim=1)
    
#--------------------------------------------------------------------------------------------------------------------------------------------------  

def train(model, loader, val_loader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        rec_pos = 0  
        rec_neg = 0 
        val_loss = 0
        val_acc = 0
        val_rec_pos = 0  
        val_rec_neg = 0 

        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            rec_pos += recall_positiu(out.argmax(dim=1), data.y) / len(loader)
            rec_neg += recall_negatiu(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()
        train_losses.append(total_loss / len(loader))

        # Validation
        val_loss, val_acc, val_rec_pos, val_rec_neg = test(model, val_loader)

        # Print metrics every 20 epochs
        if(epoch % 20 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | '
                  f'Train Recall Pos: {rec_pos*100:.2f}% | Train Recall Neg: {rec_neg*100:.2f}% | '
                  f'Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}% | '
                  f'Val Recall Pos: {val_rec_pos*100:.2f}% | Val Recall Neg: {val_rec_neg*100:.2f}%')
    return model, train_losses


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    rec_pos = 0 
    rec_neg = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
        rec_pos += recall_positiu(out.argmax(dim=1), data.y) / len(loader)
        rec_neg += recall_negatiu(out.argmax(dim=1), data.y) / len(loader)
    return loss, acc, rec_pos, rec_neg


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def recall_positiu(pred_y, y):
    cm = confusion_matrix(y.numpy(), pred_y.numpy(), labels=[0, 1])
    TP = cm[1, 1]
    FN = cm[1, 0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def recall_negatiu(pred_y, y):
    cm = confusion_matrix(y.numpy(), pred_y.numpy(), labels=[0, 1])
    TN = cm[0, 0]
    FP = cm[0, 1]
    return TN / (TN + FP) if (TN + FP) > 0 else 0
