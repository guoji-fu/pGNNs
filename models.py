import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge
from src.pgnn_conv import pGNNConv
from src.gpr_conv import GPR_prop


class pGNNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_hid=16, 
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=True):
        super(pGNNNet, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)        
        return F.log_softmax(x, dim=1)


class MLPNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5):
        super(MLPNet, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)


class GCNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5,
                 cached=True):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid, cached=cached)
        self.conv2 = GCNConv(num_hid, out_channels, cached=cached)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GCN_Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 num_hid=16):
        super(GCN_Encoder, self).__init__()
        self.conv = GCNConv(in_channels, num_hid, cached=True)
        self.prelu = torch.nn.PReLU(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class SGCNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 K=2,
                 cached=True):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=8,
                 num_heads=8,
                 dropout=0.6,
                 concat=False):

        super(GATNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, num_hid, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(num_heads * num_hid, out_channels, heads=1, concat=concat, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)


class JKNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0,
                 num_layes=4,
                 dropout=0.5):
        super(JKNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, num_hid)
        self.lin1 = torch.nn.Linear(num_hid, out_channels)
        self.one_step = APPNP(K=K, alpha=alpha)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=num_hid,
                                   num_layers=num_layes)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index, edge_weight)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0.1,
                 dropout=0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GPRGNNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_hid,
                 ppnp,
                 K=10,
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None,
                 dprate=0.5,
                 dropout=0.5):
        super(GPRGNNNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)