import torch
import argparse
import time

from data_proc import load_data
from models import *
import torch_geometric.transforms as T

def build_model(args, num_features, num_classes):
    if args.model == 'pgnn':
        model = pGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout)
    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'sgc':
        model = SGCNet(in_channels=num_features,
                        out_channels=num_classes,
                        K=args.K)
    elif args.model == 'gat':
        model = GATNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        num_heads=args.num_heads,
                        dropout=args.dropout)
    elif args.model == 'jk':
        model = JKNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        K=args.K,
                        alpha=args.alpha,
                        dropout=args.dropout)
    elif args.model == 'appnp':
        model = APPNPNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            K=args.K,
                            alpha=args.alpha,
                            dropout=args.dropout)
    elif args.model == 'gprgnn':
        model = GPRGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            ppnp=args.ppnp,
                            K=args.K,
                            alpha=args.alpha,
                            Init=args.Init,
                            Gamma=args.Gamma,
                            dprate=args.dprate,
                            dropout=args.dropout)
    return model

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(args):
    print(args)
    data, num_features, num_classes = load_data(args, rand_seed=2021)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for run in range(args.runs):
        model = build_model(args, num_features, num_classes)
        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        
        t1 = time.time()
        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs+1):
            train(model, optimizer, data)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        t2 = time.time()
        # print('{}, {}, Accuacy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, test_acc, t2-t1))
        results.append(test_acc)
    results = 100 * torch.Tensor(results)
    print(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='cora',                    
                        help='Input graph.')
    parser.add_argument('--train_rate', 
                        type=float, 
                        default=0.025,
                        help='Training rate.')
    parser.add_argument('--val_rate', 
                        type=float, 
                        default=0.025,
                        help='Validation rate.')
    parser.add_argument('--model',
                        type=str,
                        default='pgnn',
                        choices=['pgnn', 'mlp', 'gcn', 'cheb', 'sgc', 'gat', 'jk', 'appnp', 'gprgnn'],
                        help='GNN model')
    parser.add_argument('--runs',
                        type=int,
                        default=10,
                        help='Number of repeating experiments.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid', 
                        type=int, 
                        default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--mu', 
                        type=float, 
                        default=0.1,
                        help='mu.')
    parser.add_argument('--p', 
                        type=float, 
                        default=2,
                        help='p.')
    parser.add_argument('--K', 
                        type=int, 
                        default=2,
                        help='K.')
    parser.add_argument('--num_heads', 
                        type=int, 
                        default=8,
                        help='Number of heads.')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.0,
                        help='alpha.')
    parser.add_argument('--Init', 
                        type=str, 
                        default='PPR',
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'])
    parser.add_argument('--Gamma', 
                        default=None)
    parser.add_argument('--ppnp', 
                        type=str,
                        default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--dprate',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())
