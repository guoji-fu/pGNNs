from typing import Optional, Tuple

from torch._C import BenchmarkExecutionStats
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

from torch_geometric.utils import num_nodes
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros


@torch.jit._overload
def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=False, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t, deg_inv_sqrt

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        return edge_index, edge_weight, deg_inv_sqrt


def calc_M(f, edge_index, edge_weight, deg_inv_sqrt, num_nodes, mu, p):
        if isinstance(edge_index, SparseTensor):
            row, col, edge_weight = edge_index.coo()
        else:
            row, col = edge_index[0], edge_index[1]

        ## calculate M
        graph_grad = torch.pow(edge_weight, 0.5).view(-1, 1) * (deg_inv_sqrt[row].view(-1, 1) * f[row] - deg_inv_sqrt[col].view(-1, 1) * f[col])
        graph_grad = torch.pow(torch.norm(graph_grad, dim=1), p-2)
        M = edge_weight * graph_grad
        M.masked_fill_(M == float('inf'), 0)
        alpha = (deg_inv_sqrt.pow(2) * scatter_add(M, col, dim=0, dim_size=num_nodes) + (2*mu)/p).pow(-1)
        beta = 4*mu / p * alpha
        M_ = alpha[row] * deg_inv_sqrt[row] * M * deg_inv_sqrt[col]
        return M_, beta


class pGNNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor, Tensor]]
    _cached_adj_t: Optional[Tuple[SparseTensor, Tensor]]

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 mu: float,
                 p: float,
                 K: int,
                 improved: bool = False, 
                 cached: bool = False,
                 add_self_loops: bool = False, 
                 normalize: bool = True,
                 bias: bool = True, 
                 return_M_: bool = False,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(pGNNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.p = p
        self.K = K
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.return_M_ = return_M_

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)

        if return_M_:
            self.new_edge_attr = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        num_nodes = x.size(self.node_dim)
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, deg_inv_sqrt)
                else:
                    edge_index, edge_weight, deg_inv_sqrt = cache[0], cache[1], cache[2]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = (edge_index, deg_inv_sqrt)
                else:
                    edge_index, deg_inv_sqrt = cache[0], cache[1]

        out = x
        for _ in range(self.K):
            edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
            out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * x
            
        out = self.lin1(out)

        if self.return_M_:
            self.new_edge_attr = edge_attr
            
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
