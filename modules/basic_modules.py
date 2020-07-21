import torch
import torch.nn as nn
import copy

from modules.basic_operations import *
from modules.genotypes import OPS_PRIMITIVES, COMBINE_MED


class MixedEdge(nn.Module):

    def __init__(self, C, stride, comb_med):
        super(MixedEdge, self).__init__()
        self._opers = nn.ModuleList()
        for name in OPS_PRIMITIVES:
            op = CANDIDATES[name](C, stride, True)
            self._opers.append(op)
        self._comb_med = comb_med

    def forward(self, x, ops_binaries):
        output = []
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                m_oi = self._opers[idx](x)
                output.append(m_oi*bin)
            else:
                m_oi = self._opers[idx](x)
                output.append(m_oi.detach()*bin)
        output = torch.stack(output, dim=0)
        output = COMBINE_MED[self._comb_med](output)
        return output

    def set_edge_ops(self, ops_binaries, comb_med):
        assert ops_binaries.size(-1) == 2, 'Wrong input for alphas'
        opers = []
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                opers.append(self._opers[idx])
        if len(opers) == 0:
            opers.append(self._opers[-1])
        return Edge(opers, comb_med)


class Edge(nn.Module):

    def __init__(self, opers, comb_med):
        super(Edge, self).__init__()
        self._opers = nn.ModuleList(opers)
        self._comb_med = comb_med

    def forward(self, x, *arg):
        output = []
        for op in self._opers:
            output.append(op(x))
        output = torch.stack(output, dim=0)
        output = COMBINE_MED[self._comb_med](output)
        return output


class Cell(nn.ModuleList):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, combine_method):
        super(Cell, self).__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier
        self._combine_method = combine_method

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._edges = nn.ModuleList()
        self._compile(C, reduction, combine_method)

    def _compile(self, C, reduction, combine_method):
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedEdge(C, stride, combine_method)
                self._edges.append(op)
        self.ops_alphas = None

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._edges[offset + j](h, self.ops_alphas[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

    def generate_rand_alphas(self, keep_prob=0.5):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_op = len(OPS_PRIMITIVES)
        ops_alps = torch.rand(k, num_op).bernoulli_(keep_prob)
        self.ops_alphas = ops_alps
        return self.ops_alphas

    def set_edge_fixed(self, ops_matrix):
        for idx, edg in enumerate(self._edges):
            self._edges[idx] = edg.set_edge_ops(ops_matrix[idx], self._combine_method)
