import torch
import torch.nn as nn

from modules.basic_operations import *
from modules.genotypes import OPS_PRIMITIVES, ATT_PRIMITIVES
from utils import binarize


class MixedEdge(nn.Module):

    def __init__(self, C, stride):
        super(MixedEdge, self).__init__()
        self._opers = nn.ModuleList()
        self._attns = nn.ModuleList()
        for name in OPS_PRIMITIVES:
            op = CANDIDATES[name](C, stride, True)
            self._opers.append(op)
        for name in ATT_PRIMITIVES:
            att = CANDIDATES[name](C, 1, True)
            self._attns.append(att)

    def forward(self, x, ops_binaries, att_binaries):
        output = []
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                m_oi = self._opers[idx](x)
                output.append(m_oi)
            else:
                m_oi = self._opers[idx](x)
                output.append(m_oi.detach()*bin)
        att_binaries *= ops_binaries[1]
        att_out = []
        for idx, bin in enumerate(att_binaries):
            if bin == 1:
                a_oi = self._attns[idx](output[1])
                att_out.append(a_oi)
            else:
                a_oi = self._attns[idx](output[1])
                att_out.append(a_oi.detach()*bin)
        output[1] = sum(att_out)
        output = sum(output)
        return output

    def set_edge_ops(self, ops_binaries, att_binaries):
        assert ops_binaries.size(-1) == 2 and att_binaries.size(-1) == 2, 'Wrong input for alphas'
        opers = []
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                opers.append(self._opers[idx])
        if len(opers) == 0:
            opers.append(self._opers[-1])

        att_binaries *= ops_binaries[1]
        atten = None
        for idx, bin in enumerate(att_binaries):
            if bin == 1:
                atten = self._attns[idx]
        return Edge(opers, atten)


class Edge(nn.Module):

    def __init__(self, opers, atten):
        super(Edge, self).__init__()
        self._opers = nn.ModuleList(opers)
        self._attns = atten

    def forward(self, x, *arg):
        output = []
        for op in self._opers:
            output.append(op(x))
        if self._attns is not None:
            output[-1] = self._attns(output[-1])
        output = sum(output)
        return output


class Cell(nn.ModuleList):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._edges = nn.ModuleList()
        self._compile(C, reduction)

    def _compile(self, C, reduction):
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedEdge(C, stride)
                self._edges.append(op)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._edges[offset + j](
                    h, 
                    self.ops_alphas[offset + j],
                    self.att_alphas[offset + j]
                ) for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

    def generate_rand_alphas(self, is_infer=False):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_op = len(OPS_PRIMITIVES)
        num_att = len(ATT_PRIMITIVES)
        ops_alp = torch.zeros(k, num_op)
        att_alp = torch.zeros(k, num_att)
        if is_infer:
            ops_alp = torch.ones(k, num_op)
            return ops_alp, att_alp
        else:
            ops_alphas = binarize(ops_alp, 2)
            att_alphas = binarize(att_alp, 1)
            return ops_alphas, att_alphas
        # self.ops_alphas = torch.ones_like(self.ops_alphas)
        # self.att_alphas = torch.zeros_like(self.ops_alphas)
        # exit()

    def set_edge_fixed(self, ops_matrix, att_matrix):
        for idx, edg in enumerate(self._edges):
            self._edges[idx] = edg.set_edge_ops(ops_matrix[idx], att_matrix[idx])
