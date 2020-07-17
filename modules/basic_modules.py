import torch
import torch.nn as nn
import copy

from modules.basic_operations import *
from modules.genotypes import OPS_PRIMITIVES
# from utils import binarize


class MixedEdge(nn.Module):

    def __init__(self, C, stride):
        super(MixedEdge, self).__init__()
        self._opers = nn.ModuleList()
        for name in OPS_PRIMITIVES:
            op = CANDIDATES[name](C, stride, True)
            self._opers.append(op)

    def forward(self, x, ops_binaries):
        output = []
<<<<<<< HEAD
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                m_oi = self._opers[idx](x)
                output.append(m_oi*bin)
            else:
                m_oi = self._opers[idx](x)
                output.append(m_oi.detach()*bin)
        output = sum(output)
=======
        for idx, bin_ in enumerate(ops_binaries):
            m_oi = self._opers[idx](x)
            if bin_:
                output.append(m_oi * bin_)
            # else:
            #     output.append(m_oi.detach() * bin_)
            # if bin == 1:
            #     m_oi = self._opers[idx](x)
            #     output.append(m_oi)
            # else:
            #     m_oi = self._opers[idx](x)
            #     output.append(m_oi.detach()*bin)
        # att_binaries *= ops_binaries[1]
        # att_out = []
        # for idx, bin in enumerate(att_binaries):
        #     if bin == 1:
        #         a_oi = self._attns[idx](output[1])
        #         att_out.append(a_oi)
        #     else:
        #         a_oi = self._attns[idx](output[1])
        #         att_out.append(a_oi.detach()*bin)
        # output[1] = sum(att_out)
        # output = sum(output)
        output = torch.mean(torch.stack(output), dim=0)
>>>>>>> origin/master
        return output

    def set_edge_ops(self, ops_binaries):
        assert ops_binaries.size(-1) == 2, 'Wrong input for alphas'
        opers = []
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                opers.append(self._opers[idx])
        if len(opers) == 0:
            opers.append(self._opers[-1])
        return Edge(opers)


class Edge(nn.Module):

    def __init__(self, opers):
        super(Edge, self).__init__()
        self._opers = nn.ModuleList(opers)

    def forward(self, x, *arg):
        output = []
        for op in self._opers:
            output.append(op(x))
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
        self.ops_alphas = None

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
<<<<<<< HEAD
            s = sum(self._edges[offset + j](h, self.ops_alphas[offset + j]) for j, h in enumerate(states))
=======
            s = sum(
                self._edges[offset + j](
                    h, 
                    self.ops_alphas[offset + j],
                    self.att_alphas[offset + j]
                ) for j, h in enumerate(states)
            )
>>>>>>> origin/master
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

<<<<<<< HEAD
    def generate_rand_alphas(self, keep_prob=0.5):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_op = len(OPS_PRIMITIVES)
        ops_alps = torch.rand(k, num_op).bernoulli_(keep_prob)
        self.ops_alphas = ops_alps
        return self.ops_alphas
=======
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
>>>>>>> origin/master

    def set_edge_fixed(self, ops_matrix):
        for idx, edg in enumerate(self._edges):
            self._edges[idx] = edg.set_edge_ops(ops_matrix[idx])
