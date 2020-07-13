import torch
import torch.nn as nn
import copy

from modules.basic_modules import Cell


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False

        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = s1.mean(3, keepdim=False).mean(2, keepdim=False) # GAP
        logits = self.classifier(out)
        return logits

    def get_sub_net(self, ops_alphas, att_alphas):
        dim = len(ops_alphas.size())
        assert dim in [2, 3], 'Does not support this case!!!'
        share = dim == 2
        subnet = copy.deepcopy(self)
        if share:
            for c in subnet.cells:
                c.set_edge_fixed(ops_alphas, att_alphas)
        else:
            assert ops_alphas.size(0) == self._layers, 'Not enough alphas for each cell'
            for i, c in enumerate(subnet.cells):
                c.set_edge_fixed(ops_alphas[i], att_alphas[i])
        return subnet

    def generate_cell_alphas(self, is_infer=False):
        ops_alphas, _ = self.cells[0].generate_rand_alphas(is_infer)
        for c in self.cells:
            c.ops_alphas = ops_alphas
            c.att_alphas = torch.zeros(14, 3)
        # for c in self.cells:
        #     c.generate_rand_alphas()
