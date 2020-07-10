import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


CANDIDATES = {
    'zero': lambda C, stride, affine: Zero(stride=stride),
    'sep_conv': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'identity': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C),
    'channel_attn': lambda C, stride, affine: ChannelAttention(C, affine=affine),
    'spatial_attn': lambda C, stride, affine: SpatialAttention(affine=affine),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(OrderedDict([
            ('act', nn.ReLU(inplace=False)),
            ('conv', nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)),
            ('bn', nn.BatchNorm2d(C_out, affine=affine)),
        ]))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(OrderedDict([
            ('act1', nn.ReLU(inplace=False)),
            ('depth_conv1', nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)),
            ('point_conv1', nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False)),
            ('bn1', nn.BatchNorm2d(C_in, affine=affine)),
            ('act2', nn.ReLU(inplace=False)),
            ('depth_conv2', nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False)),
            ('point_conv2', nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)),
            ('bn2', nn.BatchNorm2d(C_out, affine=affine)),
        ]))

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
        else:
            padding = torch.FloatTensor(n, c, h, w).fill_(0)
        return padding


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.point_conv1 = nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
        self.point_conv2 = nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.point_conv1(x), self.point_conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ChannelAttention(nn.Module):

    def __init__(self, C_in, affine=True, reduce=16):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(C_in, C_in//reduce, bias=affine)),
            ('relu', nn.ReLU(inplace=False)),
            ('fc2', nn.Linear(C_in//reduce, C_in, bias=affine)),
        ]))

    def forward(self, x):
        mean = x.mean(3, keepdim=False).mean(2, keepdim=False) # global average pool
        max = x.max(3, keepdim=False)[0].max(2, keepdim=False)[0] # global max pool
        mask = torch.sigmoid(self.fc(mean) + self.fc(max)).unsqueeze(2).unsqueeze(3)
        return x*mask.expand_as(x)


class SpatialAttention(nn.Module):

    def __init__(self, affine=True, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1)//2, bias=False)),
            ('bn', nn.BatchNorm2d(1, affine=affine)),
        ]))

    def forward(self, x):
        mean = x.mean(1, keepdim=True) # average pooling over channel dimension
        max = x.max(1, keepdim=True)[0] # max pooling over channel dimension
        feat = torch.cat((mean, max), dim=1)
        mask = torch.sigmoid(self.fc(feat))
        return x*mask.expand_as(x)
