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


class LinearCosine(torch.nn.Module):

    def __init__(self, in_features, out_features, w_init_fn=nn.init.kaiming_normal_, use_scale=True):
        super(LinearCosine, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        w_init_fn(self.weight)
        self.use_scale = use_scale
        self.out_features = out_features
        if self.use_scale:
            self.fc_scale = nn.Linear(in_features, 1, bias=False)
            self.bn_scale = nn.BatchNorm1d(1)

    def forward(self, x, get_cosine=False):
        x_normalized = F.normalize(x, dim=-1)
        w_normalized = F.normalize(self.weight)
        out = F.linear(x_normalized, w_normalized)
        cosine = out[:]
        if self.use_scale:
            tmp = self.fc_scale(x)
            sh = tmp.size()
            scale = torch.exp(self.bn_scale(tmp.view([-1, 1])))
            out = out * scale.view(sh)
        if get_cosine:
            return out, cosine
        else:
            return out


class Conv1x1Cosine(torch.nn.Module):
    def __init__(self, in_features, out_features, w_init_fn=nn.init.kaiming_normal_, use_scale=True):
        super(Conv1x1Cosine, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        w_init_fn(self.weight)
        self.use_scale = use_scale
        self.out_features = out_features
        if self.use_scale:
            self.fc = nn.Conv2d(in_features, 1, 1, bias=False)
            self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x_tmp = x.transpose(1, 3)
        x_shape = x_tmp.shape
        x_tmp = x_tmp.reshape([-1, x_shape[3]])
        x_normalized = F.normalize(x_tmp, dim=-1)
        w_normalized = F.normalize(self.weight)
        out = x_normalized.matmul(w_normalized.transpose(0, 1))
        self.out = out.reshape([x_shape[0], x_shape[1], x_shape[2], self.out_features]).transpose(1, 3)
        if self.use_scale:
            self.scale = torch.exp(self.bn(self.fc(x)))
            out = self.out * self.scale
        return out

class Conv2dCosine(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2dCosine, self).__init__()
        self.linear = LinearCosine(in_channels*kernel_size**2, out_channels)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def compute_shape(self, x):
        h = (x.shape[2] + 2 * self.padding - 1 * (self.kernel_size - 1) - 1) // self.stride + 1
        w = (x.shape[3] + 2 * self.padding - 1 * (self.kernel_size - 1) - 1) // self.stride + 1
        return h, w

    def operate(self, x_unf):
        feat = x_unf.transpose(1, 2)
        res = self.linear(feat)
        return res.transpose(1, 2)

    def forward(self, x):
        x_unf = torch.nn.functional.unfold(
            x, self.kernel_size, self.dilation, self.padding, self.stride
        )
        h, w = self.compute_shape(x)
        result = self.operate(x_unf).view(x.shape[0], -1, h, w)
        return result
