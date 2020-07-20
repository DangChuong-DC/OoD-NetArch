from collections import namedtuple


OPS_PRIMITIVES = [
    'identity',
    'sep_conv',
    'zero',
]


# ATT_PRIMITIVES = [
#     'channel_attn',
#     'spatial_attn',
#     'identity',
# ]


COMBINE_MED = {
    'mean': lambda x, dim=0: torch.mean(x, dim=dim),
    'sum': lambda x, dim=0: torch.sum(x, dim=dim),
}
