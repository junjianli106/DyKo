# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import torch.nn as nn
import torch.nn.functional as F

import sys
import os.path as osp
import numpy as np
import pandas as pd
import random
import h5py
import torch
from torch import Tensor

def posemb_sincos_2d(y, x, dim, device, dtype, temperature=10000):
    """
    Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py#L12
    """
    # y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))


def setup_device(no_cuda, cuda_id, verbose=True):
    device = 'cpu'
    if not no_cuda and torch.cuda.is_available():
        device = 'cuda' if cuda_id < 0 else 'cuda:{}'.format(cuda_id)
    if verbose:
        print('[setup] device: {}'.format(device))

    return device


# worker_init_fn = seed_worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# generator = g
def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def collect_tensor(collector, y, y_hat):
    if collector['y'] is None:
        collector['y'] = y
    else:
        collector['y'] = torch.cat([collector['y'], y], dim=0)

    if collector['y_hat'] is None:
        collector['y_hat'] = y_hat
    else:
        collector['y_hat'] = torch.cat([collector['y_hat'], y_hat], dim=0)

    return collector


def to_patient_data(df, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    return df.loc[df_idx, :]


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def coord_discretization(wsi_coord: Tensor):
    """
    Coordinate Discretization.
    If the value of coordinates is too large (such as 100,000), it will need super large space
    when computing the positional embedding of patch.
    """
    x, y = wsi_coord[:, 0].tolist(), wsi_coord[:, 1].tolist()
    sorted_x, sorted_y = sorted(list(set(x))), sorted(list(set(y)))  # remove duplicates and then sort
    xmap, ymap = {v: i for i, v in enumerate(sorted_x)}, {v: i for i, v in enumerate(sorted_y)}
    nx, ny = [xmap[v] for v in x], [ymap[v] for v in y]
    res = torch.tensor([nx, ny], dtype=wsi_coord[0].dtype, device=wsi_coord[0].device)
    return res.T


def to_relative_coord(wsi_coord: Tensor):
    ref_xy, _ = torch.min(wsi_coord, dim=-2)
    top_xy, _ = torch.max(wsi_coord, dim=-2)
    rect = top_xy - ref_xy
    ncoord = wsi_coord - ref_xy
    # print("To relative coordinates:", ref_xy, rect)
    return ncoord, ref_xy, rect


def rearrange_coord(wsi_coords, offset_coord=[1, 0], discretization=False):
    """
    wsi_coord (list(torch.Tensor)): list of all patch coordinates of one WSI.
    offset_coord (list): it is set as [1, 0] by default, which means putting WSIs horizontally.
    """
    assert isinstance(wsi_coords, list)
    ret = []
    off_coord = torch.tensor([offset_coord], dtype=wsi_coords[0].dtype, device=wsi_coords[0].device)
    top_coord = -1 * off_coord
    for coord in wsi_coords:
        if discretization:
            coord = coord_discretization(coord)
        new_coord, ref_coord, rect = to_relative_coord(coord)
        new_coord = top_coord + off_coord + new_coord
        top_coord = top_coord + off_coord + rect
        ret.append(new_coord)
    return ret


##################################################################
#
#                     Functionality: I/O
#
##################################################################
def print_config(config, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout

    print("**************** MODEL CONFIGURATION ****************", file=f)
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val), file=f)
    print("**************** MODEL CONFIGURATION ****************", file=f)

    if print_to_path is not None:
        f.close()


def print_metrics(metrics, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout

    print("**************** MODEL METRICS ****************", file=f)
    for key in sorted(metrics.keys()):
        val = metrics[key]
        for v in val:
            cur_key = key + '/' + v[0]
            keystr = "{}".format(cur_key) + (" " * (20 - len(cur_key)))
            valstr = "{}".format(v[1])
            if isinstance(v[1], list):
                valstr = "{}, avg/std = {:.5f}/{:.5f}".format(valstr, np.mean(v[1]), np.std(v[1]))
            print("{} -->   {}".format(keystr, valstr), file=f)
    print("**************** MODEL METRICS ****************", file=f)

    if print_to_path is not None:
        f.close()


def read_datasplit_npz(path: str):
    data_npz = np.load(path)

    pids_train = [str(s) for s in data_npz['train_patients']]
    pids_val = [str(s) for s in data_npz['val_patients']]
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test


def read_coords(path: str, dtype: str = 'torch'):
    r"""Read patch coordinates from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']

    with h5py.File(path, 'r') as hf:
        nfeats = hf['coords'][:]

    if isinstance(nfeats, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(nfeats)
    else:
        return nfeats


def read_nfeats(path: str, dtype: str = 'torch'):
    r"""Read node features from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            nfeats = hf['features'][:]
    elif ext == '.pt':
        nfeats = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise ValueError(f'not support {ext}')

    if isinstance(nfeats, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(nfeats)
    elif isinstance(nfeats, Tensor) and dtype == 'numpy':
        return nfeats.numpy()
    else:
        return nfeats


def save_prediction(pids, y_true, y_pred, save_path):
    r"""Save surival prediction.

    Args:
        y_true (Tensor or ndarray): true labels.
        y_pred (Tensor or ndarray): predicted values.
        save_path (string): path to save.

    If it is a discrete model:
        y: [B, 2] (col1: y_t, col2: y_c)
        y_hat: [B, BINS]
    else:
        y: [B, 1]
        y_hat: [B, 1]
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()

    print(y_pred.shape, y_true.shape)
    if y_true.shape[1] == 1:
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)
        df = pd.DataFrame({'patient_id': pids, 'pred': y_pred, 'true': y_true}, columns=['patient_id', 'true', 'pred'])
    elif y_true.shape[1] == 2:
        bins = y_pred.shape[1]
        y_t, y_e = y_true[:, [0]], 1 - y_true[:, [1]]
        survival = np.cumprod(1 - y_pred, axis=1)
        risk = np.sum(survival, axis=1, keepdims=True)
        arr = np.concatenate((y_t, y_e, risk, survival), axis=1)  # [B, 3+BINS]
        df = pd.DataFrame(arr, columns=['t', 'e', 'risk'] + ['surf_%d' % (_ + 1) for _ in range(bins)])
        df.insert(0, 'patient_id', pids)
    df.to_csv(save_path, index=False)
class GAPool(nn.Module):
    """
    GAPool: Global Attention Pooling
    """

    def __init__(self, in_dim, hid_dim, dropout=0.25):
        super(GAPool, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        emb = self.fc1(x)  # [B, N, d']
        scr = self.score(x)  # [B, N, d'] \in [0, 1]
        new_emb = emb.mul(scr)
        rep = self.fc2(new_emb)  # [B, N, 1]
        rep = torch.transpose(rep, 2, 1)
        attn = F.softmax(rep, dim=2)  # [B, 1, N]
        out = torch.matmul(attn, x).squeeze(1)  # [B, 1, d]
        return out, attn


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm=True):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=kernel_size,
                                                      stride=stride, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        return x


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


"""
Multi-Head Attention from 
https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
"""
from torch.nn.functional import *


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_raw: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim: (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:

            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw

            # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

from torch.nn import Module
import torch.nn.init as init
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)

import torch
from torch import Tensor
# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter



class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_raw=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask)

@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def topk_keep_order(score, k):
    """
    The results won't destroy the original order.

    score: [B, N].
    """
    _, sorted_idx = torch.topk(score, k)
    idx, _ = torch.sort(sorted_idx)
    new_score = torch.gather(score, -1, idx)
    return new_score, idx


def generate_mask(idxs, n):
    sz = list(idxs.size())
    res = torch.zeros(sz[:-1] + [n]).bool().to(idxs.device)
    res.scatter_(-1, idxs, True)
    return res


def extend_mask(mask, scale=4):
    """
    mask = [..., N]
    """
    if scale == 1:
        return mask

    mask_size = list(mask.size())
    tmp = mask.unsqueeze(-1)
    tensor_ex = torch.ones(mask_size + [scale * scale]).to(mask.device)
    res = tensor_ex * tmp
    res = res.view(mask_size[:-1] + [-1]).bool()
    return res


def square_seq(x):
    """
    x: [1, N, d] -> [1, L^2, d]
    """
    B, N = x.shape[0], x.shape[1]
    L = int(np.ceil(np.sqrt(N)))
    len_padding = L * L - N
    x = torch.cat([x, x[:, :len_padding, :]], dim=1)  # square [1, L^2, 512]
    x = x.reshape(B, L, L, -1)
    return x


def square_align_seq(x1, x2, scale=4):
    """
    x1: [1, 16N, d1], level = 1
    x2: [1,   N, d2], level = 2
    """
    B, N = x2.shape[0], x2.shape[1]
    D1, D2 = x1.shape[2], x2.shape[2]
    L = int(np.ceil(np.sqrt(N)))
    len_padding = L * L - N

    x2 = torch.cat([x2, x2[:, :len_padding, :]], dim=1)  # [1, L^2, 512]
    x1 = torch.cat([x1, x1[:, :(scale * len_padding), :]], dim=1)  # [1, (4L)^2, 512]

    x2 = x2.reshape(B, L, L, -1)
    # spatial alignment
    x1 = x1.view(B, L, L, scale, scale, -1)
    x1 = x1.transpose(3, 4).reshape(B, L, L * scale, scale, -1)
    x1 = x1.transpose(2, 3).reshape(B, -1, L * scale, D1)

    return x1, x2


def sequence2square(x, s):
    """
    [B, N, C] -> [B*(N/s^2), C, s, s]
    """
    size = x.size()
    assert size[1] % (s * s) == 0
    L = size[1] // (s * s)
    x = x.view(-1, s, s, size[2])
    x = x.permute(0, 3, 1, 2)
    return x, L


def square2sequence(x, L):
    """
    [B*L, C, s, s] -> [B, L*s*s, c]
    """
    size = x.size()
    assert size[0] % L == 0
    x = x.view(size[0], size[1], -1)
    x = x.transpose(2, 1).view(size[0] // L, -1, size[1])
    return x


def posemb_sincos_2d(y, x, dim, device, dtype, temperature=10000):
    """
    Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py#L12
    """
    # y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def compute_pe(coord: torch.Tensor, ndim=384, step=1, device='cpu', dtype=torch.float):
    # coord: [B, N, 2]
    assert coord.shape[0] == 1
    coord = coord.squeeze(0)
    ncoord, ref_xy, rect = to_relative_coord(coord)
    assert rect[0] % step == 0 and rect[1] % step == 0
    y = torch.div(ncoord[:, 1], step, rounding_mode='floor')
    x = torch.div(ncoord[:, 0], step, rounding_mode='floor')
    PE = posemb_sincos_2d(y, x, ndim, device, dtype)  # [N, ndim]
    PE = PE.unsqueeze(0)  # [1, N, ndim]
    return PE


def make_conv1d_layer(in_dim, out_dim, kernel_size=3, spatial_conv=True):
    conv1d_ksize = kernel_size if spatial_conv else 1
    p = (conv1d_ksize - 1) // 2
    return Conv1dPatchEmbedding(in_dim, out_dim, conv1d_ksize, stride=1, padding=p)


#####################################################################################
#
#    Functions/Classes for Patch Embedding, intended to
#    1. aggregate patches in a small field into regional features using 1D/2D Conv
#    2. reduce feature dimension
#
#####################################################################################
def make_embedding_layer(backbone: str, args):
    """
    backbone: ['conv1d', 'gapool', 'avgpool', 'capool', 'identity']
    """
    if backbone == 'conv1d':
        layer = Conv1dPatchEmbedding(args.in_dim, args.out_dim, args.ksize, stride=1, padding=(args.ksize - 1) // 2)
    elif backbone == 'gapool':
        layer = GAPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'avgpool':
        layer = AVGPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'capool':
        layer = CAPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'sconv':
        layer = SquareConvPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'identity':
        layer = IdentityPatchEmbedding(args.in_dim, args.out_dim)
    else:
        raise NotImplementedError(f'{backbone} has not implemented.')
    return layer


class IdentityPatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IdentityPatchEmbedding, self).__init__()
        if in_dim == out_dim:
            self.layer = nn.Identity()
        else:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            )

    def forward(self, x):
        out = self.layer(x)
        return out


class AVGPoolPatchEmbedding(nn.Module):
    """
    head layer (FC/Conv2D) + pooling Layer (avg pooling) for patch embedding.

    ksize = 1 -> head layer = FC
    ksize = 3 -> head layer = Conv2D

    Patch data with shape of [B, N, C]
    if scale = 1, then apply Conv2d with stride=1
        [B, N, C] -> [B, C, N] --conv1d--> [B, C', N]
    elif scale = 2/4, then apply Conv2d with stride=2
        [B, N, C] -> [B*(N/s^2), C, s, s] --conv2d--> [B*(N/s^2), C, 1, 1] -> [B, N/s^2, C]
    """

    def __init__(self, in_dim, out_dim, scale: int = 4, dw_conv=False, ksize=3, stride=1):
        super(AVGPoolPatchEmbedding, self).__init__()
        assert scale == 4, 'It only supports for scale = 4'
        assert ksize == 1 or ksize == 3, 'It only supports for ksize = 1 or 3'
        self.scale = scale
        self.stride = stride
        if scale == 4:
            # Conv2D on the grid of 4 x 4: stride=2 + ksize=3 or stride=1 + ksize=1/3
            assert (stride == 2 and ksize == 3) or (stride == 1 and (ksize == 1 or ksize == 3)), \
                'Invalid stride or kernel_size when scale=4'
            if dw_conv:
                self.conv = SeparableConvBlock(in_dim, out_dim, ksize, stride, norm=False)
            else:
                self.conv = nn.Conv2d(in_dim, out_dim, ksize, stride, padding=(ksize - 1) // 2)
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError()

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, N ,C]
        """
        x, L = sequence2square(x, self.scale)  # [B*N/16, C, 4, 4]
        x = self.conv(x)  # [B*N/16, C, 4/s, 4/s]
        x = square2sequence(x, L)  # [B, N/(s*s), C]
        x = self.norm(x)
        x = self.act(x)
        x, L = sequence2square(x, self.scale // self.stride)  # [B*N/16, C, 4/s, 4/s]
        x = self.pool(x)  # [B*N/16, C, 1, 1]
        x = square2sequence(x, L)  # [B, N/16, C]
        return x


class GAPoolPatchEmbedding(nn.Module):
    """
    head layer (FC/Conv2D) + pooling Layer (global-attention pooling) for patch embedding.

    ksize = 1 -> head layer = FC
    ksize = 3 -> head layer = Conv2D

    Global Attention Pooling for patch data with shape of [B, N, C].
    [B, N, C] -> [B, N/(scale^2), C']
    """

    def __init__(self, in_dim, out_dim, scale: int = 4, dw_conv: bool = False, ksize=3):
        super(GAPoolPatchEmbedding, self).__init__()
        assert scale == 4, 'It only supports for scale = 4'
        assert ksize == 1 or ksize == 3, 'It only supports for ksize = 1 or 3'
        self.scale = scale
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, ksize, 1, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 1, padding=(ksize - 1) // 2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.pool = GAPool(out_dim, out_dim, 0.0)

    def forward(self, x):
        # conv2d (strid=1) embedding (spatial continuity)
        x, L = sequence2square(x, self.scale)  # [B*N/(s^2), C, s, s]
        x = self.conv(x)  # [B*N/(s^2), C, s, s]
        x = square2sequence(x, L)  # [B, N, C]
        x = self.norm(x)
        x = self.act(x)

        # gapool
        sz = x.size()  # [B, N, C]
        x = x.view(-1, self.scale * self.scale, sz[2])  # [B*N/(scale^2), scale*scale, C]
        x, x_attn = self.pool(x)  # [B*N/(scale^2), C]
        x = x.view(sz[0], -1, sz[2])  # [B, N/(scale^2), C]
        return x


class CAPoolPatchEmbedding(nn.Module):
    """
    head layer (FC/Conv2D) + pooling Layer (cross-attention pooling) for patch embedding.

    ksize = 1 -> head layer = FC
    ksize = 3 -> head layer = Conv2D

    Patch Embedding guided by x5 patches
    """

    def __init__(self, in_dim, out_dim, scale: int = 4, dw_conv: bool = True, ksize=3):
        super(CAPoolPatchEmbedding, self).__init__()
        self.scale = scale
        assert scale != 1, "Please pass a scale larger than 1 for capool."
        # Conv1D-ksize_1 (= FC layer) for x5 patches to make dimension equal to the x20 patches
        self.conv_patch_x5 = Conv1dPatchEmbedding(in_dim, out_dim, 1, norm=False, activation=False)
        # Conv2D for x20 patches
        assert ksize == 1 or ksize == 3, 'It only supports for ksize=1/3 for embedding layer at scale=4'
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, ksize, 1, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 1, padding=(ksize - 1) // 2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.cross_att_pool = MultiheadAttention(embed_dim=out_dim, num_heads=4)

    def forward(self, x20, x5):
        # firstly reduce the dimension of x5
        x5 = self.conv_patch_x5(x5)

        # conv2d (strid=1) embedding (spatial continuity)
        x20, L = sequence2square(x20, self.scale)  # [B*N/(s^2), C, s, s]
        x20 = self.conv(x20)  # [B*N/(s^2), C', s, s]
        x20 = square2sequence(x20, L)  # [B, 16N, C']
        x20 = self.norm(x20)
        x20 = self.act(x20)

        assert x5.shape[1] == L  # N == L
        assert x20.shape[2] == x5.shape[2]
        # x5 patch guided pooling
        # [B, 16N, C]->[B*L, 16, C]->[16, B*L, C]
        x20 = x20.view(-1, self.scale * self.scale, x20.shape[2]).transpose(0, 1)
        # [B, N, C]->[B*N, 1, C]->[1, B*L, C]
        x5 = x5.view(-1, 1, x5.shape[2]).transpose(0, 1)

        # x: [1, B*L, C], x_attn: [B*L, num_heads, 1, 16]
        x, x_attn = self.cross_att_pool(x5, x20, x20)
        # [B*L, num_heads, 1, 16] -> [B, L, num_heads, 1, 16] -> [B, L, num_heads, 16]
        x_attn = x_attn.view(-1, L, x_attn.shape[1], x_attn.shape[2] * x_attn.shape[3])
        x5 = x5.view(-1, L, x5.shape[2])
        x = x.squeeze().view(-1, L, x20.shape[2])
        return x, x_attn, x5


class SquareConvPatchEmbedding(nn.Module):
    """
    Conv for patch data with shape of [B, N, C].
    The convolution is paticularly applied to the squared sequences (Refer to TransMIL).

    We use [conv2d + avgpool] as its architecture as same as the ConvPatchEmbedding layer to
    keep the total number of learnable parameters same.

    [B, N, C] -> [B, C, L, L] --conv2d--> [B, C', L/2, L/2] --avgpool--> [B, C', L/4, L/4]
    """

    def __init__(self, in_dim, out_dim, scale: int = 4, dw_conv=False, ksize=3):
        super(SquareConvPatchEmbedding, self).__init__()
        assert scale == 4, 'It only used in x20 magnification'
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, ksize, 2, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 2, padding=(ksize - 1) // 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, N ,C]
        """
        H = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        # [B, L*L, C]
        B, _, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, _H, _W)  # [B, C, L, L]
        x = self.conv(cnn_feat)  # [B, C', L/2, L/2]
        _, C, _H, _W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, L/2*L/2, C']

        # [B, L/2*L/2, C']
        x = self.norm(x)
        x = self.act(x)
        cnn_feat = x.transpose(1, 2).view(B, C, _H, _W)  # [B, C', L/2, L/2]
        x = self.pool(cnn_feat)  # [B, C', L/4, L/4]
        x = x.flatten(2).transpose(1, 2)  # [B, L/2*L/2, C']
        return x


from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce


# helper functions

def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


# main attention class

class NystromAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:, :, 0].unsqueeze(-2) @ attn2
            attn1 = (attn1 @ attn3)

            return out, attn1[:, :, 0, -n + 1:]

        return out


# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Nystromformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            attn_values_residual=True,
            attn_values_residual_conv_kernel=33,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim=dim, dim_head=dim_head, heads=heads, num_landmarks=num_landmarks,
                                              pinv_iterations=pinv_iterations, residual=attn_values_residual,
                                              residual_conv_kernel=attn_values_residual_conv_kernel,
                                              dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x

###################################################################################
#
#    Functions/Classes for capturing region denpendency by
#    1. Transformer: may be suitable for dense relations
#    2. SimTransformer: may be suitable for sparse relations (rubost to noises)
#    3. Conv1D/Conv2D: is used for relations of spatial neighbours
#
###################################################################################


def make_transformer_layer(backbone: str, args):
    """
    [B, N, C] --Transformer--> [B, N, C]

    Transformer/Nystromformer: for long range dependency building.
    Conv1D/Conv2D: for short range dependency building.
    """
    if backbone == 'Transformer':
        patch_encoder_layer = nn.TransformerEncoderLayer(
            args.d_model, args.nhead, dim_feedforward=args.d_model,
            dropout=args.dropout, activation='relu', batch_first=True
        )
        patch_transformer = nn.TransformerEncoder(patch_encoder_layer, num_layers=args.num_layers)
    elif backbone == 'Nystromformer':
        patch_transformer = Nystromformer(
            dim=args.d_model, depth=args.num_layers, heads=args.nhead,
            attn_dropout=args.dropout
        )
    elif backbone == 'Conv1D':
        patch_transformer = Conv1dPatchEmbedding(
            args.d_model, args.d_out, args.ksize, 1, padding=(args.ksize - 1) // 2,
            norm=True, dw_conv=args.dw_conv, activation=True
        )
    elif backbone == 'Conv2D':
        patch_transformer = Conv2dPatchEmbedding(
            args.d_model, args.d_out, args.ksize, 1, padding=(args.ksize - 1) // 2,
            norm=True, dw_conv=args.dw_conv, activation=True
        )
    elif backbone == 'SimTransformer':
        patch_transformer = SimTransformer(
            args.d_model, proj_qk_dim=args.d_out, proj_v_dim=args.d_out,
            epsilon=args.epsilon
        )
    elif backbone == 'Identity':
        patch_transformer = nn.Identity()
    else:
        raise NotImplementedError(f'{backbone} has not implemented.')
    return patch_transformer


class Conv1dPatchEmbedding(nn.Module):
    """Conv1dPatchEmbedding"""

    def __init__(self, in_dim, out_dim, conv1d_ksize, stride=1, padding=0,
                 norm=True, dw_conv=False, activation=False):
        super(Conv1dPatchEmbedding, self).__init__()
        if dw_conv:
            self.conv = nn.Conv1d(in_dim, out_dim, conv1d_ksize, stride, padding, group=out_dim)
        else:
            self.conv = nn.Conv1d(in_dim, out_dim, conv1d_ksize, stride, padding)
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        """x: [B, N, C]"""
        x = x.transpose(2, 1)  # [B, C, N]
        x = self.conv(x)  # [B, C', N]
        x = x.transpose(2, 1)  # [B, N, C']
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Conv2dPatchEmbedding(nn.Module):
    """Conv2dPatchEmbedding
    Conv2dPatchEmbedding: sequences to square and make 2d conv.
    """

    def __init__(self, in_dim, out_dim, conv2d_ksize, stride=1, padding=0,
                 norm=True, dw_conv=True, activation=True):
        super(Conv2dPatchEmbedding, self).__init__()
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, conv2d_ksize, stride, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, conv2d_ksize, stride, padding)
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        """x: [B, N, C]"""
        squ_x = square_seq(x)  # [B, L, L, C]
        squ_x = squ_x.permute(0, 3, 1, 2)  # [B, C, L, L]
        squ_x = self.conv(squ_x)  # [B, C, L, L]
        x = squ_x.flatten(2).transpose(2, 1)  # [B, L*L, C]
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SimTransformer(nn.Module):
    def __init__(self, in_dim, proj_qk_dim=None, proj_v_dim=None, epsilon=None):
        """
        in_dim: the dimension of input.
        proj_qk_dim: the dimension of projected Q, K.
        proj_v_dim: the dimension of projected V.
        topk: number of patches with highest attention values.
        """
        super(SimTransformer, self).__init__()
        self._markoff_value = 0
        self.epsilon = epsilon
        if proj_qk_dim is None:
            proj_qk_dim = in_dim
        if proj_v_dim is None:
            proj_v_dim = in_dim
        self.proj_qk = nn.Linear(in_dim, proj_qk_dim, bias=False)
        nn.init.xavier_uniform_(self.proj_qk.weight)
        self.proj_v = nn.Linear(in_dim, proj_v_dim, bias=False)
        nn.init.xavier_uniform_(self.proj_v.weight)
        self.norm = nn.LayerNorm(proj_v_dim)

    def forward(self, x):
        q, k, v = self.proj_qk(x), self.proj_qk(x), self.proj_v(x)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        attention = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        if self.epsilon is not None:
            mask = (attention > self.epsilon).detach().float()
            attention = attention * mask + self._markoff_value * (1 - mask)
        out = torch.matmul(attention, v)
        out = self.norm(out)
        return out


from collections import OrderedDict
from os.path import join
import math
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


from torch.nn.functional import *


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_raw: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim: (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:

            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw

            # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


import torch
from torch import Tensor
# from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_raw=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask)


# for graph construction
import nmslib
import math


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def pt2graph(coords, features, threshold=5000, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain

    coords, features = np.array(coords.cpu().detach()), np.array(features.cpu().detach())
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
    start_point = edge_spatial[0, :]
    end_point = edge_spatial[1, :]
    start_coord = coords[start_point]
    end_coord = coords[end_point]
    tmp = start_coord - end_coord
    edge_distance = []
    for i in range(tmp.shape[0]):
        distance = math.hypot(tmp[i][0], tmp[i][1])
        edge_distance.append(distance)

    filter_edge_spatial = edge_spatial[:, np.array(edge_distance) <= threshold]

    G = geomData(x=torch.Tensor(features),
                 edge_index=filter_edge_spatial,
                 edge_latent=edge_latent,
                 centroid=torch.Tensor(coords))

    return G


def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    return Ixy


import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        if act == 'gelu':
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K, bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self, x, no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D, bias=bias),
        ]
        if act == 'gelu':
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D, bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x, no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act='relu', gated=False, bias=False, dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim, act, bias, dropout)
        else:
            self.attention = Attention(input_dim, act, bias, dropout)

    def forward(self, x, return_attn=False, no_norm=False, **kwargs):

        x, attn = self.attention(x, no_norm)

        if return_attn:
            return x.squeeze(1), attn.squeeze(1)
        else:
            return x.squeeze(1)


import torch, einops
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_


class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=512):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size + 1, dim, padding_idx=0)
        self.pos_ids = torch.arange(1, size + 1, dtype=torch.long).cuda()

    def forward(self, emb):
        device = emb.device
        b, n, *_ = emb.shape
        pos_ids = self.pos_ids
        if n > self.size:
            zeros = torch.zeros(n - self.size, dtype=torch.long, device=device)
            pos_ids = torch.cat([pos_ids, zeros])
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=b)
        pos_emb = self.pe(pos_ids)  # [b n pe_dim]
        embeddings = torch.cat([emb, pos_emb], dim=-1)
        return embeddings


class PPEG(nn.Module):
    def __init__(self, dim=512, k=7, conv_1d=False, bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                           (k, 1), 1,
                                                                                                           (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))

        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length > 0:
            x = x[:, :-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class PEG(nn.Module):
    def __init__(self, dim=512, k=7, bias=True, conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                           (k, 1), 1,
                                                                                                           (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length > 0:
            x = x[:, :-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class SINCOS(nn.Module):
    def __init__(self, embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        # B, N, C = x.shape
        B, H, W, C = x.shape
        # # padding
        # H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        # add_length = H * W - N
        # x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        # pos_embed = torch.zeros(1, H * W + 1, self.embed_dim)
        # pos_embed = self.get_2d_sincos_pos_embed(pos_embed.shape[-1], int(H), cls_token=True)
        # pos_embed = torch.from_numpy(self.pos_embed).float().unsqueeze(0).to(x.device)

        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)

        # print(pos_embed.size())
        # print(x.size())
        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

        # x = x + pos_embed[:, 1:, :]

        # if add_length >0:
        #     x = x[:,:-add_length]

        return x


class APE(nn.Module):
    def __init__(self, embed_dim=512, num_patches=64):
        super(APE, self).__init__()
        self.absolute_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        B, H, W, C = x.shape
        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)


class RPE(nn.Module):
    def __init__(self, num_heads=8, region_size=(8, 8)):
        super(RPE, self).__init__()
        self.region_size = region_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * region_size[0] - 1) * (2 * region_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the region
        coords_h = torch.arange(region_size[0])
        coords_w = torch.arange(region_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += region_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += region_size[1] - 1
        relative_coords[:, :, 0] *= 2 * region_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        print(relative_position_bias.size())

        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)


import torch
import torch.nn as nn
import numpy as np
import math


# --------------------------------------------------------
# Modified by Swin@Microsoft
# --------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions


def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class InnerAttention(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 epeg=True, epeg_k=15, epeg_2d=False, epeg_bias=True, epeg_type='attn'):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type
        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, epeg_k, padding=padding, groups=num_heads, bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, epeg_k, padding=padding,
                                        groups=head_dim * num_heads, bias=epeg_bias)
            else:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), padding=(padding, 0), groups=num_heads,
                                        bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, (epeg_k, 1), padding=(padding, 0),
                                        groups=head_dim * num_heads, bias=epeg_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
        """
        B_, N, C = x.shape

        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.epeg_type == 'attn':
            pe = self.pe(attn)
            attn = attn + pe

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        if self.pe is not None and self.epeg_type == 'value_af':
            # print(v.size())
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_, self.num_heads * self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., region_attn='native',
                 **kawrgs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # R-MSA
        attn_regions = self.attn(x_regions)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class CrossRegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., crmsa_k=3,
                 crmsa_mlp=False, region_attn='native', **kawrgs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        self.attn = InnerAttention(
            dim, head_dim=head_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kawrgs)

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = [nn.Linear(self.dim, self.dim // 4, bias=False)]
            self.phi += [nn.Tanh()]
            self.phi += [nn.Linear(self.dim // 4, crmsa_k, bias=False)]
            self.phi = nn.Sequential(*self.phi)
        else:
            self.phi = nn.Parameter(
                torch.empty(
                    (self.dim, crmsa_k),
                )
            )
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # CR-MSA
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)  # W*B, sW, region_size*region_size
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1,
                                                                                        2)  # nW*B, sW, region_size*region_size

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (
                    logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0,
                                                                                                                 1)  # sW, nW, C

        if return_attn:
            attn_regions, _attn = self.attn(attn_regions, return_attn)  # sW, nW, C
            attn_regions = attn_regions.transpose(0, 1)  # nW, sW, C
        else:
            attn_regions = self.attn(attn_regions).transpose(0, 1)  # nW, sW, C

        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions,
                                    dispatch_weights_mm)  # nW, sW, region_size*region_size, C
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(
            dim=1)  # nW, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


import importlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error


class TBLogger(object):
    def __init__(self, log_dir=None):
        super(TBLogger, self).__init__()
        self.log_dir = log_dir
        tb_module = importlib.import_module("torch.utils.tensorboard")
        self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)

    def end(self):
        self.tb_logger.flush()
        self.tb_logger.close()

    def run(self, func_name, *args, mode="tb", **kwargs):
        if func_name == "log_scalars":
            return self.tb_log_scalars(*args, **kwargs)
        else:
            tb_log_func = getattr(self.tb_logger, func_name)
            return tb_log_func(*args, **kwargs)
        return None

    def tb_log_scalars(self, metric_dict, step):
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)


class MetricLogger(object):
    def __init__(self):
        super(MetricLogger, self).__init__()
        self.y_pred = []
        self.y_true = []

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.y_pred.append(Y_hat)
        self.y_true.append(Y)

    def get_summary(self):
        acc = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)  # accuracy
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=None)  # f1 score
        weighted_f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average='weighted')  # weighted f1 score
        kappa = cohen_kappa_score(y1=self.y_true, y2=self.y_pred, weights='quadratic')  # cohen's kappa

        print('*** Metrics ***')
        print('* Accuracy: {}'.format(acc))
        for i in range(len(f1)):
            print('* Class {} f1-score: {}'.format(i, f1[i]))
        print('* Weighted f1-score: {}'.format(weighted_f1))
        print('* Kappa score: {}'.format(kappa))

        summary = {'accuracy': acc, 'weighted_f1': weighted_f1, 'kappa': kappa}
        for i in range(len(f1)):
            summary[f'class_{i}_f1'] = f1[i]
        return summary

    def get_confusion_matrix(self):
        cf_matrix = confusion_matrix(np.array(self.y_true), np.array(self.y_pred))  # confusion matrix
        return cf_matrix


"""
Based on the differentiable Top-K operator from:

Cordonnier, J., Mahendran, A., Dosovitskiy, A.: Differentiable patch selection for
image recognition. In: IEEE Conference on Computer Vision and Pattern Recog-
nition (CVPR 2021)

https://openaccess.thecvf.com/content/CVPR2021/supplemental/Cordonnier_Differentiable_Patch_Selection_CVPR_2021_supplemental.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
  def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
    super(PerturbedTopK, self).__init__()
    self.num_samples = num_samples
    self.sigma = sigma
    self.k = k

  def __call__(self, x):
    return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
    # b = batch size
    b, num_patches = x.shape
    # for Gaussian: noise and gradient are the same.
    noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, num_patches)).to(x.device)

    perturbed_x = x[:, None, :] + noise * sigma # [b, num_s, num_p]
    topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # [b, num_s, k]
    indices = torch.sort(indices, dim=-1).values # [b, num_s, k]

    # b, num_s, k, num_p
    perturbed_output = F.one_hot(indices, num_classes=num_patches).float()
    indicators = perturbed_output.mean(dim=1) # [b, k, num_p]

    # constants for backward
    ctx.k = k
    ctx.num_samples = num_samples
    ctx.sigma = sigma

    # tensors for backward
    ctx.perturbed_output = perturbed_output
    ctx.noise = noise

    return indicators

  @staticmethod
  def backward(ctx, grad_output):
    if grad_output is None:
      return tuple([None] * 5)

    noise_gradient = ctx.noise
    expected_gradient = (
        torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
        / ctx.num_samples
        / ctx.sigma
    )
    grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
    return (grad_input,) + tuple([None] * 5)



from math import ceil, floor
from collections import OrderedDict
from typing import List, Optional, Tuple, Callable
from copy import deepcopy

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
#
# from histocartography.preprocessing.feature_extraction import GridDeepFeatureExtractor
# from histocartography.pipeline import PipelineStep
#
#
# class GridPatchExtractor(PipelineStep):
#     def __init__(
#         self,
#         patch_size: int,
#         stride: int = None,
#         fill_value: int = 255,
#         **kwargs,
#     ) -> None:
#         """
#         Create a deep feature extractor.
#         Args:
#             patch_size (int): Desired size of patches.
#             stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
#             fill_value (int): Constant pixel value for image padding. Defaults to 255.
#         """
#         self.patch_size = patch_size
#         if stride is None:
#             self.stride = patch_size
#         else:
#             self.stride = stride
#         super().__init__(**kwargs)
#         self.fill_value = fill_value
#
#     def _process(self, input_image: np.ndarray) -> np.array:
#         return self._extract_patches(input_image)
#
#     def _extract_patches(self, input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Extract patches for a given RGB image.
#         Args:
#             input_image (np.ndarray): RGB input image.
#         Returns:
#             Tuple[np.array, np.array]: patches w/ dim=[n_patches, n_channels, patch_size, patch_size], indices w/ dim=[n_patches,2].
#         """
#         n_channels = input_image.shape[-1]
#         patches = generate_patches(input_image,
#                                    patch_size=self.patch_size,
#                                    stride=self.stride)
#         valid_indices = []
#         valid_patches = []
#         for row in range(patches.shape[0]):
#             for col in range(patches.shape[1]):
#                 valid_indices.append(np.array([row, col]))
#                 valid_patches.append(patches[row, col])
#
#         valid_patches = np.array(valid_patches)
#         valid_indices = np.array(valid_indices)
#         indices = np.array([[row, col] for row in range(patches.shape[0]) for col in range(patches.shape[1])])
#         patches = patches.reshape([-1, n_channels, self.patch_size, self.patch_size])
#         return patches, indices
#
#
# class MaskedGridPatchExtractor(GridPatchExtractor):
#     def __init__(
#         self,
#         tissue_thresh: float = 0.1,
#         **kwargs
#     ) -> None:
#         """
#         Create a patch extractor that can process an image with a corresponding tissue mask.
#         Args:
#             tissue_thresh (float): Minimum fraction of tissue (vs background) for a patch to be considered as valid.
#         """
#         super().__init__(**kwargs)
#         self.tissue_thresh = tissue_thresh
#
#     def _process(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         return self._extract_patches(input_image, mask)
#
#     def _extract_patches(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Generate tissue mask and extract features of patches from a given RGB image.
#         Record which patches are valid and which ones are not.
#         Args:
#             input_image (np.ndarray): RGB input image.
#         Returns:
#             Tuple[np.array, np.array]: patches w/ dim=[n_patches, n_channels, patch_size, patch_size], indices w/ dim=[n_patches,2].
#         """
#         mask = np.expand_dims(mask, axis=2)
#
#         # load all the patches w/ shape = num_x X num_y x 3 x patch_size x patch_size
#         patches, mask_patches = generate_patches(
#             input_image,
#             patch_size=self.patch_size,
#             stride=self.stride,
#             mask=mask
#         )
#
#         valid_indices = []
#         valid_patches = []
#         for row in range(patches.shape[0]):
#             for col in range(patches.shape[1]):
#                 if self._validate_patch(mask_patches[row, col]):
#                     valid_patches.append(patches[row, col])
#                     valid_indices.append(str(row) + '_' + str(col))
#
#         valid_patches = np.array(valid_patches)
#         valid_indices = np.array(valid_indices)
#
#         return valid_patches, valid_indices
#
#     def _validate_patch(self, mask_patch: torch.Tensor) -> Tuple[List[bool], torch.Tensor]:
#         """
#         Record if patch is valid (sufficient area of tissue compared to background).
#         Args:
#             mask_patch (torch.Tensor): a mask patch.
#         Returns:
#             bool: Boolean filter for (in)valid patch
#         """
#         tissue_fraction = (mask_patch == 1).sum() / mask_patch.size
#         if tissue_fraction >= self.tissue_thresh:
#             return True
#         return False

#
# class MaskedGridDeepFeatureExtractor(GridDeepFeatureExtractor):
#     def __init__(
#         self,
#         tissue_thresh: float = 0.1,
#         seed: int = 1,
#         **kwargs
#     ) -> None:
#         """
#         Create a deep feature extractor that can process an image with a corresponding tissue mask.
#
#         Args:
#             tissue_thresh (float): Minimum fraction of tissue (vs background) for a patch to be considered as valid.
#         """
#         super().__init__(**kwargs)
#         np.random.seed(seed)
#         self.tissue_thresh = tissue_thresh
#         self.transforms = None
#         self.avg_pooler = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#     def _collate_patches(self, batch):
#         """Patch collate function"""
#         indices = [item[0] for item in batch]
#         patches = [item[1] for item in batch]
#         mask_patches = [item[2] for item in batch]
#         patches = torch.stack(patches)
#         mask_patches = torch.stack(mask_patches)
#         return indices, patches, mask_patches
#
#     def _process(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         return self._extract_features(input_image, mask)
#
#     def _extract_features(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Generate tissue mask and extract features of patches from a given RGB image.
#         Record which patches are valid and which ones are not.
#
#         Args:
#             input_image (np.ndarray): RGB input image.
#
#         Returns:
#             Tuple[pd.DataFrame, pd.DataFrame]: Boolean index filter, patch features.
#         """
#         if self.downsample_factor != 1:
#             input_image = self._downsample(input_image, self.downsample_factor)
#             mask = self._downsample(mask, self.downsample_factor)
#         mask = np.expand_dims(mask, axis=2)
#
#         # create dataloader for image and corresponding mask patches
#         masked_patch_dataset = MaskedGridPatchDataset(image=input_image,
#                                                       mask=mask,
#                                                       resize_size=self.resize_size,
#                                                       patch_size=self.patch_size,
#                                                       stride=self.stride,
#                                                       mean=self.normalizer_mean,
#                                                       std=self.normalizer_std)
#         del input_image, mask
#         patch_loader = DataLoader(masked_patch_dataset,
#                                   shuffle=False,
#                                   batch_size=self.batch_size,
#                                   num_workers=self.num_workers,
#                                   collate_fn=self._collate_patches)
#
#         # create dictionaries where the keys are the patch indices
#         all_index_filter = OrderedDict(
#             {(h, w): None for h in range(masked_patch_dataset.outshape[0])
#                 for w in range(masked_patch_dataset.outshape[1])}
#         )
#         all_features = deepcopy(all_index_filter)
#
#         # extract features of all patches and record which patches are (in)valid
#         indices = list(all_features.keys())
#         offset = 0
#         for _, img_patches, mask_patches in patch_loader:
#             index_filter, features = self._validate_and_extract_features(img_patches, mask_patches)
#             if len(img_patches) == 1:
#                 features = features.unsqueeze(dim=0)
#             features = features.reshape(features.shape[0], 1024, 16, 16)
#             features = self.avg_pooler(features).squeeze(dim=-1).squeeze(dim=-1).cpu().detach().numpy()
#             for i in range(len(index_filter)):
#                 all_index_filter[indices[offset+i]] = index_filter[i]
#                 all_features[indices[offset+i]] = features[i]
#             offset += len(index_filter)
#
#         # convert to pandas dataframes to enable storing as .h5 files
#         all_index_filter = pd.DataFrame(all_index_filter, index=['is_valid'])
#         all_features = pd.DataFrame(np.transpose(np.stack(list(all_features.values()))),
#                                     columns=list(all_features.keys()))
#
#         return all_index_filter, all_features
#
#     def _validate_and_extract_features(self, img_patches: torch.Tensor, mask_patches: torch.Tensor) -> Tuple[List[bool], torch.Tensor]:
#         """
#         Record which image patches are (in)valid.
#         Extract features from the given image patches.
#
#         Args:
#             img_patches (torch.Tensor): Batch of image patches.
#             mask_patches (torch.Tensor): Batch of mask patches.
#
#         Returns:
#             Tuple[List[bool], torch.Tensor]: Boolean filter for (in)valid patches, extracted patch features.
#         """
#         # record valid and invalid patches (sufficient area of tissue compared to background)
#         index_filter = []
#         for mask_p in mask_patches:
#             tissue_fraction = (mask_p == 1).sum() / torch.numel(mask_p)
#             if tissue_fraction >= self.tissue_thresh:
#                 index_filter.append(True)
#             else:
#                 index_filter.append(False)
#
#         # extract features of all patches unless all are invalid
#         if any(index_filter):
#             features = self.patch_feature_extractor(img_patches)
#         else:
#             features = torch.zeros(len(index_filter), 1024*16*16)
#         return index_filter, features


class GridPatchDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        patch_size: int,
        resize_size: int,
        stride: int,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Create a dataset for a given image and extracted instance maps with desired patches
        of (size, size, 3).

        Args:
            image (np.ndarray): RGB input image.
            patch_size (int): Desired size of patches.
            resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                               patches of size patch_size are provided to the network. Defaults to None.
            stride (int): Desired stride for patch extraction.
            mean (list[float], optional): Channel-wise mean for image normalization.
            std (list[float], optional): Channel-wise std for image normalization.
            transform (list[transforms], optional): List of transformations for input image.
        """
        super().__init__()
        basic_transforms = [transforms.ToPILImage()]
        self.resize_size = resize_size
        if self.resize_size is not None:
            basic_transforms.append(transforms.Resize(self.resize_size))
        if transform is not None:
            basic_transforms.append(transform)
        basic_transforms.append(transforms.ToTensor())
        if mean is not None and std is not None:
            basic_transforms.append(transforms.Normalize(mean, std))
        self.dataset_transform = transforms.Compose(basic_transforms)

        self.x_top_pad, self.x_bottom_pad = get_pad_size(image.shape[1], patch_size, stride)
        self.y_top_pad, self.y_bottom_pad = get_pad_size(image.shape[0], patch_size, stride)
        self.pad = torch.nn.ConstantPad2d((self.x_bottom_pad, self.x_top_pad, self.y_bottom_pad, self.y_top_pad), 255)
        self.image = torch.as_tensor(image)
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self._generate_patches(self.image)

    def _generate_patches(self, image):
        """Extract patches"""
        n_channels = image.shape[-1]
        patches = image.unfold(0, self.patch_size, self.stride).unfold(1, self.patch_size, self.stride)
        self.outshape = (patches.shape[0], patches.shape[1])
        patches = patches.reshape([-1, n_channels, self.patch_size, self.patch_size])
        return patches

    def __getitem__(self, index: int):
        """
        Loads an image for a given patch index.

        Args:
            index (int): Patch index.

        Returns:
            Tuple[int, torch.Tensor]: Patch index, image as tensor.
        """
        patch = self.dataset_transform(self.patches[index].numpy().transpose([1, 2, 0]))
        return index, patch

    def __len__(self) -> int:
        return len(self.patches)


class MaskedGridPatchDataset(GridPatchDataset):
    def __init__(
        self,
        mask: np.ndarray,
        **kwargs
    ) -> None:
        """
        Create a dataset for a given image and mask, with extracted patches of (size, size, 3).

        Args:
            mask (np.ndarray): Binary mask.
        """
        super().__init__(**kwargs)

        self.mask_transform = None
        if self.resize_size is not None:
            basic_transforms = [transforms.ToPILImage(),
                                transforms.Resize(self.resize_size),
                                transforms.ToTensor()]
            self.mask_transform = transforms.Compose(basic_transforms)

        self.pad = torch.nn.ConstantPad2d((self.x_bottom_pad, self.x_top_pad, self.y_bottom_pad, self.y_top_pad), 0)
        self.mask = torch.as_tensor(mask)
        self.mask_patches = self._generate_patches(self.mask)

    def __getitem__(self, index: int):
        """
        Loads an image and corresponding mask patch for a given index.

        Args:
            index (int): Patch index.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor]: Patch index, image as tensor, mask as tensor.
        """
        image_patch = self.dataset_transform(self.patches[index].numpy().transpose([1, 2, 0]))
        if self.mask_transform is not None:
            # after resizing, the mask should still be binary and of type uint8
            mask_patch = self.mask_transform(255*self.mask_patches[index].numpy().transpose([1, 2, 0]))
            mask_patch = torch.round(mask_patch).type(torch.uint8)
        else:
            mask_patch = self.mask_patches[index]
        return index, image_patch, mask_patch


def get_pad_size(size: int, patch_size: int, stride: int) -> Tuple[int, int]:
    """Computes the necessary top and bottom padding size to evenly devide an input size into patches with a given stride
    Args:
        size (int): Size of input
        patch_size (int): Patch size
        stride (int): Stride
    Returns:
        Tuple[int, int]: Amount of top and bottom-pad
    """
    target = ceil((size - patch_size) / stride + 1)
    pad_size = ((target - 1) * stride + patch_size) - size
    top_pad = pad_size // 2
    bottom_pad = pad_size - top_pad
    return top_pad, bottom_pad


def pad_image_with_factor(input_image: np.ndarray, patch_size: int, factor: int = 1) -> np.ndarray:
    """
    Pad the input image such that the height and width is the multiple of factor * patch_size.
    Args:
        input_image (np.ndarray):   RGB input image.
        patch_size (int):           Patch size.
        factor (int):               Factor multiplied with the patch size.
    Returns:
        padded_image (np.ndarray): RGB padded image.
    """
    height, width = input_image.shape[0], input_image.shape[1]
    height_new, width_new = patch_size * factor * ceil(height/patch_size/factor), patch_size * factor * ceil(width/patch_size/factor)
    padding_top = floor((height_new - height)/2)
    padding_bottom = ceil((height_new - height)/2)
    padding_left = floor((width_new - width)/2)
    padding_right = ceil((width_new - width)/2)
    padded_image = np.copy(np.pad(input_image, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='constant', constant_values=255))

    return padded_image


def generate_patches(image, patch_size, stride, mask=None):
    """
    Extract patches on an image
    Args:
        image ([np.ndarray]): image to extract patches on
        patch_size ([int]): extract patches of size patch_size x patch_size
        stride ([int]): patch stride
        mask ([np.ndarray], optional): extract same patches on associated mask. Defaults to None.
    Returns:
        [np.ndarray]: Extracted patches
    """
    x_top_pad, x_bottom_pad = get_pad_size(image.shape[1], patch_size, stride)
    y_top_pad, y_bottom_pad = get_pad_size(image.shape[0], patch_size, stride)
    pad = torch.nn.ConstantPad2d((x_bottom_pad, x_top_pad, y_bottom_pad, y_top_pad), 255)
    image = pad(torch.as_tensor(np.array(image)).permute([2, 0, 1])).permute([1, 2, 0])

    patches = image.unfold(0, patch_size, stride).unfold(1, patch_size, stride).detach().numpy()

    if mask is not None:
        pad = torch.nn.ConstantPad2d((x_bottom_pad, x_top_pad, y_bottom_pad, y_top_pad), 0)
        mask = pad(torch.as_tensor(np.array(mask)).permute([2, 0, 1])).permute([1, 2, 0])
        mask_patches = mask.unfold(0, patch_size, stride).unfold(1, patch_size, stride).detach().numpy()
        return patches, mask_patches

    return patches


def get_image_at(image, magnification):
    """
    Get image at a specified magnification.
    Args:
        image (openslide.OpenSlide):    Whole-slide image opened with openslide.
        magnification (float):          Desired magnification.
    Returns:
        image_at (np.ndarray):          Image at given magnification.
    """

    # get image info
    down_factors = image.level_downsamples
    level_dims = image.level_dimensions
    if image.properties.get('aperio.AppMag') is not None:
        max_mag = int(image.properties['aperio.AppMag'])
        assert max_mag in [20, 40]
    else:
        print('WARNING: Assuming max. magnification is 40x!')
        max_mag = 40

    # get native magnifications
    native_mags = [max_mag / int(round(df)) for df in down_factors]

    # get image at the native magnification closest to the requested magnification
    if magnification in native_mags:
        down_level = native_mags.index(magnification)
    else:
        down_level = image.get_best_level_for_downsample(max_mag / magnification)
    print(f'Given magnification {magnification}, best level={down_level}, best mag={native_mags[down_level]}')
    image_at = image.read_region(location=(0, 0),
                                 level=down_level,
                                 size=level_dims[down_level]).convert('RGB')

    # downsample if necessary
    if native_mags[down_level] > magnification:
        w, h = image_at.size
        down_factor = int(native_mags[down_level] // magnification)
        print(f'Downsampling with factor {down_factor}')
        image_at = image_at.resize((w // down_factor, h // down_factor), Image.BILINEAR)

    # convert to np array -> h x w x c
    image_at = np.array(image_at)
    return image_at


def get_tissue_mask(image, mask_generator):
    # Color conversion
    img = image.copy()
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    image_s = image_hsv[:, :, 1]

    # Otsu's thresholding
    _, raw_mask = cv2.threshold(image_s, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img[raw_mask == 0] = 255

    # Extract mask
    refined_mask = mask_generator.process(image=img)
    return refined_mask

import os
import torch
import numpy as np

class ModelSaver:
    def __init__(self, save_path, save_metric='loss'):
        """
        Args:
            save_path (str): Path to save the model
            save_metric (str, optional): Save metric. Defaults to 'loss'.
        """
        self.save_metric = save_metric
        self.save_path = save_path
        self.best_loss = np.inf
        self.best_f1 = 0.

    def __call__(self, model, summary):
        if self.save_metric == 'loss':
            if summary["val_loss"] < self.best_loss:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {summary["val_loss"]:.6f}).  Saving model ...')
                torch.save(model.state_dict(), os.path.join(self.save_path, "model_best_loss.pt"))
                self.best_loss = summary["val_loss"]
        elif self.save_metric == 'f1':
            if summary["val_weighted_f1"] > self.best_f1:
                print(f'Validation weighted f1 increased ({self.best_f1:.6f} --> {summary["val_weighted_f1"]:.6f}).  Saving model ...')
                torch.save(model.state_dict(), os.path.join(self.save_path, "model_best_f1.pt"))
                self.best_f1 = summary["val_weighted_f1"]
        else:
            raise NotImplementedError
