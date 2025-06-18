# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------


"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from score_sde.op import upfirdn3d

# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)


class Conv3d(nn.Module):
  """Conv3d layer with optimal upsampling and downsampling (StyleGAN2)."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
               resample_kernel=(1, 3, 3, 3, 1),
               use_bias=True,
               kernel_init=None):
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel, kernel))
    if kernel_init is not None:
      self.weight.data = kernel_init(self.weight.data.shape)
    if use_bias:
      self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel
    self.kernel = kernel
    self.use_bias = use_bias

  def forward(self, x):
    if self.up:
      x = upsample_conv_3d(x, self.weight, k=self.resample_kernel)
    elif self.down:
      x = conv_downsample_3d(x, self.weight, k=self.resample_kernel)
    else:
      x = F.conv3d(x, self.weight, stride=1, padding=self.kernel // 2)

    if self.use_bias:
      x = x + self.bias.reshape(1, -1, 1, 1, 1)

    return x


def naive_upsample_3d(x, factor=2):
    _N, C, D, H, W = x.shape  # D is the depth dimension
    x = torch.reshape(x, (-1, C, D, 1, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, D * factor, H * factor, W * factor))


def naive_downsample_3d(x, factor=2):
    _N, C, D, H, W = x.shape
    x = torch.reshape(x, (-1, C, D // factor, factor, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5, 7))


def upsample_conv_3d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_3d()` followed by `tf.nn.conv3d()`.

    Padding is performed only once at the beginning, not between the
    operations. The fused op is considerably more efficient than performing the same
    calculation using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x: Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
        w: Weight tensor of the shape `[filterD, filterH, filterW, inChannels, outChannels]`.
           Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k: FIR filter of the shape `[firD, firH, firW]` or `[firN]` (separable).
           The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, D * factor, H * factor, W * factor]` or
        `[N, D * factor, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 5
    convD, convH, convW = w.shape[0], w.shape[1], w.shape[2]
    inC, outC = w.shape[3], w.shape[4]

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel_3d(k) * (gain * (factor ** 3))  # Adjust gain for 3D
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor, factor)

    # Determine data dimensions.
    output_shape = (
        (_shape(x, 2) - 1) * factor + convD,
        (_shape(x, 3) - 1) * factor + convH,
        (_shape(x, 4) - 1) * factor + convW
    )
    output_padding = (
        output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convD,
        output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convH,
        output_shape[2] - (_shape(x, 4) - 1) * stride[2] - convW
    )
    assert all(pad >= 0 for pad in output_padding)
    num_groups = _shape(x, 1) // inC

    # Transpose weights.
    w = torch.reshape(w, (num_groups, -1, inC, convD, convH, convW))
    w = w[..., ::-1, ::-1, ::-1].permute(0, 2, 1, 3, 4, 5)
    w = torch.reshape(w, (num_groups * inC, -1, convD, convH, convW))

    x = F.conv_transpose3d(x, w, stride=stride, output_padding=output_padding, padding=0)

    # Calculate padding for each dimension
    p_d = k.shape[0] - factor
    p_h = k.shape[1] - factor
    p_w = k.shape[2] - factor

    pad = (
        (p_d + 1) // 2 + factor - 1, p_d // 2,
        (p_h + 1) // 2 + factor - 1, p_h // 2,
        (p_w + 1) // 2 + factor - 1, p_w // 2
    )

    return upfirdn3d(x, torch.tensor(k, device=x.device),
                     pad=pad)


def conv_downsample_3d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv3d()` followed by `downsample_3d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x: Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
        w: Weight tensor of the shape `[filterD, filterH, filterW, inChannels, outChannels]`.
           Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k: FIR filter of the shape `[firD, firH, firW]` or `[firN]` (separable).
           The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, D // factor, H // factor, W // factor]` or
        `[N, D // factor, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convD, convH, convW = w.shape
    assert convW == convH == convD  # Ensure the convolution kernel is cubic

    if k is None:
        k = [1] * factor
    k = _setup_kernel_3d(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor, factor]

    # Apply the FIR filter and downsample
    x = upfirdn3d(x, torch.tensor(k, device=x.device),
                  pad=((p + 1) // 2, p // 2))


    # Calculate padding for each dimension
    p_d = k.shape[0] - factor  # Padding for depth
    p_h = k.shape[1] - factor  # Padding for height
    p_w = k.shape[2] - factor  # Padding for width

    # Apply padding symmetrically
    pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = (p_d) // 2, p_d // 2, (p_h) // 2, p_h // 2, (p_w) // 2, p_w // 2

    #print('x before pad', x.shape)
    #x = F.pad(x, (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1), mode='constant', value=0)
    #print('x after pad', x.shape)

    out = F.conv3d(x, w, stride=factor)
    return out


def _setup_kernel_3d(k):
    k = np.asarray(k, dtype=np.float32)

    if k.ndim == 1:
        # Create a 3D cubic kernel from a 1D array
        k = k[:, np.newaxis, np.newaxis] * k[np.newaxis, :, np.newaxis] * k[np.newaxis, np.newaxis, :]  # Broadcasting to create a cubic kernel
    elif k.ndim == 2:
        # Create a 3D cubic kernel from a 2D array
        k = k[:, :, np.newaxis] * k[np.newaxis, :, :] * k[np.newaxis, np.newaxis, :]  # Broadcasting to create a cubic kernel
    elif k.ndim == 3:
        # If k is already 3D, we can proceed to normalize it
        if k.shape[0] != k.shape[1] or k.shape[0] != k.shape[2]:
            raise ValueError("Input 3D array must be cubic (same size in all dimensions).")
    else:
        raise ValueError("Input array must be 1D, 2D, or 3D.")

    # Normalize the kernel
    k /= np.sum(k)

    assert k.ndim == 3
    assert k.shape[0] == k.shape[1] == k.shape[2]  # Ensure it's a cubic kernel
    return k


def _shape(x, dim):
    return x.shape[dim]


def upsample_3d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 3D images with the given filter.

    Accepts a batch of 3D images of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the upsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
        k:            FIR filter of the shape `[firD, firH, firW]` or `[firN]`
                      (separable). The default is `[1] * factor`, which corresponds to
                      nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, D * factor, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel_3d(k) * (gain * (factor ** 3))  # Adjust for 3D scaling
    # Calculate padding for each dimension
    p = k.shape[0] - factor
    return upfirdn3d(x, torch.tensor(k, device=x.device),
                     up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_3d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 3D images with the given filter.

    Accepts a batch of 3D images of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the downsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
        k:            FIR filter of the shape `[firD, firH, firW]` or `[firN]`
                      (separable). The default is `[1] * factor`, which corresponds to
                      average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, D // factor, H // factor, W // factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel_3d(k)
    k = k * gain  # Adjust for 3D kernel setup
    #p = k.shape[0] - factor

    # Calculate padding for each dimension
    p_d = k.shape[0] - factor  # Padding for depth
    p_h = k.shape[1] - factor  # Padding for height
    p_w = k.shape[2] - factor  # Padding for width

    # Apply padding symmetrically
    pad = ((p_d + 1) // 2, p_d // 2, (p_h + 1) // 2, p_h // 2, (p_w + 1) // 2, p_w // 2)

    return upfirdn3d(x, torch.tensor(k, device=x.device),
                     down=factor,
                     pad=pad)  # Adjust padding for 3D


