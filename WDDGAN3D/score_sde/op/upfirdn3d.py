import torch
from torch import nn
import torch.nn.functional as F


class FusedLeakyReLU3D(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu3d(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu3d(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * F.leaky_relu(input + bias.view((1, -1) + (1,) * (len(input.shape) - 2)),
                                negative_slope=negative_slope)


def upfirdn3d_native(input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0,
                     pad_z1):
    input = input.permute(0, 2, 3, 4, 1)  # Change to (N, D, H, W, C)
    _, in_d, in_h, in_w, minor = input.shape
    kernel_d, kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_d, 1, in_h, 1, in_w, 1, minor)

    # Upsample
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0, 0, up_z - 1])
    out = out.view(-1, in_d * up_z, in_h * up_y, in_w * up_x, minor)

    # Padding
    out = F.pad(out,
                [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)])
    out = out[:, max(-pad_z0, 0): out.shape[1] - max(-pad_z1, 0), max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), :]

    out = out.permute(0, 4, 1, 2, 3)  # Change back to (N, C, D, H, W)
    out = out.reshape(
        [-1, 1, in_d * up_z + pad_z0 + pad_z1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])

    # Convolution
    w = torch.flip(kernel, [0, 1, 2]).view(1, 1, kernel_d, kernel_h, kernel_w)
    out = F.conv3d(out, w)

    out = out.reshape(-1, minor, in_d * up_z + pad_z0 + pad_z1 - kernel_d + 1,
                      in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    return out[:, :, ::down_z, ::down_y, ::down_x]


def upfirdn3d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn3d_native(input, kernel, up, up, up, down, down, down, pad[0], pad[1], pad[0],
                           pad[1], pad[0], pad[1])
    return out
