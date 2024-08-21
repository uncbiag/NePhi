import torch.nn as nn


class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        batchnorm=False,
        residual=False,
        nonlinear=nn.LeakyReLU(0.2),
        dim=3,
    ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convBlock, self).__init__()
        conv = getattr(nn, f"Conv{dim}d")
        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = getattr(nn, f"BatchNorm{dim}d")(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        if residual:
            self.residual = conv(in_channels, out_channels, 1, stride=stride, bias=bias)
        else:
            self.residual = None

    def forward(self, x):
        x_1 = self.conv(x)
        if self.bn:
            x_1 = self.bn(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        if self.residual:
            x_1 = self.residual(x) + x_1
        return x_1

class FullyConnectBlock(nn.Module):
    """
    A fully connect block including fully connect layer, nonliear activiation
    """

    def __init__(
        self, in_channels, out_channels, bias=True, nonlinear=nn.LeakyReLU(0.2)
    ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(FullyConnectBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = nonlinear

    def forward(self, x):
        x_1 = self.fc(x)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        return x_1