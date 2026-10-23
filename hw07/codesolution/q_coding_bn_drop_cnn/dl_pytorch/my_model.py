import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################
# TODO: Design your own neural network
# You can define utility functions/classes here
#######################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_a = nn.BatchNorm2d(out_channels)

        self.conv_b1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_b1 = nn.BatchNorm2d(out_channels)
        self.conv_b2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn_b2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x_a = self.bn_a(self.conv_a(x))
        x_b = F.relu(self.bn_b1(self.conv_b1(x)))
        x_b = F.relu(self.bn_b2(self.conv_b2(x_b)))
        return self.pool(x_a + x_b)
#######################################################################
# End of your code
#######################################################################


class MyNeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1)
        self.block1 = ConvBlock(16, 32)
        self.block2 = ConvBlock(32, 128)
        self.block3 = ConvBlock(128, 256)
        self.drop = nn.Dropout(p_dropout)
        self.fc = nn.Linear(256, 100)
        #######################################################################
        # End of your code
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        x = self.conv0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=-1).mean(dim=-1)
        x = self.fc(self.drop(x))
        return x
        #######################################################################
        # End of your code
        #######################################################################
