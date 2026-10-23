import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        #######################################################################
        # TODO: Complete the implementation of the first convolutional block
        #
        # Feel free to check the hints of tensor shapes in the `forward` method
        #
        # Refer to pytorch.org for API documentations.
        #
        # The first conv block consists of a conv layer, an optional spatial
        #  batch normalization layer, a ReLU activation layer, and a maxpooling
        #  layer:
        # [conv1] -> ([bn1]) -> [relu1] -> [pool1]
        #
        # All conv layers in this neural network uses a kernel size of 3 and a
        #  padding of 1.
        #
        # Batch norm is enabled if and only if `do_batchnorm` is true
        #
        # All max-pooling layers in this neural network pools each non-
        #  overlapping 2x2 patch to a single pixel, shrinking the height/width
        #  of the feature map by 1/2.
        #
        # The first conv block has 16 filters
        #######################################################################
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        if do_batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        #######################################################################
        # End of your code
        #######################################################################

        #######################################################################
        # TODO: Implement the second convolutional block
        #
        # The second convolutional block has the same structure as the first,
        #  except that the conv layer has 32 filters
        #######################################################################
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        if do_batchnorm:
            self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        #######################################################################
        # End of your code
        #######################################################################

        #######################################################################
        # TODO: Implement the third convolutional block
        #
        # The third convolutional block uses a strided conv layer with a stride
        #  of 2. It has 64 filters
        #
        # The conv layer is followed by an optional spatial batch norm layer,
        #  and a ReLU activation layer. No pooling in this block.
        #######################################################################
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        if do_batchnorm:
            self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        #######################################################################
        # End of your code
        #######################################################################

        #######################################################################
        # TODO: Complete the 2-layer fully-connected classifier
        #
        # The input to this classifier is a flattened 1024-d vector for each
        #  input image.
        #
        # The classifier consists of two fully-connected layers, with a hidden
        #  dimension of 256 and an output dimension of 100. Dropout after the
        #  activation layer is enabled if and only if `p_dropout > 0.0`
        # [fc1] -> [relu4] -> ([drop]) -> [fc2]
        #
        # Feel free to check the hints of tensor shapes in the `forward` method
        #
        #######################################################################
        self.fc1 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        if p_dropout > 0.0:
            self.drop = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(256, 100)
        #######################################################################
        # End of your code
        #######################################################################

    def forward(self, x):
        # The shape of `x` is [bsz, 3, 32, 32]

        x = self.conv1(x)  # [bsz, 16, 32, 32]
        if self.do_batchnorm:
            x = self.bn1(x)
        x = self.pool1(self.relu1(x))  # [bsz, 16, 16, 16]

        x = self.conv2(x)  # [bsz, 32, 16, 16]
        if self.do_batchnorm:
            x = self.bn2(x)
        x = self.pool2(self.relu2(x))  # [bsz, 32, 8, 8]

        x = self.conv3(x)  # [bsz, 64, 4, 4]
        if self.do_batchnorm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = torch.flatten(x, 1)  # [bsz, 1024]
        x = self.relu4(self.fc1(x))  # [bsz, 256]
        if self.p_dropout > 0.0:
            x = self.drop(x)
        x = self.fc2(x)  # [bsz, 100]
        return x
