import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

"""
Densenet implementation, in 1d for signal processing
Guidance used from: https://amaarora.github.io/2020/08/02/densenets.html
Densenet arxiv: https://arxiv.org/abs/1608.06993
"""


class TransitionBlock(nn.Module):
    # Transition block reduces feature map dimension between dense blocks
    # Batch norm - RELU - 1d convolution - avg pooling
    # Use 1d convolution to reduce number of channels
    # Use average pooling to reduce size of window
    def __init__(self, input_features, output_features):
        super().__init__()
        self.BN = nn.BatchNorm1d(input_features)
        self.conv = nn.Conv1d(input_features, output_features, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.BN(inputs)
        x = F.relu(self.conv(x))
        x = self.pool(x)

        return x


class DenseLayer(nn.Module):
    # Dense Layer
    #
    def __init__(self, input_features, bottleneck_factor, growth, dropout_rate=0):
        # bottleneck_factor is used to REDUCE the number of features, with a 1x1 convolution
        # growth is the number of new features added per layer
        super().__init__()
        self.BN1 = nn.BatchNorm1d(input_features)
        self.conv1 = nn.Conv1d(input_features, bottleneck_factor * growth, kernel_size=1)
        self.BN2 = nn.BatchNorm1d(bottleneck_factor * growth)
        self.conv2 = nn.Conv1d(bottleneck_factor * growth, growth, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def concat_features(self, inputs):
        # Combines the input list - features from all previous layers
        concatenated_feature_maps = torch.cat(inputs, dim=1)
        return (concatenated_feature_maps)

    def bottleneck(self, inputs):
        # Use 1-d convolutions as bottleneck
        x = self.BN1(inputs)
        x = F.relu(self.conv1(x))
        return x

    def forward(self, inputs):
        # Turn inputs into list, to allow for concatenating
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        else:
            pass

        # First, concatenate all the features (input is a list of all previous layer outputs)

        x = self.concat_features(inputs)
        # Usebottleneck 1-d convolution - each 1x1 convolution produces bn_size*growth feature maps
        # print("Dense Layer input shape (after concat):", x.shape)
        x = self.bottleneck(x)
        x = self.dropout(x)
        x = self.BN2(x)
        # Second convolution is kernel-3, reduces output to just growth (K) feature maps
        new_features = F.relu(self.conv2(x))
        # Dropout, if desired
        new_features = self.dropout(new_features)

        # print("Dense Output shape:", new_features.shape)

        return new_features


class DenseBlock(nn.ModuleDict):
    # DenseBlock is a ModuleDICT, not a Module
    # Series of DenseLayers, where each layer receives as an input all of the feature maps from previous layers
    # As a result, the input size of each dense layer increases by growth_rate with each iteration

    def __init__(self, num_layers, num_input_features, bottleneck_factor, growth_rate, dropout=0):
        super().__init__()
        for i in range(num_layers):
            # Each layer has an input_features size that increases by growth_rate (K) each layer
            layer = DenseLayer(
                input_features=num_input_features + (i * growth_rate),
                bottleneck_factor=bottleneck_factor,
                growth=growth_rate,
                dropout_rate=dropout,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, input_features):
        # First, turn input tensor into a list (will be a list of tensors)
        features = [input_features]

        # Cycle through each layer in the ModuleDict using self.items
        for name, layer in self.items():
            new_features = layer(features)
            # New features from each layer are appended to the list, to concatenate for
            # use in later layers
            features.append(new_features)

        # concatenate all features together prior to completing the DenseBlock
        concatenated_features = torch.cat(features, 1)
        # print("DenseBlock OUTPUT: ", concatenated_features.shape)
        return concatenated_features


class DenseNet(nn.Module):
    def __init__(self, *, growth_rate=32, block_config=(3, 6, 12, 8), num_init_features=32, bottleneck_factor=4, dropout=0.2,
                 num_outputs=12):
        '''

        :param growth_rate:
        :param block_config:
        :param num_init_features: initial number of feature maps to create from the first convolution
        :param bottleneck_factor:
        :param dropout:
        :param num_outputs:
        '''
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv1d(in_channels=1, out_channels=num_init_features, kernel_size=7, stride=2, padding=1)),
                ("bn0", nn.BatchNorm1d(num_init_features)),
                ("relu0", nn.ReLU()),
                ("avgpool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            ])
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bottleneck_factor=bottleneck_factor,
                growth_rate=growth_rate,
                dropout=dropout,
            )

            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + (num_layers * growth_rate)
            if i != len(block_config) - 1:
                transition = TransitionBlock(
                    input_features=num_features,
                    output_features=num_features // 2,
                )
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = num_features // 2

        self.final_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear_final = nn.Linear(num_features, num_outputs)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.num_params = self.__get_num_params__()
        print(f"Number of parameters: {self.num_params}")

    def forward(self, x):
        # Unsqueeze to put data in (Batch size, 1 channel, freq dim) shape
        x = torch.unsqueeze(x, 1)
        # self.features puts through Densenet
        features = self.features(x)
        # Use adaptive average pool to go from (B, C, L) shape to (B, C, 1) shape
        # print("Before final pool: ", features.shape)
        features = self.final_pool(features)
        # print("After final pool: ", features.shape)
        # Use torch flatten to set up matrix for final fully-connected layer
        features = torch.flatten(features, start_dim=1)
        # print("After flatten: ", features.shape)
        # Fully connected layer reduces to 12 features
        features = self.linear_final(features)

        return features

    def __get_num_params__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params


class DenseNetMultiHead(nn.Module):
    def __init__(self, *, growth_rate=32, block_config=(3, 6, 12, 8), num_init_features=32, bottleneck_factor=4, dropout=0.2,
                 num_outputs=12):
        '''

        :param growth_rate:
        :param block_config:
        :param num_init_features: initial number of feature maps to create from the first convolution
        :param bottleneck_factor:
        :param dropout:
        :param num_outputs:
        '''
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv1d(in_channels=1, out_channels=num_init_features, kernel_size=7, stride=2, padding=1)),
                ("bn0", nn.BatchNorm1d(num_init_features)),
                ("relu0", nn.ReLU()),
                ("avgpool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            ])
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bottleneck_factor=bottleneck_factor,
                growth_rate=growth_rate,
                dropout=dropout,
            )

            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + (num_layers * growth_rate)
            if i != len(block_config) - 1:
                transition = TransitionBlock(
                    input_features=num_features,
                    output_features=num_features // 2,
                )
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = num_features // 2

        self.final_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear_final = nn.Linear(num_features, 400)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.num_params = self.__get_num_params__()
        print(f"Number of parameters: {self.num_params}")

        #Multihead
        self.head1 = HeadBlock(400, 2)
        self.head2 = HeadBlock(400, 2)
        self.head3 = HeadBlock(400, 2)
        self.head4 = HeadBlock(400, 2)
        self.head5 = HeadBlock(400, 2)
        self.head6 = HeadBlock(400, 2)

    def forward(self, x):
        # Unsqueeze to put data in (Batch size, 1 channel, freq dim) shape
        x = torch.unsqueeze(x, 1)
        # self.features puts through Densenet
        features = self.features(x)
        # Use adaptive average pool to go from (B, C, L) shape to (B, C, 1) shape
        features = self.final_pool(features)
        # Use torch flatten to set up matrix for final fully-connected layer
        features = torch.flatten(features, start_dim=1)
        # Fully connected layer reduces to 12 features
        X = self.linear_final(features)

        x1 = self.head1(X)
        x2 = self.head2(X)
        x3 = self.head3(X)
        x4 = self.head4(X)
        x5 = self.head5(X)
        x6 = self.head6(X)

        features = torch.cat((x1, x2, x3, x4, x5, x6), dim=-1)

        return features

    def __get_num_params__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params


class HeadBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(HeadBlock, self).__init__()

        self.linear1 = nn.Linear(in_channels, 200)
        self.linear2 = nn.Linear(200, 50)
        self.linear3 = nn.Linear(50, out_channels)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        # print('1d shape', one_d.shape)

        return X