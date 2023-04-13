import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
1-d implementation of InceptionNew based Architecture
"""


class InceptionBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels_1,
                 out_channels_3,
                 out_channels_5,
                 out_channels_max,
                 three_dim_red,
                 five_dim_red):
        super().__init__()
        # 1-k convs
        self.conv1 = nn.Conv1d(in_channels, out_channels_1, kernel_size=1, stride=1, padding=0)

        # 3-k convs
        self.dimred3 = nn.Conv1d(in_channels, three_dim_red, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(three_dim_red, out_channels_3, kernel_size=3, stride=1, padding=1)

        # 5-k convs
        self.dimred5 = nn.Conv1d(in_channels, five_dim_red, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(five_dim_red, out_channels_5, kernel_size=5, stride=1, padding=2)

        # max pool
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.dimredmax = nn.Conv1d(in_channels, out_channels_max, 1)

    def forward(self, X):
        one_d = F.relu(self.conv1(X))

        three_d = F.relu(self.dimred3(X))
        three_d = F.relu(self.conv3(three_d))

        five_d = F.relu(self.dimred5(X))
        five_d = F.relu(self.conv5(five_d))

        max_d = F.relu(self.maxpool(X))
        max_d = F.relu(self.dimredmax(max_d))

        # depthwise concatenation
        final = torch.cat((one_d, three_d, five_d, max_d), dim=1)

        return final


class InceptionNet(nn.Module):
    def __init__(self, num_outputs=12):
        super().__init__()
        self.name = "inception"
        self.conv1 = nn.Conv1d(1, 32, 3, stride=2, padding=3)
        # self.max1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, stride=1, padding=1)
        self.max = nn.MaxPool1d(3, stride=2, padding=1)

        self.incept1 = InceptionBlock(in_channels=64,
                                      out_channels_1=64,
                                      out_channels_3=128,
                                      out_channels_5=32,
                                      out_channels_max=32,
                                      three_dim_red=96,
                                      five_dim_red=16)

        self.max1 = nn.MaxPool1d(3, stride=2, padding=1)

        self.incept2 = InceptionBlock(in_channels=256,
                                      out_channels_1=64,
                                      out_channels_3=128,
                                      out_channels_5=32,
                                      out_channels_max=32,
                                      three_dim_red=96,
                                      five_dim_red=16)

        self.max2 = nn.MaxPool1d(3, stride=2, padding=1)

        self.incept3 = InceptionBlock(in_channels=256,
                                      out_channels_1=64,
                                      out_channels_3=128,
                                      out_channels_5=32,
                                      out_channels_max=32,
                                      three_dim_red=96,
                                      five_dim_red=16)

        self.max3 = nn.MaxPool1d(3, stride=2, padding=1)

        self.incept4 = InceptionBlock(in_channels=256,
                                      out_channels_1=64,
                                      out_channels_3=128,
                                      out_channels_5=32,
                                      out_channels_max=32,
                                      three_dim_red=96,
                                      five_dim_red=16)


        self.avg_pool = nn.AvgPool1d(5, stride=2)

        # self.linear1 = nn.Linear(40448,400)
        self.linear1 = nn.Linear(3328, 400)
        self.linear2 = nn.Linear(400, num_outputs)

        self.num_params = self.__get_num_params__()
        print(f"Number of parameters: {self.num_params}")

    def forward(self, X):
        X = torch.unsqueeze(X, 1)
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X)) # Bx32x454
        X = F.relu(self.conv3(X)) # Bx32x454
        X = self.max(X) # Bx64x454
        # print('e', X.shape)

        X = self.incept1(X) # Bx64x227
        X = self.max1(X) # Bx257x227
        # print('g', X.shape)

        X = self.incept2(X) # Bx256x114
        X = self.max2(X)

        X = self.incept3(X) #Bx256x57
        X = self.max3(X)

        X = self.incept4(X) #Bx256x29
        X = self.avg_pool(X)
        # print('i', X.shape)

        X = torch.flatten(X, start_dim=1) #Bx256x13
        # print('j', X.shape)

        X = F.relu(self.linear1(X)) #Bx3328
        # print('k', X.shape)
        X = self.linear2(X)
        # print('l', X.shape)

        return X

    def __get_num_params__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params


class ConvolutionalNet(nn.Module):

    # define model elements
    def __init__(self, n_params=12):
        super().__init__()

        self.name = "std_convolutional"

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, stride=5, padding=2)
        self.conv4 = nn.Conv1d(64, 64, 5, stride=5, padding=2)
        self.conv_final = nn.Conv1d(64, 5, 1, stride=1, padding=0)

        # Linear layers
        self.linear1 = nn.Linear(410, 100)
        self.linear2 = nn.Linear(100, n_params)

    # forward propagate input
    def forward(self, X):
        # unsqueeze to fit dimension for nn.Conv1d
        X = torch.unsqueeze(X, 1)

        # Four convolutional layers
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))

        # 1D convolution, then flatten for FC layers
        X = F.relu(self.conv_final(X))
        X = torch.flatten(X, start_dim=1)

        # Two FC layers to give output
        X = F.relu(self.linear1(X))
        out = self.linear2(X)

        return out


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def summary(self):
        self.logger.info(f'Number of trainable parameters: {self.num_params}')

    def __get_num_params__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])

    def __str__(self):
        return super(BaseModel, self).__str__() + f'\nNumber of trainable parameters: {self.num_params}'
