import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
1-d implementation of InceptionNew based Architecture
"""


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
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.dimredmax = nn.Conv1d(in_channels, out_channels_max, 1)

    def forward(self, X):
        one_d = F.relu(self.conv1(X))
        # print('1d shape', one_d.shape)

        three_d = F.relu(self.dimred3(X))
        three_d = F.relu(self.conv3(three_d))
        # print('3d shape', three_d.shape)

        five_d = F.relu(self.dimred5(X))
        five_d = F.relu(self.conv5(five_d))
        # print('5d shape', five_d.shape)

        max_d = F.relu(self.maxpool(X))
        max_d = F.relu(self.dimredmax(max_d))
        # print('maxd shape', max_d.shape)

        # depthwise concatenation
        final = torch.cat((one_d, three_d, five_d, max_d), dim=1)

        return final


class InceptionMultiNet(nn.Module):
    def __init__(self, num_outputs=12):
        super().__init__()
        self.name = "inception"
        self.conv1 = nn.Conv1d(1, 32, 7, stride=2, padding=3)
        self.max1 = nn.MaxPool1d(3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 32, 7, stride=2, padding=3)
        self.max2 = nn.MaxPool1d(3, stride=2, padding=1)

        self.incept1 = InceptionBlock(in_channels=32,
                                      out_channels_1=64,
                                      out_channels_3=128,
                                      out_channels_5=32,
                                      out_channels_max=32,
                                      three_dim_red=96,
                                      five_dim_red=16)

        self.max3 = nn.MaxPool1d(3, stride=2, padding=1)

        self.incept2 = InceptionBlock(in_channels=256,
                                      out_channels_1=64,
                                      out_channels_3=128,
                                      out_channels_5=32,
                                      out_channels_max=32,
                                      three_dim_red=96,
                                      five_dim_red=16)

        self.avg_pool = nn.AvgPool1d(5, stride=2)

        # self.linear1 = nn.Linear(40448,400)
        self.linear1 = nn.Linear(3328, 400)

        self.head1 = HeadBlock(400, 2)
        self.head2 = HeadBlock(400, 2)
        self.head3 = HeadBlock(400, 2)
        self.head4 = HeadBlock(400, 2)
        self.head5 = HeadBlock(400, 2)
        self.head6 = HeadBlock(400, 2)

    def forward(self, X):
        X = torch.unsqueeze(X, 1)
        # print('a', X.shape)
        X = F.relu(self.conv1(X))
        # print('b', X.shape)
        X = self.max1(X)
        # print('c', X.shape)
        X = F.relu(self.conv2(X))
        # print('d', X.shape)
        X = self.max2(X)
        # print('e', X.shape)

        X = self.incept1(X)
        # print('f', X.shape)

        X = self.max3(X)
        # print('g', X.shape)

        X = self.incept2(X)
        # print('h', X.shape)

        X = self.avg_pool(X)
        # print('i', X.shape)

        X = torch.flatten(X, start_dim=1)
        # print('j', X.shape)

        X = F.relu(self.linear1(X))
        # print('k', X.shape)
        # print('l', X.shape)

        x1 = self.head1(X)
        x2 = self.head2(X)
        x3 = self.head3(X)
        x4 = self.head4(X)
        x5 = self.head5(X)
        x6 = self.head6(X)

        X = torch.cat((x1, x2, x3, x4, x5, x6), dim=-1)

        return X

