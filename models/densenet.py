import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
    x = F.relu(x)
    x = self.conv(x)
    x = self.pool(x)

    return x


class DenseLayer(nn.Module):
  # Dense Layer
  # 
  def __init__(self, input_features, bn_size, growth, dropout_rate=0):
    # bn_size is used to reduce the number of features, with a 1x1 convolution
    # growth is the number of new features added per layer
    super().__init__()
    self.BN1 = nn.BatchNorm1d(input_features)
    self.conv1 = nn.Conv1d(input_features, bn_size*growth, kernel_size=1)
    self.BN2 = nn.BatchNorm1d(bn_size*growth)
    self.conv2 = nn.Conv1d(bn_size*growth, growth, kernel_size=3, stride=1, padding=1)
    self.dropout = nn.Dropout(p=dropout_rate)

  def concat_features(self, inputs):
    # Combines the input list - features from all previous layers
    concatenated_feature_maps = torch.cat(inputs, dim=1)
    return(concatenated_feature_maps)

  def bottleneck(self, inputs):
    # Use 1-d convolutions as bottleneck
    x = F.relu(self.BN1(inputs))
    x = self.conv1(x)
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
    x = self.bottleneck(x)
    x = F.relu(self.BN2(x))
    # Second convolution is kernel-3, reduces output to just growth (K) feature maps
    new_features = self.conv2(x)
    # Dropout, if desired
    new_features = self.dropout(new_features)

    return new_features


class DenseBlock(nn.ModuleDict):
  # DenseBlock is a ModuleDICT, not a Module
  # Series of DenseLayers, where each layer receives as an input all of the feature maps from previous layers
  # As a result, the input size of each dense layer increases by growth_rate with each iteration

  def __init__(self, num_layers, num_input_features, bn_size, growth_rate, dropout=0):
    super().__init__()
    for i in range(num_layers):
      # Each layer has an input_features size that increases by growth_rate (K) each layer
      layer = DenseLayer(
        input_features=num_input_features+(i*growth_rate),
        bn_size = bn_size,
        growth=growth_rate,
        dropout_rate=dropout,
      )
      self.add_module(f"denselayer{i+1}", layer)
    
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
    return concatenated_features


class DenseNet(nn.Module):
  def __init__(self, growth_rate=20, block_config=(3, 6, 9, 12), num_init_features=12, bn_size=4, dropout=0, output=12):
    super().__init__()

    self.features = nn.Sequential(
      OrderedDict([
        ("conv0", nn.Conv1d(1, num_init_features, kernel_size=7, stride=2, padding=3)),
        ("bn0", nn.BatchNorm1d(num_init_features)),
        ("relu0", nn.ReLU()),
        ("avgpool0", nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
      ])
    )

    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
      block = DenseBlock(
        num_layers = num_layers,
        num_input_features = num_features,
        bn_size = bn_size,
        growth_rate = growth_rate,
        dropout = dropout,
      )

      self.features.add_module(f"denseblock{i+1}", block)
      num_features = num_features+(num_layers*growth_rate)
      if i != len(block_config)-1:
        transition = TransitionBlock(
          input_features = num_features,
          output_features = num_features//2,
        )
        self.features.add_module(f"transition{i+1}", transition)
        num_features = num_features//2

    self.final_pool = nn.AdaptiveAvgPool1d(output_size=1)
    self.linear_final = nn.Linear(num_features, output)

    # Official init from torch repo.
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

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
    features = self.linear_final(features)

    return features




