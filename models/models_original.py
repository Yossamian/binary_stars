import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch import nn, Tensor
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, n_output: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, d_hid)
        self.flat = torch.nn.Flatten(1)
        self.init_weights()

        # Last hidden layer and output
        self.hidden3 = nn.Linear(50*d_hid, n_output)
        nn.init.xavier_uniform_(self.hidden3.weight)


    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        batch_size = src.shape[0]
        src = torch.reshape(src[:,:10000].T, (50,batch_size,200))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)
        output = self.decoder(output)
        output = self.flat(output.transpose(0,1))
        output =  self.hidden3(output)
        return output



################################################################################
#################### Attention-based Model #####################################
################################################################################

######################### Simple 1D attention layer ############################
class Attention1D(nn.Module):

    # Constructor
    def __init__(
        self,
        d_input=512,
        d_k=64,
        d_v=64,
    ):
        super().__init__()

        # LAYERS
        # W_k (Dx X Dq)
        self.key_linear = nn.Linear(d_input, d_k)
        # W_q (Dx X Dq)
        self.query_linear = nn.Linear(d_input, d_k)
        # W_v (Dx X Dv)
        self.value_linear = nn.Linear(d_input, d_v)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, X):
        # Create key, query, and value
        K = self.key_linear(X)
        Q = self.query_linear(X)
        V = self.value_linear(X)

        # Scale factor
        D_q = K.shape[-1]

        # Matmuls
        E = torch.matmul(K, Q.permute(0, 2, 1))/math.sqrt(D_q)
        # E = torch.bmm(K, torch.permute(Q, (0, 2, 1)))/math.sqrt(D_q)
        A = self.softmax(E)
        # Y = torch.bmm(A, V)
        Y = torch.matmul(A, V)
        return Y


class MultiheadAttention1D(nn.Module):
    # Constructor
    def __init__(
        self,
        num_heads=8,
        d_model=512
    ):
        super().__init__()

        d_k = d_model // num_heads

        # Create layer for each head
        # self.heads = []
        # for i in range(num_heads):
        #     head = Attention1D(d_input=d_model, d_k=d_k, d_v=d_k)
        #     self.heads.append(head)

        self.heads = nn.ModuleList([Attention1D(d_input=d_model, d_k=d_k, d_v=d_k) for i in range(num_heads)])

        self.linear = nn.Linear(num_heads*d_k, d_model)

        # self.head1=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)
        # self.head2=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)
        # self.head3=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)
        # self.head4=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)

    def forward(self, x):

        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        # a = self.head1(X)
        # b = self.head2(X)
        # c = self.head3(X)
        # d = self.head4(X)
        # out = torch.hstack((a, b, c, d))
        out = torch.cat(outputs, -1)
        y = self.linear(out)

        return y


class EncoderBlock(nn.Module):

    # Constructor
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        # INFO
        self.mhattention = MultiheadAttention1D(num_heads=num_heads, d_model=d_model)
        
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.linear2 = nn.Linear(d_model*4, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, X):
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)

        attention_output = self.mhattention(X)
        attention_output = self.dropout(attention_output) + X
        attention_output = self.norm(attention_output)

        out = F.relu(self.linear1(attention_output))
        out = self.dropout(out)
        out = F.relu(self.linear2(out))
        out = self.dropout(out)
        out = out+attention_output
        out = self.norm(out)
      
        return out


class AttentionBlockModel(nn.Module):

    # Constructor
    def __init__(
        self,
        num_outputs=12,
        input_dim=903,
        d_model=512,
        num_heads=8,
        dropout_p=0.2,
        patch_size=50
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)

        self.remainder = input_dim % patch_size
        num_patches = input_dim // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (n p) -> b n p', p=patch_size),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, d_model),
            nn.LayerNorm(d_model),
        )
        # self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model=d_model)

        self.MHA1 = EncoderBlock(d_model=d_model, num_heads=num_heads)
        self.MHA2 = EncoderBlock(d_model=d_model, num_heads=num_heads)
        self.MHA3 = EncoderBlock(d_model=d_model, num_heads=num_heads)
        self.MHA4 = EncoderBlock(d_model=d_model, num_heads=num_heads)
        self.MHA5 = EncoderBlock(d_model=d_model, num_heads=num_heads)
        self.MHA6 = EncoderBlock(d_model=d_model, num_heads=num_heads)

        self.linear = nn.Linear(d_model, 20)
        self.linear2 = nn.Linear(20*num_patches, num_outputs)


    def forward(self, X):

        #Start with 2 FC layers, with dropout
        X = X[:, :-self.remainder]
        embedding = self.to_patch_embedding(X)
        input = self.pos_embedding(embedding)

        # Seven MHA Blocks
        X = self.MHA1(input)
        X = self.MHA2(X)
        X = self.MHA3(X)
        X = self.MHA4(X)
        X = self.MHA5(X)
        X = self.MHA6(X)

        # Two FC layers at end before output
        out = F.relu(self.linear(X))
        out = torch.reshape(out, (out.shape[0], -1))
        out = self.linear2(out)

        return out


# model definition
class ConvolutionalNet(nn.Module):

    # define model elements
    def __init__(self, num_outputs=12):
        super().__init__()

        self.name = "std_convolutional"

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 3, stride=3, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, stride=3, padding=2)
        self.conv4= nn.Conv1d(64, 64, 3, stride=1, padding=2)
        self.conv_final= nn.Conv1d(64, 3, 1, stride=1, padding=0)

        # Linear layers
        self.linear1 = nn.Linear(312, 80)
        self.linear2 = nn.Linear(80, num_outputs)
        
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