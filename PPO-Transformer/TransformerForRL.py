import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class Attention(nn.Module):
  def __init__(self, h_dim, max_T, n_heads, drop_p):
    
    super().__init__()
    self.n_heads = n_heads
    self.max_T = max_T
    self.q_net = nn.Linear(h_dim, h_dim) 
    self.k_net = nn.Linear(h_dim, h_dim) 
    self.v_net = nn.Linear(h_dim, h_dim)
    self.proj_net = nn.Linear(h_dim, h_dim)
    self.att_drop = nn.Dropout(drop_p)
    self.proj_drop = nn.Dropout(drop_p)

  def forward(self, x):
    B, T, C = x.shape # batch size, seq length, h_dim * n_heads
    N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

    # rearrange q, k, v as (B, N, T, D)
    q = self.q_net(x).view(B, T, N, D).transpose(1,2) 
    k = self.k_net(x).view(B, T, N, D).transpose(1,2)
    v = self.v_net(x).view(B, T, N, D).transpose(1,2)

    # weights (B, N, T, T)
    # @: The matrix multiplications are done between the last two dimensions, like torch.bmm
    weights = q @ k.transpose(2,3) / math.sqrt(D)
    # normalize weights
    normalized_weights = F.softmax(weights, dim=-1)

    # attention (B, N, T, D)
    attention = self.att_drop(normalized_weights @ v)

    # gather heads and project (B, N, T, D) -> (B, T, N*D)
    attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

    out = self.proj_drop(self.proj_net(attention))
    return out

class Block(nn.Module):

  def __init__(self, h_dim, max_T, n_heads, drop_p):
    super().__init__()

    self.attention = Attention(h_dim, max_T, n_heads, drop_p)
    self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p),
        )
    self.ln1 = nn.LayerNorm(h_dim)
    self.ln2 = nn.LayerNorm(h_dim)

  def forward(self, x):
    # Pre-LN
    x = x + self.attention(self.ln1(x) ) # residual
    x = x + self.mlp(self.ln2(x)) # residual
    return x

class TransformerForRL(nn.Module):

  def __init__(self, state_dim, output_dim, n_blocks, h_dim, T, n_heads, drop_p):
    super().__init__()
    
    self.h_dim = h_dim
    self.T = T
    self.state_to_tokens = nn.Linear(state_dim, h_dim*T)

    # parameter = trainable weight matrix, size: (1, T, h_dim)
    init_param_vals = torch.randn(1, T, h_dim) / math.sqrt(h_dim)
    self.position_embedding = nn.Parameter(init_param_vals)

    # transformer blocks
    blocks = [Block(h_dim, T, n_heads, drop_p) for _ in range(n_blocks)]
    self.transformer = nn.Sequential(*blocks)

    # projection head
    self.proj_head = nn.Linear(h_dim*T, output_dim)

  def forward(self, x):
    if len(x.shape) == 1:
        batch_size = 1
    else:
        batch_size = x.shape[0]

    # project state to tokens and pos embedding
    h = self.state_to_tokens(x).view(batch_size, self.T, self.h_dim) + self.position_embedding

    # transformer and prediction
    h = self.transformer(h)
    pred = self.proj_head(h.view(batch_size, -1))
    return pred 