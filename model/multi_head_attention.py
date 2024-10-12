import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum("bhqd, bhkd -> bhqk", [queries, keys])
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum("bhqk, bhvd -> bhqd", [attention, values])
        out = out.reshape(batch_size, num_patches, embed_dim)
        return self.fc_out(out)
