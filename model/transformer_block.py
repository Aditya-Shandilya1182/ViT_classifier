import torch.nn as nn
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_expansion_factor=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = attn_out + x
        ff_out = self.feed_forward(self.norm2(x))
        return ff_out + x
