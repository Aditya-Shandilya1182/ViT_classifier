import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.fc2 = nn.Linear(embed_dim * expansion_factor, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
