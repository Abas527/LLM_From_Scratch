import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,vocab_size,max_len,n_embed):
        super(Embedding,self).__init__()
        self.tok_emb = nn.Embedding(vocab_size,n_embed)
        self.pos_emb = nn.Parameter(torch.randn(1,max_len,n_embed))  #random numbers from a normal distribution
        # nn.Paramter() is alearnable parameter
    
    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb[:,:x.size(1),:]
        return tok_emb + pos_emb