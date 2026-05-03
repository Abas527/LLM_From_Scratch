import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .embedding import Embedding

class selfAttentionHead(nn.Module):

    def __init__(self,n_embed,head_size):   #n_embed ÷ num_heads=head_size
        super().__init__()
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
    
    def forward(self,x):
        B,T,C=x.shape   #B=batch size, T=sequence length, C=embedding dimension
        q=self.query(x) # (B,T,head_size)
        k=self.key(x)   # (B,T,head_size)
        v=self.value(x) # (B,T,head_size)

        attention_weight=q @ k.transpose(-2,-1)  #swaps last two dimensions: (B, T, head_size) → (B, head_size, T)  and multiply with q:(B,T,head_size) @ (B, head_size, T) = (B,T,T)
        attention_weight=attention_weight/math.sqrt(k.size(-1)) # (B,T,T)

        # masking
        mask=torch.tril(torch.ones(T,T)).to(x.device) # (T,T) lower triangular matrix with ones on and below the diagonal
        attention_weight=attention_weight.masked_fill(mask==0,-float('inf'))


        attention_weight=F.softmax(attention_weight,dim=-1) # (B,T,T)
        out=attention_weight @ v # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,n_embed,num_heads):
        super().__init__()
        assert n_embed % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads=num_heads
        self.head_size=n_embed//num_heads

        self.heads=nn.ModuleList([selfAttentionHead(n_embed,self.head_size) for _ in range(num_heads)])

        self.proj=nn.Linear(n_embed,n_embed)  #project the concatenated output of all heads back to the original embedding dimension

    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1) #concatenate the output of all heads along the last dimension
        out=self.proj(out) #project the concatenated output back to the original embedding dimension
        return out


class TransformerBlock(nn.Module):
    def __init__(self,n_embed,num_heads):
        super().__init__()
        self.ln1=nn.LayerNorm(n_embed)
        self.ln2=nn.LayerNorm(n_embed)
        self.attn=MultiHeadAttention(n_embed,num_heads)
        self.mlp=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),  #feedforward network with hidden layer of size 4 times the embedding dimension
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed)   #project back to the original embedding dimension
        )
    def forward(self,x):
        x=x+self.attn(self.ln1(x)) #residual connection
        x=x+self.mlp(self.ln2(x))  #residual connection
        return x


class LLM(nn.Module):
    def __init__(self,vocab_size,block_size,n_embed,num_heads,num_layers):
        super().__init__()
        self.block_size=block_size
        self.embed=Embedding(vocab_size,block_size,n_embed)

        self.blocks=nn.Sequential(*
                [TransformerBlock(n_embed,num_heads) for _ in range(num_layers)])
        
        self.ln_f=nn.LayerNorm(n_embed)  #final layer norm
        self.head=nn.Linear(n_embed,vocab_size)  #final linear layer to project to vocab size

    def forward(self,x):
        x=self.embed(x)  # (B,T,n_embed)
        x=self.blocks(x) # (B,T,n_embed)
        x=self.ln_f(x)   # (B,T,n_embed)
        logits=self.head(x) # (B,T,vocab_size)
        return logits