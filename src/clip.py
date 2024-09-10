import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        """
        Initializes the CLIPEmbedding module with token and positional embeddings.
        """
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
    
    def forward(self, tokens):
        """
        Computes token embeddings and adds positional embeddings.
        """
        x = self.token_embedding(tokens)
        x += self.position_embedding
        
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        """
        Initializes the transformer block with LayerNorm, self-attention, and feedforward layers.
        """
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
       
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):

        """
        Applies layer normalization, self-attention, and feedforward operations with residual connections.
        """
        residue = x
        

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residue

        ## FEEDFORWARD LAYER

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)     
        x = x * torch.sigmoid(1.702 * x)   
        x = self.linear_2(x)
        
        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self):
        """
        Initializes the CLIP model with token embeddings, multiple transformer layers, and a final LayerNorm.
        """
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Processes input tokens through embedding, multiple transformer layers, and final layer normalization.
        """
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens)

        for layer in self.layers: 
            state = layer(state)
        output = self.layernorm(state)
        
        return output