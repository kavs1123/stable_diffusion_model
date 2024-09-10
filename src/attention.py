import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x, causal_mask=False):
        """
        Applies multi-head self-attention to the input tensor `x`, with optional causal masking.
        
        - Projects the input into queries, keys, and values.
        - Computes attention weights and applies them to the values.
        - Reshapes and projects the result to the original dimensions.
        
        Returns the transformed output tensor.
        """

        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) 
        
        # for masking -> pick the upper triangular matrix and set it to -inf 
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        
        # all attention formula !!

        # Divide by d_k (Dim / H). 
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 
        output = weight @ v

        output = output.transpose(1, 2) 

        # all of them are concatenated here 
        output = output.reshape(input_shape) 

        # the final W0 matrix is multiplied with this using the linear layer !
        output = self.out_proj(output) 
        
        return output

# for the decoder -> the keys and values are from the encoder and the query is from the
# decoder 
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        """
        Performs the forward pass of a multi-head self-attention mechanism.
        
        - Projects the input tensors `x` and `y` into queries, keys, and values.
        - Reshapes and transposes these projections for multi-head attention.
        - Computes the attention weights by scaling and applying softmax.
        - Applies the attention weights to the values to produce the output.
        - Reshapes the output back to the original input dimensions.
        """
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output
    
