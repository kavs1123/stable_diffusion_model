import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):

        residue = x 

        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        
        # no of tokens = height* width and the embedding is just the number of features
        # and we calculate the attenion btw these tokens !
        x = x.transpose(-1, -2)
        # Perform self-attention WITHOUT mask
        x = self.attention(x)

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        # add to the input !!
        x += residue
        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) # 1 x1 conv to keep the same size as output as its strided by 2 
    
    def forward(self, x):
        """
        Processes the input tensor through a series of normalization, activation, and convolution operations,
        and adds a residual connection.
        """

        residue = x

        x = self.groupnorm_1(x)  # Apply first Group Normalization
        x = F.silu(x)           # Apply SiLU activation

        x = self.conv_1(x)      # Apply first convolution layer

        x = self.groupnorm_2(x)  # Apply second Group Normalization
        x = F.silu(x)           # Apply SiLU activation

        x = self.conv_2(x)      # Apply second convolution layer

        return x + self.residual_layer(residue)  # Add residual connection

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        """
        Complete architecture of the VAE Decoder
        """
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            

            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
          # as size increases , the no of features decreases 
            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            nn.Upsample(scale_factor=2), 
            

            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 
            

            nn.SiLU(), 
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
   
        x /= 0.18215

        for module in self: 
            x = module(x)
        return x
    
