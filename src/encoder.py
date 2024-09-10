import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

 
class VAE_Encoder(nn.Sequential):
    def __init__(self):

        """
        Initializes the VAE encoder with a series of convolutional layers, residual blocks, 
        and an attention block for encoding images into a latent space representation.
        """

        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # since padding is present -> image size is retained !!
            
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # this padding is changed to assymentric padding later in the code !


            VAE_ResidualBlock(128, 256),       
            VAE_ResidualBlock(256, 256),      
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # as we decrease the size -> increase the number of features !! (so increasing the features using the
            # residual block !! )
            VAE_ResidualBlock(256, 512), 
            VAE_ResidualBlock(512, 512), 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512), 
            

            # channels = d_embed = 512 and heads is assigned to 1 in this case !
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512),       
            nn.GroupNorm(32, 512), 
            
            nn.SiLU(), 

            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise): # takes an image !!
        """
        Processes the input image `x` through the encoder layers, applies the reparameterization trick,
        and returns the encoded latent representation.
        """

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
             
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)

        # Use the reparameterisation trick to sample from the latent space
        # from the VAE using the mean + sigma(noise)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # clamp the values so they stay in a range !
        log_variance = torch.clamp(log_variance, -30, 20)
       
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        x = mean + stdev * noise
        
        # Scale by a constant (theory)
        x *= 0.18215
        
        return x
