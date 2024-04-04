import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F



class GroundingDownsampler(nn.Module):
    def __init__(self, resize_input=256, out_dim=8):
        super().__init__()
        self.resize_input = resize_input
        self.out_dim = out_dim 

        self.layers = nn.Sequential(
            nn.Conv2d(3,4,4,2,1),
            nn.SiLU(),
            nn.Conv2d(4,self.out_dim,4,2,1)
        )

    def forward(self, grounding_extra_input):

        out = torch.nn.functional.interpolate(grounding_extra_input, (self.resize_input,self.resize_input), mode='bicubic')
        out = self.layers(out)

        assert out.shape[1] == self.out_dim 
        return out


