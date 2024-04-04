from abc import abstractmethod
from functools import partial
import math

import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.GLIGEN.demo.gligen.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from modules.GLIGEN.demo.gligen.ldm.modules.attention import SpatialTransformer
from torch.utils import checkpoint

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context, objs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, objs)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x




class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, emb )
        else:
            return self._forward(x, emb) 


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h




class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        use_scale_shift_norm=False,
        transformer_depth=1,          
        positive_len = 768, # this is pre-processing embedding len for each 'obj/box'    
        context_dim=None,  
        fuser_type = None,    
        is_inpaint = False,
        is_style = False,           
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.positive_len = positive_len
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.is_inpaint = is_inpaint
        self.is_style = is_style
        self.use_o2 = False # This will be turned into True by externally if use o2 durining training
        assert fuser_type in ["gatedSA", "gatedCA"]


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )


        total_in_channels = in_channels+in_channels+1 if self.is_inpaint else in_channels
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, total_in_channels, model_channels, 3, padding=1))])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        # = = = = = = = = = = = = = = = = = = = = Down Branch = = = = = = = = = = = = = = = = = = = = #
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ ResBlock(ch,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=mult * model_channels,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm,) ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint))
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1: # will not go to this downsample branch in the last feature
                out_ch = ch
                self.input_blocks.append( TimestepEmbedSequential( Downsample(ch, conv_resample, dims=dims, out_channels=out_ch ) ) )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        dim_head = ch // num_heads

        # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]


        # = = = = = = = = = = = = = = = = = = = = BottleNeck = = = = = = = = = = = = = = = = = = = = #
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint),
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm))



        # = = = = = = = = = = = = = = = = = = = = Up Branch = = = = = = = = = = = = = = = = = = = = #

        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ ResBlock(ch + ich,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=model_channels * mult,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm) ]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append( SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint) )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append( Upsample(ch, conv_resample, dims=dims, out_channels=out_ch) )
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))


        # self.output_blocks = [ R  R  RU | RT  RT  RTU |  RT  RT  RTU  |  RT  RT  RT  ]


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if self.is_style:
            from .positionnet_with_image  import PositionNet
        else:
            from .positionnet  import PositionNet
        self.position_net = PositionNet(positive_len=positive_len, out_dim=context_dim)
        



    def forward_position_net(self,input):
        if ("boxes" in input):
            boxes, masks, text_embeddings = input["boxes"], input["masks"], input["text_embeddings"]
            _ , self.max_box, _ = text_embeddings.shape
        else: 
            dtype = input["x"].dtype
            batch = input["x"].shape[0]
            device = input["x"].device
            boxes = th.zeros(batch, self.max_box, 4,).type(dtype).to(device) 
            masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
            text_embeddings = th.zeros(batch, self.max_box, self.positive_len).type(dtype).to(device) 
        if self.training and random.random() < 0.1: # random drop for guidance  
            boxes, masks, text_embeddings = boxes*0, masks*0, text_embeddings*0
  
        objs = self.position_net( boxes, masks, text_embeddings ) # B*N*C 

        return objs


    


    def forward_position_net_with_image(self,input):

        if ("boxes" in input):
            boxes = input["boxes"] 
            masks = input["masks"]
            text_masks = input["text_masks"]
            image_masks = input["image_masks"]
            text_embeddings = input["text_embeddings"]
            image_embeddings = input["image_embeddings"]
            _ , self.max_box, _ = text_embeddings.shape
        else: 
            dtype = input["x"].dtype
            batch = input["x"].shape[0]
            device = input["x"].device
            boxes = th.zeros(batch, self.max_box, 4,).type(dtype).to(device) 
            masks = th.zeros(batch, self.max_box).type(dtype).to(device)
            text_masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
            image_masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
            text_embeddings =  th.zeros(batch, self.max_box, self.positive_len).type(dtype).to(device) 
            image_embeddings = th.zeros(batch, self.max_box, self.positive_len).type(dtype).to(device) 
        
        if self.training and random.random() < 0.1: # random drop for guidance  
            boxes = boxes*0
            masks = masks*0
            text_masks = text_masks*0
            image_masks = image_masks*0
            text_embeddings = text_embeddings*0
            image_embeddings = image_embeddings*0
  
        objs = self.position_net( boxes, masks, text_masks, image_masks, text_embeddings, image_embeddings ) # B*N*C 
        
        return objs





    def forward(self, input):

        if self.is_style:
            objs = self.forward_position_net_with_image(input)
        else:
            objs = self.forward_position_net(input)

        
        hs = []

        t_emb = timestep_embedding(input["timesteps"], self.model_channels, repeat_only=False)
        if self.use_o2:
            t_emb = t_emb.to(th.float16)  # not sure why apex will not cast this 
        emb = self.time_embed(t_emb)


        h = input["x"]
        if self.is_inpaint:
            print('enter2pant:', input["inpainting_extra_input"].size())
            print('h:', h.size())
            if  input["inpainting_extra_input"].size(2) != h.size(2):
                print("enter")
                input["inpainting_extra_input"] = F.interpolate(input["inpainting_extra_input"], size=(h.size(2), input["inpainting_extra_input"].size(3)), mode='bilinear', align_corners=False)
                print("tempsize",input["inpainting_extra_input"].size())
            h = th.cat( [h, input["inpainting_extra_input"]], dim=1 )
        context = input["context"]
        

        for module in self.input_blocks:
            h = module(h, emb, context, objs)
            hs.append(h)
        print(h.size())
        h = self.middle_block(h, emb, context, objs)
        print(h.size())
        for module in self.output_blocks:
            temp_h = hs.pop()
            print("temp_h.size",temp_h.size())
            print("h.size",h.size())
            if  temp_h.size(2) != h.size(2):
                print("enter")
                temp_h = F.interpolate(temp_h, size=(h.size(2), temp_h.size(3)), mode='bilinear', align_corners=False)
                print("tempsize",temp_h.size())
            h = th.cat([h, temp_h], dim=1)
            h = module(h, emb, context, objs)

        return self.out(h)










