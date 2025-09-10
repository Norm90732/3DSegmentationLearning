import torch
import torch.nn as nn
import torch.nn.functional as F
from layerModules import ConvolutionalBlock, EncoderBlock,DecoderBlock,FeedForward,Encoder,DecoderBlockForTransU #change to .layerModules after 

class MultiScalePatchEmbed(nn.Moduel):
    def __init__(self,numSkipConnections,projectedPatch):
        super().__init__()
        