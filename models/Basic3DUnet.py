import torch
import torch.nn as nn
from layerModules import ConvolutionalBlock,EncoderBlock,DecoderBlock,BottleNeckBlock


class Basic3DUnet(nn.Module):
    def __init__(self,input_channels, output_channel, dropoutProb):
        super().__init__()
        #Encoder Blocks
        self.encoding1 = EncoderBlock(input_channel=input_channels,output_channels=64, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        self.encoding2 = EncoderBlock(input_channel=64,output_channels=128, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        self.encoding3 = EncoderBlock(input_channel=128,output_channels=256, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        self.encoding4 = EncoderBlock(input_channel=256,output_channels=512, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        
        #Bottle Neck Block
        self.bottleneck = BottleNeckBlock(input_channel=512, output_channels=1024, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        
        #Decoder blocks
        self.decoder1 = DecoderBlock(input_channel=1024, output_channels=512, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)
        self.decoder2 = DecoderBlock(input_channel=512, output_channels=256, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)
        self.decoder3 = DecoderBlock(input_channel=256, output_channels=128, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)
        self.decoder4 = DecoderBlock(input_channel=128, output_channels=64, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)

        #One by one conv
        self.onebyone = nn.Conv3d(in_channels=64,out_channels=output_channel,kernel_size=1,padding=0,stride=1,dilation=1)
    
    def forward(self,x):
        skip1, encode1 = self.encoding1(x)
        skip2, encode2 = self.encoding2(encode1)
        skip3, encode3 = self.encoding3(encode2)
        skip4, encode4 = self.encoding4(encode3)
        
        bottleneckout = self.bottleneck(encode4)
        
        decode1 = self.decoder1(bottleneckout,skip4)
        decode2 = self.decoder2(decode1,skip3)
        decode3 = self.decoder3(decode2,skip2)
        decode4 = self.decoder4(decode3,skip1)
        
        result = self.onebyone(decode4)
        return result
 
device = 'cuda'
x = torch.rand(4,1,96,96,96).to(device)
model = Basic3DUnet(input_channels=1,output_channel=10,dropoutProb=.1).to(device)
result = model(x)
print(x.shape)
print(result.shape)

