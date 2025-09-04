import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size, padding, stride, dilation,dropoutprob):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=output_channels,kernel_size=kernel_size,stride=stride, padding=padding, dilation =dilation, bias = True)
        self.batchnorm = nn.BatchNorm3d(output_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropoutprob)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        preactivation = x 
        x = self.activation(x)
        postactivation = self.dropout(x)
        return preactivation, postactivation


class EncoderBlock(nn.Module):
    def __init__(self,input_channel, output_channels, kernel_size, padding, stride, dilation,dropoutProb):
        super().__init__()
        self.conv1 = ConvolutionalBlock(input_channels=input_channel,output_channels=output_channels,kernel_size=kernel_size,padding=padding,stride = stride, dilation=dilation,dropoutprob=dropoutProb)
        self.conv2 = ConvolutionalBlock(input_channels=output_channels,output_channels=output_channels,kernel_size=kernel_size,padding=padding,stride = stride, dilation=dilation,dropoutprob=dropoutProb)
        self.pool = nn.MaxPool3d(kernel_size=2,stride=2)
        
    def forward(self,x):
        _, postactivation = self.conv1(x)
        skip, toPool = self.conv2(postactivation)
        downsample = self.pool(toPool)
        return skip,downsample
    
class DecoderBlock(nn.Module):
    def __init__(self,input_channel, output_channels, kernel_size, padding, stride, dilation, dropoutProb,outputPadding):
        super().__init__()
        
        self.uptranspose = nn.ConvTranspose3d(input_channel,out_channels=output_channels,kernel_size=kernel_size,padding=padding,stride=stride,dilation=dilation,output_padding=outputPadding)
        self.batchnorm = nn.BatchNorm3d(output_channels)
        self.activation = nn.ReLU()
        
        self.conv1 = ConvolutionalBlock(input_channels=output_channels*2,output_channels=output_channels,kernel_size=3,padding=1,stride=1,dilation=1,dropoutprob=dropoutProb)
        self.conv2 = ConvolutionalBlock(input_channels=output_channels,output_channels=output_channels,kernel_size=3,padding=1,stride=1,dilation=1,dropoutprob=dropoutProb)

    def forward(self,x,skip):
        x = self.uptranspose(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = torch.cat([x,skip],dim=1)
        _,postconv1 = self.conv1(x)
        _,postconv2 = self.conv2(postconv1)
        return postconv1
    
class BottleNeckBlock(nn.Module):
    def __init__(self,input_channel, output_channels, kernel_size, padding, stride, dilation,dropoutProb):
        super().__init__()
        self.conv1 = ConvolutionalBlock(input_channels=input_channel,output_channels=output_channels,kernel_size=kernel_size,padding=padding,stride = stride, dilation=dilation,dropoutprob=dropoutProb)
        self.conv2 = ConvolutionalBlock(input_channels=output_channels,output_channels=output_channels,kernel_size=kernel_size,padding=padding,stride = stride, dilation=dilation,dropoutprob=dropoutProb)
    
    def forward(self,x):
        _, x = self.conv1(x)
        _, result = self.conv2(x)
        return result
        
        
"""

device = 'cuda'
x = torch.rand(4,1,48,48,48).to(device)
skip = torch.rand(4,10,96,96,96).to(device)
#model = DecoderBlock(input_channel=1,output_channels=10,kernel_size=3,padding=1,stride=2,dilation=1,dropoutProb=.2,outputPadding=1).to(device)
model = BottleNeckBlock(input_channel=1,output_channels=10,kernel_size=3,padding=1,stride=1,dilation=1,dropoutProb=.2).to(device)
result = model(x)
print(x.shape)
print(skip.shape)
print(result.shape)
"""

