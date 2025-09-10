from .layerModules import ConvolutionalBlock, EncoderBlock,DecoderBlock,FeedForward,Encoder,DecoderBlockForTransU
import torch
import torch.nn as nn
class TransUnet(nn.Module):
    def __init__(self,heightImg,widthImg,depthImg,patchsize,input_channels, output_channel, dropoutProb,embedDim,ffDim,numHeads,dropoutMLP,dropoutAttention,numEncoderBlock):
        super().__init__()
        self.heightImg = heightImg
        self.widthImg = widthImg
        self.depthImg = depthImg
        self.patchsize = patchsize
    
        
        self.encoding1 = EncoderBlock(input_channel=input_channels,output_channels=32, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        self.encoding2 = EncoderBlock(input_channel=32,output_channels=64, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        self.encoding3 = EncoderBlock(input_channel=64,output_channels=128, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        self.encoding4 = EncoderBlock(input_channel=128,output_channels=256, kernel_size=3, padding=1, stride=1, dilation=1,dropoutProb=dropoutProb)
        
        self.numPatches = (self.depthImg*self.widthImg* self.heightImg) // self.patchsize**3
        
        self.generatePatches = nn.Conv3d(in_channels=256,out_channels=embedDim,kernel_size=self.patchsize,stride=self.patchsize)
        
        
        
        self.posEmbedding = nn.Embedding(num_embeddings=self.numPatches,embedding_dim=embedDim)
        
        self.encoderBlocks = nn.ModuleList(
            [Encoder(embedDim=embedDim,ffDim=ffDim,numHeads=numHeads,dropoutMLP=dropoutMLP,dropoutAttention=dropoutAttention) for _ in range(numEncoderBlock)]
        )
        
        self.dropout = nn.Dropout(dropoutProb)
        self.decoder1 = DecoderBlockForTransU(input_channel=embedDim, output_channels=256,skipdim=256, kernel_size=6, padding=0, stride=6, dilation=1, dropoutProb=dropoutProb,outputPadding=0)
        self.decoder2 = DecoderBlock(input_channel=256, output_channels=128, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)
        self.decoder3 = DecoderBlock(input_channel=128, output_channels=64, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)
        self.decoder4 = DecoderBlock(input_channel=64, output_channels=32, kernel_size=3, padding=1, stride=2, dilation=1, dropoutProb=dropoutProb,outputPadding=1)
        
        self.onebyone = nn.Conv3d(in_channels=32,out_channels=output_channel,kernel_size=1,padding=0,stride=1,dilation=1)
        
    def forward(self,x):
        skip1, encoder1 = self.encoding1(x)
        skip2, encoder2 = self.encoding2(encoder1)
        skip3, encoder3 = self.encoding3(encoder2)
        skip4, encoder4 = self.encoding4(encoder3)
        
        #Project for Transformer Layer
        #Batch, Channel, Depth, Height, Width        
        #Generate patches - > flatten -> embed -> dropout -> 
        patches = self.generatePatches(encoder4)
        patchShape = patches.shape
        patches = torch.flatten(patches,2) #B, embed,seq len
        patches = torch.permute(patches,(0,2,1)) # Batch, seq len,embed
        current_device = x.device 
        posEmbed = (torch.arange(0,patches.shape[1],step=1,dtype=torch.long)).to(current_device)
 
    
        embedding = self.posEmbedding(posEmbed)
        embedding= embedding.unsqueeze(0)
        
        output = self.dropout(patches + embedding)
        encoderResult = output
        for block in self.encoderBlocks:
            encoderResult = block(encoderResult)
        #Now it needs to be reshaped back to how it was before transformer. 
        encoderResult = torch.permute(encoderResult,(0,2,1))
        reshaped = encoderResult.reshape(patchShape) #Now back to B,C,D,H,W
        #decoder upsampling
        decoder1 = self.decoder1(reshaped,skip4)
        decoder2 = self.decoder2(decoder1,skip3)
        decoder3 = self.decoder3(decoder2,skip2)
        decoder4 = self.decoder4(decoder3,skip1)
        finalOutput = self.onebyone(decoder4)
        
        
        
        
        return finalOutput

#device = 'cuda'
#x = torch.rand(4,1,96,96,96).to(device)
#model = TransUnet(heightImg=12,widthImg=12,depthImg=12,patchsize=3,input_channels=1, output_channel=2, dropoutProb=0.0,embedDim=768,ffDim=2048,numHeads=4,dropoutMLP=0,dropoutAttention=0,numEncoderBlock=1).to(device)
#result = model(x)
#print(x.shape)
#print(result.shape)

