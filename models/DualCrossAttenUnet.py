import torch
import torch.nn as nn
import torch.nn.functional as F
from layerModules import ConvolutionalBlock, EncoderBlock,DecoderBlock,FeedForward,Encoder,DecoderBlockForTransU #change to .layerModules after 

class MultiScalePatchEmbed(nn.Module): 
    def __init__(self,skipchannels:list,outputPatch,projectionDim):
        super().__init__()
        #R(C X D/2^i-1 x H/2^i-1 x W/2^i-1) --> AvgPool3D(Skips)
        #Reshape --> B,C, D*H*W -> permute -> project
        self.outputPatch = outputPatch
        self.projections = nn.ModuleList([nn.Linear(in_features=channels,out_features=channels) for channels in skipchannels])
        
    def forward(self,*numSkips):
        avgPoolProjection = [F.adaptive_avg_pool3d(input=skip,output_size=(self.outputPatch,self.outputPatch,self.outputPatch)) for skip in numSkips]
        #Flatten then reshape to B, Seq, Embed dim 
        flattenReshape = [torch.flatten(tensor,2).permute(0,2,1) for tensor in avgPoolProjection] 
        embedded = [self.projections[i](flattenReshape[i]) for i in range(len(numSkips))]
        return embedded

#Now Tokenized, create ChannelCrossAttention
class ChannelCrossAttention(nn.Module):
    def __init__(self,projectionDim,outputPatch,skipchannels:list,numHeads,dropout):
        super().__init__()
        #First is layer norm of a list
        self.layernorm = nn.ModuleList([nn.LayerNorm(normalized_shape=skip) for skip in skipchannels])
        self.projectionPerChannel = nn.ModuleList([nn.Linear(in_features=channels,out_features=projectionDim) for channels in skipchannels])
        #number of channels count
        self.sum = sum(skipchannels)
        self.totalProjection = nn.Linear(in_features=self.sum,out_features=projectionDim)
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=outputPatch,num_heads=numHeads,dropout=dropout) for _ in range(len(skipchannels))])
    def forward(self,*numEmbeddedVectors):
        normed = [self.layernorm[i](numEmbeddedVectors[i]) for i in range(len(numEmbeddedVectors))]
        
        #Concat to K.T and V in Attention -> Patch Size by Total number of channels
        concated = torch.cat(normed,dim=2)
        #Projection -> K.T and Projection V
        projectedPerChannel = [self.projectionPerChannel[i](normed[i]) for i in range(len(numEmbeddedVectors))]
        print(concated.shape)
        projectedConcat = self.totalProjection(concated)
        print(projectedConcat.shape)
        attention_result = []
        for i in range(len(projectedPerChannel)):
            attention,_ = self.attention[i](
            query=projectedPerChannel[i].permute(0,2,1),key=projectedConcat.permute(0,2,1),value=projectedConcat.permute(0,2,1))
            attention_result.append(attention.permute(0,2,1))
        return attention_result

class SpatialCrossAttention(nn.Module):
    def __init__(self,skipchannels:list,projectionDim,numHeads,dropout):
        super().__init__()
        self.layernorm = nn.ModuleList([nn.LayerNorm(normalized_shape=skip) for skip in skipchannels])
        self.projectionPerChannel = nn.ModuleList([nn.Linear(in_features=channels,out_features=projectionDim) for channels in skipchannels])
        self.sum = sum(skipchannels)
        self.totalProjection = nn.Linear(in_features=self.sum,out_features=projectionDim)
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=projectionDim,num_heads=numHeads,dropout=dropout) for _ in range(len(skipchannels))])
    def forward(self,*numEmbeddedVectors):
        normed = [self.layernorm[i](numEmbeddedVectors[i]) for i in range(len(numEmbeddedVectors))]
        concated = torch.cat(normed,dim=2)
        projectedPerChannel = [self.projectionPerChannel[i](normed[i]) for i in range(len(numEmbeddedVectors))]
        projectedConcat = self.totalProjection(concated)
        attention_result = []
        attention_final = []
        for i in range(len(projectedPerChannel)):
            attention,_ = self.attention[i](
            query=projectedConcat,key=projectedConcat,value=projectedPerChannel[i])
            attention_result.append(attention)
        return attention_result
    
class DCABLOCK(nn.Module):
    def __init__(self,skipchannels:list,projectionDim,outputPatch,numHeads,dropout):
        super().__init__()
        self.activation = nn.GELU()
        self.layernormfinal = nn.ModuleList([nn.LayerNorm(normalized_shape=skip) for skip in skipchannels])
        self.ChannelCrossAttention = ChannelCrossAttention(projectionDim=projectionDim,outputPatch=outputPatch,skipchannels=skipchannels,numHeads=numHeads,dropout=dropout)
        self.SpatialCrossAttention = SpatialCrossAttention(skipchannels=skipchannels,projectionDim=projectionDim,numHeads=numHeads,dropout=dropout)
        
    def forward(self,*numEmbeddedVectors):
        residual1 = numEmbeddedVectors
        outputChannel = self.ChannelCrossAttention(*numEmbeddedVectors)
        result1 = outputChannel+residual1
        residual2 = result1
        outputSpatial = self.SpatialCrossAttention(result1)
        result2 = outputSpatial+ residual2
        #layernorm=[self.activation(self.layernormfinal[i](result2[i])) for i in range(len(result2))]
        return result2
        
        
        

device = 'cuda'
skip1 = torch.rand(4,512,10).to(device)
skip2 = torch.rand(4,512,20).to(device)
skip3 =torch.rand(4,512,30).to(device)
skip4 =torch.rand(4,512,40).to(device)
model = DCABLOCK(skipchannels=[10,20,30,40],projectionDim=768,outputPatch=512,numHeads=1,dropout=0).to(device)
result = model.forward(skip1,skip2,skip3,skip4)
for heyhihello in result:
    print(heyhihello.shape)
#model = ChannelCrossAttention(projectionDim=768,outputPatch=512,skipchannels=[10,20,30,40],numHeads=1,dropout=0.0).to(device)
#second_model = SpatialCrossAttention(skipchannels=[10,20,30,40],projectionDim=768,numHeads=1,dropout=0.0).to(device)
#result = model.forward(skip1,skip2,skip3,skip4)
#result2 = model.forward(skip1,skip2,skip3,skip4)
#for hello in result:
    #print(f"Channel Cross Attention {hello.shape}")
#for hey in result2:
    #print(f"Spatial Cross Attention {hey.shape}")




        
        