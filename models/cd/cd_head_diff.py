# Code for Change Detection head with Feature Difference (Subtraction)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cd.se import ChannelSpatialSELayer

def get_in_channels(feat_scales, inner_channel, channel_mults):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3: #256 x 256
            in_channels += inner_channel*channel_mults[0]
        elif scale < 6: #128 x 128
            in_channels += inner_channel*channel_mults[1]
        elif scale < 9: #64 x 64
            in_channels += inner_channel*channel_mults[2]
        elif scale < 12: #32 x 32
            in_channels += inner_channel*channel_mults[3]
        elif scale < 15: #16 x 16
            in_channels += inner_channel*channel_mults[4] # channel muliplier가 None 
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14') 
    return in_channels

class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim*len(time_steps), dim, 1)
            if len(time_steps)>1
            else None,
            nn.ReLU()
            if len(time_steps)>1
            else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class cd_head_diff(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_mults=None, img_size=256, time_steps=None):
        super(cd_head_diff, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_mults)
        self.img_size       = img_size
        self.time_steps     = time_steps

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_mults)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_mults)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats_A, feats_B):
        # Decoder
        lvl=0
        for layer in self.decoder: # 한 layer는 한 Block 또는 AttentionBlock (마지막 Block은 그냥 block , attention X )
            if isinstance(layer, Block): # layer = Block인 경우  
                f_A = feats_A[0][self.feat_scales[lvl]]
                f_B = feats_B[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)): # layer = Block인 경우  
                    f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)
                    f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
    
                diff = torch.abs( layer(f_A)  - layer(f_B) ) #두 feature map의 pixel값 차이 -> feature exchange 를 써보면 좋지 않을까 ? change detection network에서 개선할 여지가 많아보임. 
                if lvl!=0:
                    diff = diff + x # residual learning # attention block을 interpolation한 것을 현재 difference에 더해줌. 
                lvl+=1
            else: # layer != Block인 경우 
                diff = layer(diff) # 이 부분을 이해하지 못함.. -> block으로 감싸는 것
                x = F.interpolate(diff, scale_factor=2, mode="bilinear") # bilinear interpolation (2배로 키우는 upsampling) 

        # Classifier
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x))) #conv2d 2번 쌓고 ReLU 적용 

        return cm # Change detection map을 return 

    