import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import List

from functools import wraps
import torch.distributed as dist
import copy


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = torch.nn.BatchNorm2d
    normalization_layer = layer(in_channels)
    return normalization_layer

    
class Self_Attn(nn.Module):
    def __init__(self, num_class=2, attention_dim=128, is_training=True, linformer=True, valid_mask=None, factor=8, downsample_type='nearest'):
        super(Self_Attn, self).__init__()
    
        self.num_class = num_class
        self.attention_dim = attention_dim
        self.is_training = is_training
        self.linformer = linformer
        self.valid_mask = valid_mask
        self.factor = factor
        self.downsample_type = downsample_type

        self.conv1 = nn.Conv2d(128, attention_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(2, self.num_class, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.num_class)

    def forward(self, conv_layer_list, logits):
    
        h, w = logits.shape[2], logits.shape[3]
        
        conv_layer_list = [
            F.interpolate(conv, size=(h, w), mode='bilinear', align_corners=True)
            for conv in conv_layer_list
        ]
        conv_layer_merged = torch.cat(conv_layer_list, dim=1)
        conv_layer_merged = conv_layer_merged.detach()
        score = logits.detach()
        value_dim = score.shape[1]

        if self.downsample_type == 'bilinear':
            resize_fn = F.interpolate
        else:
            resize_fn = F.interpolate
        
        k = self.conv1(conv_layer_merged)
        q = self.conv1(conv_layer_merged)

        q = q.view(-1, h * w, self.attention_dim)
        if self.valid_mask is not None:
            valid_mask_q = self.valid_mask.view(-1, h * w, 1)

        if self.linformer:
            k = resize_fn(k, size=((h // self.factor + 1), (w // self.factor + 1)), mode='bilinear', align_corners=True)
            k = k.view(-1, (h // self.factor + 1) * (w // self.factor + 1), self.attention_dim)

            if self.valid_mask is not None:
                valid_mask_k = F.interpolate(self.valid_mask.float(), size=((h // self.factor + 1), (w // self.factor + 1)))
                valid_mask_k = valid_mask_k.view(-1, (h // self.factor + 1) * (w // self.factor + 1), 1)
        else:
            k = k.view(-1, h * w, self.attention_dim)
#             valid_mask_k = self.valid_mask.view(-1, h * w, 1)

        matmul_qk = torch.matmul(q, k.transpose(1, 2))
        scaled_att_logits = matmul_qk / torch.sqrt(torch.tensor(self.attention_dim).float())

        if self.valid_mask is not None:
            final_mask = torch.matmul(valid_mask_q, valid_mask_k.transpose(1, 2))
            scaled_att_logits += (1 - final_mask) * -1e9
        att_weights = F.softmax(scaled_att_logits, dim=-1)

        if self.linformer:
            value = resize_fn(score, size=((h // self.factor + 1), (w // self.factor + 1)), mode='bilinear', align_corners=True)
            value = value.view(-1, (h // self.factor + 1) * (w // self.factor + 1), value_dim)
        else:
#             value = score.view(-1, h * w, value_dim)
            value = score.reshape(-1, h * w, value_dim)
            
        att_score = torch.matmul(att_weights, value)
        att_score = att_score.view(logits.shape)

        att_score += score
        if value_dim != self.num_class:
            bg = 2 - torch.max(att_score, dim=1, keepdim=True)[0]
            att_score = torch.cat([bg, att_score], dim=1)
            
        out_att_logits = self.bn1(self.conv2(att_score))
        return out_att_logits


class GANet_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=2, layer=3, pos_injection=2, is_encoding=1,
                pos_rfactor=2, pooling='max', dropout_prob=0.1, pos_noise=0.0):
        super(GANet_Conv, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)
        self.attention_first = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=mid_1_channel, kernel_size=1, stride=1, padding=0, bias=False),
                Norm2d(mid_1_channel),
                nn.ReLU(inplace=True))    

        if layer == 2:
            self.attention_second = nn.Sequential(
                    nn.Conv2d(in_channels=mid_1_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                    nn.Conv2d(in_channels=mid_1_channel, out_channels=mid_2_channel, kernel_size=3, stride=1, padding=1, bias=True),
                    Norm2d(mid_2_channel),
                    nn.ReLU(inplace=True))    
            
            self.attention_third = nn.Sequential(
                    nn.Conv2d(in_channels=mid_2_channel, out_channels=out_channel,kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))

        if self.pooling == 'mean':
            #print("##### average pooling")
            self.rowpool = nn.AdaptiveAvgPool2d((128//pos_rfactor,128//pos_rfactor))
        else:
            #print("##### max pooling")
            self.rowpool = nn.AdaptiveMaxPool2d((128//pos_rfactor,128//pos_rfactor))
            
    def forward(self, source_x, target_x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        H = source_x.size(3)
        W = source_x.size(2)
        
        x = torch.cat((source_x, target_x), dim=1)
        
        x1d = self.rowpool(x)
        
        if self.dropout_prob > 0:
            x1d = self.dropout(x1d)

        x1d = self.attention_first(x1d)
        x1d = self.attention_second(x1d)

        x1d = self.attention_third(x1d)       
        x1d = self.sigmoid(x1d)  
        x1d = F.interpolate(x1d, size=(W,H), mode='bilinear')
        
        return x1d

    
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
    

def default(val, def_val):
    return def_val if val is None else val    
    
    
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

        
def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
    
    
def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
#         MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

    
class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] # 32, 64, 128, 256, 512

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])
        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])
        
        self.convcm = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=4, kernel_size=2, stride=2),)  
          
        self.fc = nn.Sequential( 
                  nn.Linear(4096, 512),
                  nn.ReLU(),
                  nn.Linear(512, 2),
                  )
        
        self.classification_head = nn.Sequential(
            self.convcm,
            nn.AvgPool2d(2),
            nn.Flatten(1),
            self.fc
        )
        
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        
#         self.stda = GANet_Conv(in_channel=129, out_channel=1)
        self.self_attn = Self_Attn(num_class=2, attention_dim=128, is_training=True, linformer=False, valid_mask=None)
 
        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)
   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, xA, xB):
        
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        return (out, )
        
    def forward_with_class(self, xA, xB, gcam=None):
        
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))
        
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        
        out_copy = out

        class_output = self.classification_head(out)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))

        out = self.conv_final(out)
 
        out_copy = [out_copy]
        if gcam != None:
            score_map = self.self_attn(out_copy, gcam)
            score_map = F.interpolate(
            score_map, xA.shape[2:], mode="bilinear", align_corners=False)
            return (out, class_output, score_map)

        return (out, class_output)