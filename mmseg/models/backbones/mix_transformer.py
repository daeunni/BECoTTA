# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math
import numpy as np 
import pickle 
from mmseg.models.prev_backbone import *     # NOTE Ingredients 

class SimpleAdapter(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features           
        hidden_features = hidden_features or in_features     
        self.fc1 = nn.Linear(in_features, hidden_features)       # Downconv
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)      # UPconv
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)            
        x = self.act(x)            
        x = self.drop(x)
        x = self.fc2(x)           
        x = self.drop(x)   
        return x
    

class MOEAdapterBlock(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, 
                 expert_num=None, select_mode=None, MoE_hidden_dim=None, num_k = None):   
        
        super().__init__()
        
        self.expert_num = expert_num
        self.domain_num = 4
        self.select_mode = select_mode
        self.acc_freq = 0 
        self.num_K = num_k
        self.MI_task_gate = torch.zeros(self.domain_num, self.expert_num).cuda()    

        self.norm1 = norm_layer(dim) 
        self.norm2 = norm_layer(dim)
        
        self.attn = Attention(
            dim,
            num_heads=num_heads,     
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop, 
            sr_ratio=sr_ratio)
        
        mlp_hidden_dim = int(dim * mlp_ratio)      
        
        self.mlp = Mlp(in_features=dim,    
                       hidden_features=mlp_hidden_dim,   
                       act_layer=act_layer, 
                       drop=drop)
        
        if self.select_mode == 'new_topk':      # NOTE you can add other routing methods 
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.f_gate = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * expert_num, bias=False)) for i in range(self.domain_num)])   
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()   
        self.apply(self._init_weights)    
        
        expert_lists = []
        for _ in range(expert_num) : 
            tmp_adapter = SimpleAdapter(in_features=dim, hidden_features=MoE_hidden_dim, act_layer=act_layer, drop=drop)  
            tmp_adapter.apply(self._init_weights)      
            expert_lists.append(tmp_adapter)

        self.adapter_experts = nn.ModuleList(expert_lists)   

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)   
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def minmax_scaling(self, top_k_logits) : 
            m1 = top_k_logits.min() ; m2 = top_k_logits.max()
            return (top_k_logits-m1)/ (m2-m1)
            
    def one_hot_encoding(self, index, num_classes):
        one_hot = np.zeros(num_classes)  
        one_hot[index] = 1  
        return one_hot

    def forward(self, x, H, W, expert_num, select_mode, pseudo_domain_label=None):    
        try : 
            tot_x = torch.zeros_like(x)   
        except : 
            tot_x = None
            
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   
        
        if self.select_mode == 'random' :    
            select = torch.randint(low=0, high=expert_num, size=(1,)).item()  
            MI_loss= self.one_hot_encoding(select, self.expert_num)      # empty    
            x = x + self.adapter_experts[select](x, H, W)    
            
        elif self.select_mode == 'new_topk' : 
            task_bh = pseudo_domain_label 
            total_w = self.f_gate[task_bh](x)    
            clean_logits, raw_noise_stddev = total_w.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            eps = torch.randn_like(clean_logits)
            logits = clean_logits + eps * noise_stddev   
            exp_wise_sum = logits.sum(dim=1)    

            # select topk
            top_logits, top_indices = exp_wise_sum.topk(min(self.num_K + 1, self.expert_num), dim=1)
            top_k_logits = top_logits[:, :self.num_K]      # [batch, k]
            top_k_indices = top_indices[:, :self.num_K]    # [batch, k] -> selected experts
            
            if len(top_k_logits) > 1 : 
                top_k_gates = self.softmax(self.minmax_scaling(top_k_logits))       # [batch, k] -> probabilities
            else : 
                top_k_gates = self.softmax(top_k_logits)       
            assert top_k_indices.shape == top_k_gates.shape

            # adapter weighted output
            for idx in range(len(top_k_indices[0])) : 
                tot_x += top_k_gates[0][idx].item() * self.adapter_experts[idx](x, H, W)   
            
            x = x + tot_x
            MI_loss = (exp_wise_sum / (H * W)).detach().cpu().numpy()
    
        else : 
            select = None 
            print('No attribute')
         
        return x, MI_loss   
    

class MOE_MixVisionTransformer_EveryAdapter_wDomain(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], expert_num=16, select_mode=None, hidden_dims=None, 
                 num_k=None):
        
        super().__init__()

        self.hidden_dims = hidden_dims
        print(self.hidden_dims)
        self.select_mode = select_mode
        print(self.select_mode)
        self.expert_num = expert_num
        self.num_classes = num_classes
        self.depths = depths
        self.flatten = nn.Flatten()
        self.num_k = num_k
        
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
    
        
        """ Stage 1 : H/4 x W/4 x C (High resolution) """
        self.block1 = nn.ModuleList([
                    MOEAdapterBlock(
                        dim=embed_dims[0], 
                        num_heads=num_heads[0], 
                        mlp_ratio=mlp_ratios[0], 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=dpr[cur + i], 
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[0], expert_num=expert_num, 
                        select_mode=self.select_mode, 
                        MoE_hidden_dim = self.hidden_dims[0], 
                        num_k = self.num_k)  
                        for i in range(depths[0])     
                    ]   
                    )    # depth = [3, 6, 40, 3] = # of encoder layer in stage i 
        
        self.norm1 = norm_layer(embed_dims[0])    
        cur += depths[0]
        
        """ Stage 2 : H/8 x W/8 x C  """
        self.block2 = nn.ModuleList(
                [
                MOEAdapterBlock(
                   dim=embed_dims[1], 
                   num_heads=num_heads[1], 
                   mlp_ratio=mlp_ratios[1], 
                   qkv_bias=qkv_bias, 
                   qk_scale=qk_scale,
                   drop=drop_rate, 
                   attn_drop=attn_drop_rate, 
                   drop_path=dpr[cur + i], 
                   norm_layer=norm_layer,
                   sr_ratio=sr_ratios[1], expert_num=expert_num, 
                   select_mode=self.select_mode, 
                   MoE_hidden_dim = self.hidden_dims[1], 
                   num_k = self.num_k)
                  for i in range(depths[1])
            ]
            )
        self.norm2 = norm_layer(embed_dims[1])     
        cur += depths[1]
        
        
        """ Stage 3 : H/16 x W/16 x C  """
        self.block3 = nn.ModuleList(
            [
            MOEAdapterBlock(
            dim=embed_dims[2], 
            num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, 
            drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], expert_num=expert_num, 
            select_mode=self.select_mode, 
            MoE_hidden_dim = self.hidden_dims[2], 
            num_k = self.num_k)
            for i in range(depths[2])
            ])
        
        self.norm3 = norm_layer(embed_dims[2])  
        cur += depths[2]
        
        """ Stage 4 : H/32 x W/32 x C  """
        self.block4 = nn.ModuleList(
            [
            MOEAdapterBlock(
            dim=embed_dims[3], num_heads=num_heads[3], 
            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
            norm_layer=norm_layer,sr_ratio=sr_ratios[3], 
            expert_num=expert_num, 
            select_mode=self.select_mode, 
            MoE_hidden_dim = self.hidden_dims[3], 
            num_k = self.num_k)
            for i in range(depths[3])
            ])
        
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # freeze
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, pseudo_domain_label):      
        
        outs = [] ; prev = [] ; after = []
        B = x.shape[0]
        total_MI_loss = torch.zeros((B, self.expert_num))
    
        # == stage 1 == #
        x, H, W = self.patch_embed1(x) 
        for i, blk in enumerate(self.block1):
            x, MI_loss = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)    
            total_MI_loss += MI_loss 
        x = self.norm1(x) 
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   
        outs.append(x)

        # == stage 2 == #
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x, MI_loss = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)   
            total_MI_loss += MI_loss 
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # == stage 3 == #
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x, MI_loss = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)   
            total_MI_loss += MI_loss 
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # == stage 4 == #
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x, MI_loss = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
            total_MI_loss += MI_loss 
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs, total_MI_loss
    
    def forward(self, x, pseudo_domain_label):    
        x, total_MI_loss = self.forward_features(x, pseudo_domain_label)
        return x, total_MI_loss      
        

class MOE_MixVisionTransformer_LastAdapter_wDomain(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], expert_num=16, select_mode=None, hidden_dims=None, num_k=None):
        
        super().__init__()
        
        self.hidden_dims = hidden_dims
        print(self.hidden_dims)
        
        self.select_mode = select_mode
        self.expert_num = expert_num
        self.num_classes = num_classes
        self.depths = depths
        self.flatten = nn.Flatten()
        self.num_k = num_k
        
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  
        cur = 0
    
        """ Stage 1 : H/4 x W/4 x C (High resolution) """
        self.block1 = nn.ModuleList([
                    Block(
                        dim=embed_dims[0], 
                        num_heads=num_heads[0], 
                        mlp_ratio=mlp_ratios[0], 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=dpr[cur + i], 
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[0], 
                        )  
                        for i in range(depths[0])     
                    ]   
                    )    
        self.norm1 = norm_layer(embed_dims[0])    
        cur += depths[0]
        
        """ Stage 2 : H/8 x W/8 x C  """
        self.block2 = nn.ModuleList(
                [
                Block(
                   dim=embed_dims[1], 
                   num_heads=num_heads[1], 
                   mlp_ratio=mlp_ratios[1], 
                   qkv_bias=qkv_bias, 
                   qk_scale=qk_scale,
                   drop=drop_rate, 
                   attn_drop=attn_drop_rate, 
                   drop_path=dpr[cur + i], 
                   norm_layer=norm_layer,
                   sr_ratio=sr_ratios[1]) 
                  for i in range(depths[1])
            ]
            )
        self.norm2 = norm_layer(embed_dims[1])     
        cur += depths[1]
        
        """ Stage 3 : H/16 x W/16 x C  """
        self.block3 = nn.ModuleList(
            [
            Block(
            dim=embed_dims[2], 
            num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, 
            drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])
            ])
        self.norm3 = norm_layer(embed_dims[2])  
        cur += depths[2]
        
        """ Stage 4 : H/32 x W/32 x C  """
        self.block4 = nn.ModuleList(
            [
            MOEAdapterBlock(
            dim=embed_dims[3], num_heads=num_heads[3], 
            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
            norm_layer=norm_layer,sr_ratio=sr_ratios[3], 
            expert_num=expert_num, 
            select_mode=self.select_mode, 
            MoE_hidden_dim = self.hidden_dims[3], 
            num_k = self.num_k)  
            for i in range(depths[3])
            ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # freeze
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, pseudo_domain_label):     
        
        B = x.shape[0]
        outs = []
        total_MI_loss = torch.zeros((B, self.expert_num))
    
        # == stage 1 == #
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)    
        x = self.norm1(x)        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   
        outs.append(x)

        # == stage 2 == #
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)   
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # == stage 3 == #
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)     
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # == stage 4 == #
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x, MI_loss = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
            total_MI_loss += MI_loss 
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs, total_MI_loss
    
    def forward(self, x, pseudo_domain_label):    
        x, total_MI_loss = self.forward_features(x, pseudo_domain_label)
        return x, total_MI_loss      
        

""" B5 """
@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        
@BACKBONES.register_module()
class mit_b5_EveryMOEadapter_wDomain(MOE_MixVisionTransformer_EveryAdapter_wDomain):
    def __init__(self, **kwargs):
        super(mit_b5_EveryMOEadapter_wDomain, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 40, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, 
            expert_num=6, 
            select_mode='new_topk', 
            hidden_dims = [2, 4, 10, 16],        # NOTE you can set them freely 
            # hidden_dims = [16, 32, 60, 80], 
            # hidden_dims = [8, 8, 16, 32] 
            num_k = 3 
            )   

@BACKBONES.register_module()
class mit_b5_LastMOEadapter_wDomain(MOE_MixVisionTransformer_LastAdapter_wDomain):
    def __init__(self, **kwargs):
        super(mit_b5_LastMOEadapter_wDomain, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, 
            expert_num=4, 
            select_mode='new_topk', 
            hidden_dims = [2, 2, 4, 6], 
            num_k = 3 
            )    
