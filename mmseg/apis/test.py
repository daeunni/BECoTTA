import os.path as osp
import wandb 
import pickle
import shutil
import tempfile
import datetime
import math 
import mmcv
import time
import numpy as np
from copy import deepcopy
import torch
import torch.distributed as dist

from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.runner import build_optimizer, build_runner
from mmseg import __version__
from mmseg.core import DistEvalHook, EvalHook
from mmseg.utils import get_root_logger, collect_env
from mmseg.ops import resize

from IPython import embed
from tqdm import tqdm 
import random
import torch.nn as nn
import pdb

def CHECK_NUM_PARAMS(model) : 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params 

def Entropy_based(model, 
                  data_loader, 
                  show=False, 
                  out_dir=None, 
                  efficient_test=False, 
                  source_model=None, 
                  w_domain_pred=None, 
                  is_viz=False, 
                ):    
    model.eval()       
    source_model.eval()
    dataset = data_loader.dataset
    source_model_dict = source_model.state_dict()
    results = [] 
    
    """ Select updated params(adapter) """
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "adapter" in name or 'f_gate' in name : 
            param.requires_grad = True   
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00006/100, betas=(0.9, 0.999))     
    print('* TRAINABLE PARAMS_COUNT : ', CHECK_NUM_PARAMS(model)) ; print('-' * 50)
    
    """ Auxilary model select """
    selected_auxil = model.module.auxiliary_head           
    selected = torch.nn.Sequential(selected_auxil)
    selected.eval().to('cuda')  
                
    """ Weather-per dataloader """    
    for i, data in tqdm(enumerate(data_loader)):     
        with torch.no_grad():        
            Aug_lens = len(data['img'])
            if Aug_lens > 1 : 
                ORI = 2   
            else : 
                ORI = 0 

            """ Domain prediction using auxil head """
            ORI_Pseudo_Domain = selected(data['img'][ORI].cuda()).argmax().item()   
            for j in range(Aug_lens) :  
                data['img_metas'][j].data[0][0]['domain_prefix'] = ORI_Pseudo_Domain
            
            """ Pseudo label generation """
            model.eval()          
            result, probs, preds = model(return_loss=False,              
                            w_domain_pred=w_domain_pred,          
                            **data)
            mask = (probs[ORI][0] > 0.69).astype(np.int64)   # 0.69  : reuse cotta's threshold 
               
            ori_result, ori_probs, ori_preds = source_model(return_loss=False,                 
                                                    w_domain_pred=w_domain_pred,          
                                                    **data)
            y_0 = ori_probs[ORI][0]      
            y_cur = probs[ORI][0]
            mask_filtering = (y_cur - y_0 >= 0).astype(np.int64)           
            mask = np.where((mask == 1) & (mask_filtering == 1), 1, 0)     
            result = [ (mask * preds[ORI][0] + (1.- mask) * result[0]).astype(np.int64) ]     
                
            # result = [ (mask * preds[ORI][0] + (1.- mask) * result[0]).astype(np.int64) ]    
                
        if isinstance(result, list): 
            PSEUDO_LABEL = torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0)    
            loss, cur_MutualMat = model.forward(return_loss=True, 
                                    w_domain_pred=w_domain_pred,          
                                    warmup=False,                         
                                    img=data['img'][ORI],      
                                    img_metas=data['img_metas'][ORI].data[0],    
                                    gt_semantic_seg=PSEUDO_LABEL)   
            results.extend(result)

        else : 
            loss = model(return_loss=True, 
                         w_domain_pred=w_domain_pred,
                         warmup=False,       
                         img=data['img'][0], 
                         img_metas=data['img_metas'][0].data[0], 
                         gt_semantic_seg=PSEUDO_LABEL)
            results.append(result)
                                    
        seg_loss = torch.mean(loss["decode.loss_seg"])        
                    
        seg_loss.backward()        
        optimizer.step()
        optimizer.zero_grad()
        
        for nm, m  in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().cuda()    
                    with torch.no_grad():
                        p.data = source_model_dict[f"{nm}.{npp}"] * mask + p * (1.-mask)   
    return results
