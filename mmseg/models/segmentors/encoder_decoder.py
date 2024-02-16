import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import pdb
import numpy as np
import warnings;warnings.filterwarnings('ignore')

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        
        super(EncoderDecoder, self).__init__()
        
        self.backbone = builder.build_backbone(backbone)     
        self.mutual_loss = builder.build_loss(dict( type='MutualLoss'))      
        
        if neck is not None:
            self.neck = builder.build_neck(neck)
            
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)   
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)    

        try : 
            self.Total_MutualMat = torch.zeros(self.train_cfg['num_step1_domain'], self.train_cfg['expert_num'])  
        except : 
            self.Total_MutualMat = None     
        
        assert self.with_decode_head

    def _init_MutualMat(self) : 
        try : 
            self.Total_MutualMat = torch.zeros(self.train_cfg['num_step1_domain'], self.train_cfg['expert_num'])
        except : 
            self.Total_MutualMat = None  
    
    
    # decode head 
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)     
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    # auxiliary head 
    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)     
        self.decode_head.init_weights()
        
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:  
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()


    def extract_feat(self, img, img_metas):   
        """Extract features from images."""
        x, loss_mean = self.backbone(img)       
        if self.with_neck:
            x = self.neck(x)
        return x, loss_mean

    def extract_feat_w_DomainLabel(self, img, pseudo_domain_label):   
        """Extract features from images."""
        x, total_MI_loss = self.backbone(img, pseudo_domain_label)       
        if self.with_neck:
            x = self.neck(x)
        return x, total_MI_loss   


    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        
        x, loss_mean = self.extract_feat(img, img_metas)    
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(   
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        return out, loss_mean    

    def encode_decode_w_DomainLabel(self, img, img_metas, pseudo_domain_label):
        x, total_MI_loss = self.extract_feat_w_DomainLabel(img, pseudo_domain_label)    
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(   
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        return out, total_MI_loss 
    
    
    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, Total_MutualMat):
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg, 
                                                    Total_MutualMat, 
                                                    )    
        
        losses.update(add_prefix(loss_decode, 'decode'))    
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                
                loss_aux = aux_head.forward_domain_train(x, img_metas,    
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
                
        else:
            loss_aux = self.auxiliary_head.forward_domain_train(
                x, img_metas, self.train_cfg)
        
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def update_MutualLoss(self, Total_MutualMat) : 
        losses = {}
        MI_loss = self.mutual_loss(Total_MutualMat)
        losses.update({'mutual_loss' : MI_loss.cuda()})
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)   
        return seg_logit

    def forward_train(self, img, img_metas, w_domain_pred, pseudo_domain_label, warmup, gt_semantic_seg):   
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
                
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()   
        
        """ Domain discriminator loss """
        if self.with_auxiliary_head and warmup :                
            loss_aux = self._auxiliary_head_forward_train(img, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        loss_mean = None 

        if w_domain_pred : 
            x, cur_MutualMat = self.extract_feat_w_DomainLabel(img, pseudo_domain_label)    
            if warmup : 
                self.Total_MutualMat[pseudo_domain_label, :] = cur_MutualMat
            
        else : 
            x, loss_mean = self.extract_feat(img, img_metas)     
            cur_MutualMat = loss_mean           
        
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, self.Total_MutualMat)
        losses.update(loss_decode)
        
        if warmup and pseudo_domain_label == 3 :    
            loss_mt = self.update_MutualLoss(self.Total_MutualMat)    
            losses.update(loss_mt)     
            self._init_MutualMat()        

        return losses, cur_MutualMat 

    def slide_inference(self, img, img_meta, w_domain_pred, pseudo_domain_label, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                
                if w_domain_pred : 
                    crop_seg_logit, _ = self.encode_decode_w_DomainLabel(crop_img, img_meta, pseudo_domain_label)
                else : 
                    crop_seg_logit, _ = self.encode_decode(crop_img, img_meta)
                
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
                
        assert (count_mat == 0).sum() == 0
        
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        
        preds = preds / count_mat
    
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        return preds

    def whole_inference(self, img, img_meta, w_domain_pred, pseudo_domain_label, rescale):
        
        """Inference with full image."""
        if w_domain_pred : 
            seg_logit, _, _ = self.encode_decode_w_DomainLabel(img, img_meta, pseudo_domain_label)    
        else : 
            seg_logit, _ = self.encode_decode(img, img_meta)
            
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, w_domain_pred, pseudo_domain_label, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        
        if self.test_cfg.mode == 'slide':    # NOTE we use this !! 
            seg_logit = self.slide_inference(img, img_meta, w_domain_pred, pseudo_domain_label, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, w_domain_pred, pseudo_domain_label, rescale)
        
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta,  w_domain_pred, pseudo_domain_label, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta,  w_domain_pred, pseudo_domain_label, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, w_domain_pred, pseudo_domain_label, rescale=True):
        """
        Test with augmentations.
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], w_domain_pred, pseudo_domain_label, rescale)
        #preds = [seg_logit.argmax(dim=1).cpu().numpy()]
        
        prob, pred = seg_logit.max(dim=1)
        preds = [pred.cpu().numpy()]
        probs = [prob.cpu().numpy()]
        
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], w_domain_pred, pseudo_domain_label, rescale)  # [1, 19, 720, 1280]
            seg_logit += cur_seg_logit   # seg_logit : [1, 19, 720, 1280]

            prob, pred = cur_seg_logit.max(dim=1)   
            preds.append(pred.cpu().numpy())
            probs.append(prob.cpu().numpy())
        
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)   
        seg_pred = seg_pred.cpu().numpy()
        
        # unravel batch dim
        seg_pred = list(seg_pred)    
        return seg_pred, probs, preds
