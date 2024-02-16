_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/acdc_1024x1024_repeat_origin.py',  # NOTE Cotta scenario
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]


ITERATION = 30000
workflow = [('train', 1)]  

BATCH_SIZE = 1                        
EXP_NUMS = 4                  # NOTE Expert numbers        
num_step1_domain = 4          # NOTE use it at the decoder part
w_domain_pred = True          # NOTE control forward function when we use pseudo domain label
TTA_SEGLOSS_weight = 1.0
warmup_domains = {'train' : 0, 'rain_syn' : 1, 'night_syn' : 2, 'fog_rendering' : 3}

checkpoint_config = dict(by_epoch=False, interval=30000) 
data = dict(samples_per_gpu=BATCH_SIZE)     
evaluation = dict(interval=1, metric='mIoU')   

TTA_SEGLOSS_weight = 1.0      

norm_cfg = dict(type='BN', requires_grad=True)       
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/d1/daeun/acdc-submission/pretrained/mit_b5.pth',
    
    backbone=dict(
        # type = 'mit_b5_EveryMOEadapter_wDomain',       # NOTE Every MoDE 
        type = 'mit_b5_LastMOEadapter_wDomain',          # NOTE Last MoDE 

        style='pytorch'),
    
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='Soft_CrossEntropyLoss', use_sigmoid=False, loss_weight=TTA_SEGLOSS_weight)),
    
    auxiliary_head=dict(
        type='DomainDiscriminator',
        n_outputs = num_step1_domain,
        in_channels=3,    
        channels=16,      
        num_classes = num_step1_domain,     
        loss_decode = dict(type='CrossEntropyLoss', is_domain_loss=True, loss_weight=0.1)    
    ),
    
    train_cfg=dict(mode=None, 
                   num_step1_domain = num_step1_domain, 
                   domains = warmup_domains, 
                   expert_num = EXP_NUMS,     
                    ),
    
    test_cfg=dict(mode='slide', crop_size=(1024,1024), stride=(768,768), )
    )


optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 })) 

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


