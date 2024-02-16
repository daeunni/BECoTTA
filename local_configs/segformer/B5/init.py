_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/acdc_1024x1024_repeat.py',       
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]


ITERATION = 40000
warmup_domains = {'train' : 0, 'rain_syn' : 1, 'night_syn' : 2, 'fog_rendering' : 3}  # NOTE source style-transfered domains!! -> for domain-wise warmup
workflow = [('train', 1), ('train', 1), ('train', 1), ('train', 1)]  
checkpoint_config = dict(by_epoch=False, interval=10000)    # 30000
runner = dict(type='IterBasedRunner', max_iters=ITERATION)      

BATCH_SIZE = 1                
one_valid = True             
EXP_NUMS = 4                 
train_scratch = False        
num_step1_domain = 4         
w_domain_pred = True         

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/d1/daeun/acdc-submission/pretrained/mit_b5.pth',
    
    backbone=dict(
        type = 'mit_b5_EveryMOEadapter_wDomain', 
        # type = 'mit_b5_LastMOEadapter_wDomain', 
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
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),    

    auxiliary_head=dict(
        type='DomainDiscriminator',
        n_outputs = num_step1_domain,
        in_channels=3,    # custom 
        channels=16,      # custom 
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


data = dict(samples_per_gpu=BATCH_SIZE)      
evaluation = dict(interval=50000, metric='mIoU')  

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,     # 0.00006
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 })) 

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


