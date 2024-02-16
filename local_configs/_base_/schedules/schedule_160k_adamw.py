optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=30000)  

checkpoint_config = dict(by_epoch=False, interval=30000)    
evaluation = dict(interval=3000, metric='mIoU')
