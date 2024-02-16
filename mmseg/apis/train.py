import random
import warnings
from tqdm import tqdm 
import wandb 
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner, load_checkpoint
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger

def CHECK_NUM_PARAMS(model) : 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params 

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor_DEBUG(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):

    # logger = get_root_logger(cfg.log_level)
    checkpoint = load_checkpoint(model, '/d1/daeun/acdc-submission/segformer.b2.1024x1024.city.160k.pth', map_location='cpu')   
    
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(     # NOTE batch size is 1 .. 
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    # if cfg.get('runner') is None:
    #     cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
    #     warnings.warn(
    #         'config is now expected to have a `runner` section, '
    #         'please set `runner` in your config.', UserWarning)
        
    
    # NOTE(7/4) Only update adapters 
    if not cfg.train_scratch :  
        
        for param in model.parameters():
            param.requires_grad = False
            
        # NOTE Added if u want to activate during the warmup training 
        for name, param in model.named_parameters():
            if "adapter" in name or 'auxiliary' in name or 'f_gate' in name or 'Meta' in name or 'experts' in name : \
                param.requires_grad = True
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        
    # Model param counts + logging 
    print('=' * 30)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) 
    print(params, ' number of parameters are updated! ')
    print('=' * 30)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00006/100, betas=(0.9, 0.999)) 
    
    for i, data in tqdm(enumerate(data_loaders[0])):     
        loss, cur_MutualMat = model.forward(return_loss=True, 
                        w_domain_pred=True ,  
                        warmup=True,                         
                        **data)   
        seg_loss = torch.mean(loss["decode.loss_seg"])        
        print(loss["decode.loss_seg"])
        
        wandb.log({"loss_seg": torch.mean(loss["decode.loss_seg"])})
        seg_loss.backward()              # 이거 했었어야했을 것 같은데 
    
def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    
    # dataloader -> 하나씩 병렬적으로 들어감. 
    data_loaders = [
        build_dataloader(     # NOTE batch size is 1 .. 
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # # build runner
    # optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
        
    
    # NOTE(7/4) Only update adapters 
    if not cfg.train_scratch :  
        
        for param in model.parameters():
            param.requires_grad = False
            
        # NOTE Added if u want to activate during the warmup training 
        for name, param in model.named_parameters():
            if "adapter" in name or 'auxiliary' in name or 'f_gate' in name or 'Meta' in name or 'experts' in name : \
                param.requires_grad = True
            
            # if param.requires_grad : 
            #     print('Require grad : ', name)
                    
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    

    ''' TTA parameter estimation '''
    adapter_params_count = 0
    for name, param in model.named_parameters():
        if 'adapter' in name or 'f_gate' in name :
            adapter_params_count += param.numel()

    print('TTA Parameters!! : ', adapter_params_count)

    # Model param counts + logging 
    print('=' * 30)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters]) 
    print(CHECK_NUM_PARAMS(model), ' number of parameters are updated at warmup! ')
    print('=' * 30)

    # build runner -> Modify NOTE(10/31)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,        
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    
    # register hooks
    runner.register_training_hooks(cfg.lr_config, 
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config, 
                                   cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
        
    # register eval hooks -> NOTE(9/4) We are using just clear cityscapes dataset!
    if validate:
        
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
            
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))   # eval hook register 

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
        
    # load from 
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    runner.run(data_loaders, 
               cfg.workflow,
               )     