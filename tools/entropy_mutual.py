import wandb
import argparse
import os
import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import * 
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed
from copy import deepcopy

def to_MB(a):
    return a/1024.0/1024.0

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    
    parser.add_argument('--show', action='store_true', help='show results')
    
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    
    parser.add_argument(
        '--tmpdir', default = '/d1/daeun/acdc-submission/work_dirs/tmp/', 
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
        
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True       

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    datasets = [
                build_dataset(cfg.data.test),  
                build_dataset(cfg.data.test1), 
                build_dataset(cfg.data.test2),  
                build_dataset(cfg.data.test3), 
                ]

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))   
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')   
    
    print('*********** Load model from .. ', args.checkpoint, '***********')

    cfg['meta'] = checkpoint['meta']        
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False 
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    # Multi GPU
    model = MMDataParallel(model.cuda(), device_ids=[0])
    source_model = deepcopy(model) 
    
    is_weather_restore = False                        
    is_round_restore = False
    
    try : 
        ratio_len = len(cfg.data.test.pipeline[1].img_ratios)
    except : 
        ratio_len = 0 
        
    for i in range(10):
        print("====== Round ", i,  " ====== ")

        data_loaders = [build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False) for dataset in datasets]

        for dataset, data_loader in zip(datasets, data_loaders):

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   
            starter.record()
        
            outputs= Entropy_based(model, 
                                data_loader, 
                                args.show, 
                                args.show_dir,     
                                efficient_test=False,
                                source_model = source_model,  
                                w_domain_pred = cfg.w_domain_pred,
                                is_viz = True, 
                                ) 
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            print('=' * 50)
            print('1 round time : ', curr_time)
            print(f"After model to device: {to_MB(torch.cuda.memory_allocated()):.2f}MB")

            rank, _ = get_dist_info()
            
            if rank == 0:
                kwargs = {} if args.eval_options is None else args.eval_options
                
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                
                if args.eval:
                    dataset.evaluate(outputs, args.eval, **kwargs)     
        del data_loaders

    
if __name__ == '__main__':
    main()
