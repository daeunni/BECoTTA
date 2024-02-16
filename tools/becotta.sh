# #!/usr/bin/env bash
TTA_CONFIG=./local_configs/segformer/B5/tta.py  

GPUS=1
TTA_PORT=${PORT:-34622}

ROOT='your directory'   
NAME='ours'         
WORKDIR=$ROOT$NAME        
SAVEDIR=$WORKDIR

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
WARMUP_CHECKPOINT=$WORKDIR/latest.pth      # warmup model (our version : becotta/ours)
TTA_LOG_PATH=$SAVEDIR/$NAME.log           

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$TTA_PORT \
    $(dirname "$0")/entropy_mutual.py $TTA_CONFIG $WARMUP_CHECKPOINT --launcher pytorch ${@:4} | tee $TTA_LOG_PATH
