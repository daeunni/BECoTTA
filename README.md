# BECoTTA: Input-dependent Online Blending of Experts for Continual Test-time Adaptation
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://becotta-ctta.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2402.08712-b31b1b.svg)](https://arxiv.org/pdf/2402.08712.pdf)   

- Authors: [Daeun Lee*](https://daeunni.github.io/),  [Jaehong Yoon*](https://jaehong31.github.io/),  [Sung Ju Hwang](http://www.sungjuhwang.com/) (* denotes equal contribution)    


<img width="928" alt="image" src="https://github.com/daeunni/BECoTTA/assets/62705839/511417bf-bb8b-46c2-81ec-7ab9c208b92e">

## Abstract 
Continual Test Time Adaptation (CTTA) is required to adapt efficiently to continuous unseen domains while retaining previously learned knowledge. However, despite the progress of
CTTA, forgetting-adaptation trade-offs and efficiency are still unexplored. Moreover, current CTTA scenarios assume only the disjoint situation, even though real-world domains are seamlessly changed. 
To tackle these challenges, this paper proposes BECoTTA, an input-dependent yet efficient framework for CTTA. We propose Mixture-of-Domain Low-rank Experts (MoDE) that contains two core components: i) Domain-Adaptive Routing, which aids in selectively capturing the domain-adaptive knowledge with multiple domain routers, and (ii) Domain-Expert Synergy Loss to maximize the dependency between each domain and expert. We validate our method outperforms multiple CTTA scenarios including disjoint and gradual domain shits, while only requiring ∼98% fewer trainable parameters. We also provide analyses of our method, including the construction of experts, the effect of domain-adaptive experts, and visualizations. 

## 🚗 Main process of CTTA (Continual Test-time Adaptation) 
- You can set our main config file. `becotta/local_configs/segformer/B5/tta.py`
- You can find our initialized model [here](https://drive.google.com/drive/folders/1e1ZIyYVlZL4OS67K1vD6TmFvyFlCsBxA?usp=sharing). 
```
# CTTA process 
bash ./tools/becotta.sh
```


## 🖥️ Setup 
### [1] Environment
- We follow [mmsegmentation code base](https://drive.qin.ee/api/raw/?path=/cv/cvpr2022/acdc-seg.tar.gz) provided by [CoTTA](https://github.com/qinenergy/cotta) authors.

  - You can refer to [this issue](https://github.com/qinenergy/cotta/issues/13) related to environment setup. 
  - 📣 Note that our source model (Segformer) mainly uses *pretty low mmcv version*. (`mmcv==1.2.0`)
  
**1. You can create conda environment using `.yaml` file we provided.** 
```shell
conda env update --name cotta --file environment.yml
conda activate cotta
```

**2. You can install mmcv by yourself.**
- Our code is tested `torch 1.7.0 + cuda 11.0 + mmcv 1.2.0`
```shell
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 https://download.pytorch.org/whl/torch_stable.html
```
- Install lower version of mmcv refer to [this issue](https://github.com/open-mmlab/mmcv/issues/1386#issuecomment-933577744).
```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```


### [2] Dataset
- You can download the target domain **ACDC dataset** from [here](https://acdc.vision.ee.ethz.ch/download).
  - Setup `Fog -> Night -> Rain -> Snow` scenario using train dataset.
  - You need to change `becotta/local_configs/_base_/datasets/acdc_1024x1024_repeat_origin.py` to your own path.
    
    ```shell
    # dataset settings
    dataset_type = 'ACDCDataset'
    data_root = 'your data path'   
    ``` 
- We also provide a bunch of config files of driving datasets at `becotta/mmseg/datasets`! However, note that you should match the segmentation label format with Cityscapes style. You can freely use these data configs and design your own scenario.
  
  - **BDD100k**: `bdd.py`
  - **Kitti Seg**: `kitti.py`
  - **Foggy Driving:** `fog.py`
  - **GTAV & Synthetia**: `gtav_syn.py`
  - **Dark Zurich**: `dark.py`


### [3] Pre-trained model 
- We mainly adopt pre-trained Segformer with Cityscapes dataset.
  - You can `segformer.b5.1024x1024.city.160k.pth` [here](https://drive.google.com/drive/folders/1e1ZIyYVlZL4OS67K1vD6TmFvyFlCsBxA?usp=sharing). 
  - Also, you can find `mit_b5.pth` backbone [here](https://drive.google.com/drive/folders/1e1ZIyYVlZL4OS67K1vD6TmFvyFlCsBxA?usp=sharing). Please located them at `./pretrained/` directory. 
  

## 📁 Note 
### [1] Checkpoint of our model 
- We provide our trained initialized model checkpoints [here](https://drive.google.com/drive/folders/1e1ZIyYVlZL4OS67K1vD6TmFvyFlCsBxA?usp=sharing). 
  - If you need more experiments, feel free to email `goodgpt@korea.ac.kr`. :-) 

### [2] Flexibility of BECoTTA 
- As we mentioned in our paper, you can freely change the rank of experts, number of experts, and selected number of experts ($K$).
- e.g. You can modify it as follows.

  ```python
  class mit_b5_EveryMOEadapter_wDomain(MOE_MixVisionTransformer_EveryAdapter_wDomain):
      def __init__(self, **kwargs):
          super(mit_b5_EveryMOEadapter_wDomain, self).__init__(
              patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], 
              mlp_ratios=[4, 4, 4, 4],
              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
              depths=[3, 6, 40, 3], 
              sr_ratios=[8, 4, 2, 1],
              drop_rate=0.0, drop_path_rate=0.1, 
              expert_num=6,                        # Modify here 
              select_mode='new_topk', 
              hidden_dims = [2, 4, 10, 16],        # Modify here 
              num_k = 3                            # Modify here 
              )   
  ```
- Our model utilized these parameters as follows. Please refer to our main paper for more details. 
  
  |           | Exp | K |       Rank       |  MoDE |
  |-----------|:---:|:-:|:----------------:|:-----:|
  | BECoTTA-S |  4  | 3 |   [0, 0, 0, 6]   |  Last |
  | BECoTTA-M |  6  | 3 |  [2, 4, 10, 16]  | Every |
  | BECoTTA-L |  6  | 3 | [16, 32, 60, 80] | Every |


### [3] TODO 
- [ ] Construction process of Continual Gradual Shifts (CGS) scenario will be updated. 
- [ ] Warmup initializing process will be updated. 
- [x] Whole process of CTTA was added.


## Reference 
```
@inproceedings{Lee2024BECoTTAIO,
  title={BECoTTA: Input-dependent Online Blending of Experts for Continual Test-time Adaptation},
  author={Daeun Lee and Jaehong Yoon and Sung Ju Hwang},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267657972}
}
```
