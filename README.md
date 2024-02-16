# BECoTTA: Input-dependent Online Blending of Experts for Continual Test-time Adaptation
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://becotta-ctta.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2402.08712-b31b1b.svg)](https://arxiv.org/pdf/2402.08712.pdf)   

- Authors: [Daeun Lee*](https://daeunni.github.io/),  [Jaehong Yoon*](https://jaehong31.github.io/),  [Sung Ju Hwang](http://www.sungjuhwang.com/) (* denotes equal contribution)    
> More details will be updated soon! 

<img width="928" alt="image" src="https://github.com/daeunni/BECoTTA/assets/62705839/511417bf-bb8b-46c2-81ec-7ab9c208b92e">

## Abstract 
Continual Test Time Adaptation (CTTA) is required to adapt efficiently to continuous unseen domains while retaining previously learned knowledge. However, despite the progress of
CTTA, forgetting-adaptation trade-offs and efficiency are still unexplored. Moreover, current CTTA scenarios assume only the disjoint situation, even though real-world domains are seamlessly changed. 
To tackle these challenges, this paper proposes BECoTTA, an input-dependent yet efficient framework for CTTA. We propose Mixture-of-Domain Low-rank Experts (MoDE) that contains two core components: i) Domain-Adaptive Routing, which aids in selectively capturing the domain-adaptive knowledge with multiple domain routers, and (ii) Domain-Expert Synergy Loss to maximize the dependency between each domain and expert. We validate our method outperforms multiple CTTA scenarios including disjoint and gradual domain shits, while only requiring ∼98% fewer trainable parameters. We also provide analyses of our method, including the construction of experts, the effect of domain-adaptive experts, and visualizations. 

## CTTA (Continual Test-time Adaptation) 
```
bash becotta.sh
```


## Setup 
### [1] Environment
- We follow [mmsegmentation code base](https://drive.qin.ee/api/raw/?path=/cv/cvpr2022/acdc-seg.tar.gz) provided by [CoTTA](https://github.com/qinenergy/cotta) authors. 


### [2] Dataset
- You can download **ACDC dataset** from [here](https://acdc.vision.ee.ethz.ch/download).

### [3] Our warm-up model 
- We provide our trained initialized model checkpoints. 

