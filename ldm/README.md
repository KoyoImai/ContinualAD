# Latent Diffusionの主要プログラムを格納

## 1. プログラムの概要
ldmディレクトリに配置されたプログラムの全体像は以下の通りです．

```
ldm
├── data      : 
├── models    : 
    ├── diffusion  :
        ├── dpm_solver        :
        ├── __init__.py       :
        ├── ddim.py           :
        ├── ddpm.py           : DDPMとLatentDiffusionの実装
        ├── plms.py           : 
        └── sampling_util.py  :
    └── autoencoder.py : 
├── modules   : 
└── utils.py  : 
```