# CDADの主要プログラムを格納

## 1. プログラムの概要
cdmディレクトリに配置されたプログラムの全体像は以下の通りです．

```
cdm
├── amn.py          : 
├── ddim_hacked.py  : Latent Diffusion Model関連のプログラムを格納したディレクトリ
├── func.py         : 
├── gpm.py          : CDADの本体とgradient projection memory (gpm) 関連の処理を実装
├── hack.py         : 
├── logger.py       : 
├── mha.py          : multi head attention (mha) 関連の処理を実装
├── model.py        : model作成の大元の関数を実装
├── param.py        : 
├── safe_open.py    : 
├── sd_amn.py       : SD_AMNを実装．CDADはSD_AMNを継承．SDの実装を拡張している．
└── share.py        : 
```

### 損失の計算
CDADの本体は`gpm.py`で実装されています．
拡散モデルの通常の損失は`../ldm/models/diffusion/ddpm.py`で計算されます．
その後，`sd_amn.py`でCDADのAnomaly Masked Network (AMN) 損失を計算します．

### 最適化
パラメータの最適化は`sd_amn.py`の`SD_AMN.training_step()`で行われます．
ベースタスクは通常の最適化を行い，task id > 0の場合はGradient Projection memory (GPM) によって勾配直行射影を適用します．


