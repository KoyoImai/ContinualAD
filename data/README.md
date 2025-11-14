# データセット関連のプログラム

## 1. プログラム・ディレクトリの概要
データセット関連のプログラム・ディレクトリの全体像は以下の通りです．

```
data
├── MVTecAD                   : 通常の継続学習を行うためのjsonファイルを格納
├── MVTecAD-MU                : MU用の学習を行うためのjsonファイルver1を格納
├── MVTecAD-MU2               : MU用の学習を行うためのjsonファイルver2を格納
├── VisA                      : 
├── mvtecad_dataloader.py     : 
├── mvtecad_mu_dataloader.py  : MU用のデータローダー作成プログラム
├── nsa.py                    : 
├── visa_dataloader.py        :  
├──                           : 
└──                           : 
```


## 2. jsonファイルの概要
- MVTecADディレクトリ
通常の継続学習用にデータセットを分割するためのjsonファイルを格納しています．

- MVTecAD-MUディレクトリ
Machine Unlearning (MU) × 継続学習用ににデータセットを分割するためのjsonファイルを格納しています．


- MVTecAD-MU2ディレクトリ
Machine Unlearning (MU) × 継続学習用ににデータセットを分割するためのjsonファイルを格納しています．
学習用正常データを削減して，学習用異常（扱いとしては正常データ）と同程度になるように抑えています．
