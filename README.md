# One-for-Moreをベースにした異常検知の継続学習
**One-for-More: Continual Diffusion Model for Anomaly Detection**

## 1. プログラムの概要
学習・評価に使用するプログラムの全体像は以下の通りです．

```
ContinualAD
├── cdm                  : CDADの主要プログラムが格納されたディレクトリ
├── ldm                  : Latent Diffusion Model関連のプログラムを格納したディレクトリ
├── models               : 
├── scripts              : 
├── training             : 
├── utils                : 
├── build_base_model.py  : 
├── config.py            : 
├── install.sh           : 
└── share.py             : 
```


## 2.Dataset
### 2.1 MVTec-AD
<!-- - **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move them to `./data/mvtec_anomaly_detection/`. The MVTec-AD dataset directory should be as follows.  -->
MVTec-ADデータセットのディレクトリ構成は以下の通りです．
[MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)からデータセットをダウンロードし．以下のように配置してください．

```
|-- data
    |-- mvtec_anomaly_detection
        |-- bottle
            |-- ground_truth
                |-- broken_large
                    |-- 000_mask.png
                |-- broken_small
                    |-- 000_mask.png
                |-- contamination
                    |-- 000_mask.png
            |-- test
                |-- broken_large
                    |-- 000.png
                |-- broken_small
                    |-- 000.png
                |-- contamination
                    |-- 000.png
                |-- good
                    |-- 000.png
            |-- train
                |-- good
                    |-- 000.png
```

### 2.2 VisA
<!-- - **Create the VisA dataset directory**. Download the VisA dataset from [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Unzip the file and move them to `./VisA/`. The VisA dataset directory should be as follows.  -->
Visaデータセットのディレクトリ構成は以下の通りです．
[VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)からデータセットをダウンロードし．以下のように配置してください．

```
|-- data
    |-- VisA
        |-- candle
            |-- Data
                |-- Images
                    |-- Anomaly
                        |-- 000.JPG
                    |-- Normal
                        |-- 0000.JPG
                |-- Masks
                    |--Anomaly 
                        |-- 000.png        
```


## 3. 事前学習済みモデルの準備
<!-- First download the checkpoint of AutoEncoder and diffusion model, we use the pre-trained stable diffusion v1.5. -->
以下の手順に従ってAutoEncoderとDiffusion Modelの学習済みパラメータを用意してください．

    $ wget https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
    $ unzip kl-f8.zip
    $ wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
    

<!-- Then run the code to get the output model `./models/base.ckpt`. -->
その後，`./models/base.ckpt`.を実行してモデルを獲得してください．

    $ python build_base_model.py


## 4. 訓練の実行方法
<!-- The incremental settings for the MVTec and VisA datasets are shown in the table. -->
MVTecとVisaデータセットのタスク設定は以下の表に示す通りです．

| Dataset | Setting ID |   Incremental setting   |
|:-------:|:--:|:-----------------------:|
|  MVTec  |  1 |    14 - 1 with 1 Step   |
|  MVTec  |  2 |    10 - 5 with 1 Step   |
|  MVTec  |  3 |    3 ✖️ 5 with 5 Steps   |
|  MVTec  |  4 | 10 - 1 ✖️ 5 with 5 Steps |
|   VisA  |  1 |    11 - 1 with 1 Step   |
|   VisA  |  2 |    8 - 4 with 1 Step    |
|   VisA  |  3 |   8 - 1 ✖️ with 4 Steps   |

通常異常検知継続学習の実行は以下を実行してください．

- CDAD ([paper](https://arxiv.org/pdf/2502.19848)):
    ```
    python scripts/train_mvtec.py --config_path "models/cdad_mvtec.yaml" --setting [ID]
    python scripts/train_visa.py --config_path "models/cdad_mvtec.yaml" --setting [ID]
    ```
- Diffusion Model
    ```
    python scripts/train_mvtec.py --config_path "models/cdad_mvtec.yaml" --gpm "off" --setting [ID]
    python
    ```

Few-shot 異常検知継続学習の実行は以下を実行してください．
- CDAD ([paper](https://arxiv.org/pdf/2502.19848)):
    ```
    python scripts/train_mvtec_fs.py --config_path "models/cdad_mvtec.yaml" --setting [ID]
    python scripts/train_visa_fs.py --config_path "models/cdad_mvtec.yaml" --setting [ID]
    ```
- Diffusion Model
    ```
    python scripts/train_mvtec_fs.py --config_path "models/cdad_mvtec.yaml" --gpm "off" --setting [ID]
    python
    ```

Machine Unlearning 異常検知継続学習の実行は以下を実行してください．
- CDAD ([paper](https://arxiv.org/pdf/2502.19848)):
    ```
    python scripts/train_mvtec_mu.py --config_path "models/cdad_mvtec.yaml" --setting [ID]
    python scripts/train_visa_mu.py --config_path "models/cdad_mvtec.yaml" --setting [ID]
    ```
- Diffusion Model
    ```
    python scripts/train_mvtec_mu.py --config_path "models/cdad_mvtec.yaml" --gpm "off" --log_base "default"  --setting 1
    python  
    ```

TensorBoardは以下で実行可能です．
    
    tensorboard --logdir /home/kouyou/ContinualLearning/repexp/One-for-More/tb_logs/ --port 6006 --host localhost


The images are saved under `./log_image/`
The training logs are saved under `./log`

## 5. Test
The trained checkpoints of `MVTec, setting [s], task [t]` are saved under `./incre_val/mvtec_setting[s]/task[t]_best.ckpt`. For evaluation and visualization, run the following code:

    $ python scripts/test_mvtec.py --setting [s] --task [t]
    $ python scripts/test_visa.py --setting [s] --task [t]

The test results are saved under `./Test/`


