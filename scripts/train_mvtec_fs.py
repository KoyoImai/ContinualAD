import sys
import os
sys.path.append(os.getcwd())
from share import *
from utils.util import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.mvtecad_dataloader import MVTecDataset_cad
from cdm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# few-shot用の追加
from utils.fewshot_helper import stratified_ratio_indices
from torch.utils.data import Subset


def main(args):
    setup_seed(args.seed)

    # log名の決定
    log_name = f'debug_fewshot_mvtec_setting{args.setting}'

    # modelの作成
    model = create_model(args.config_path).cpu()
    model.set_gpm(args.gpm == "on")
    print("model.use_gpm: ", model.use_gpm)

    # stable difusionの事前学習済みパラメータを読み込み
    weights = torch.load(args.resume_path)
    select_weights = {key: weights[key] for key in weights if not 'control_model' in key}
    model.load_state_dict(select_weights, strict=False)

    # 学習率の決定
    model.learning_rate = args.learning_rate

    # データセットの作成
    train_dataset, task_num = MVTecDataset_cad('train', args.data_path, args.setting)
    test_dataset, _ = MVTecDataset_cad('test', args.data_path, args.setting)
    

    # few-shot用に追加タスクのデータセットを修正
    if args.inc_sample_ratio < 1.0:
        for i in range(1, task_num):

            full_size = len(train_dataset[i])
            idx = stratified_ratio_indices(
                train_dataset[i],
                ratio=args.inc_sample_ratio,
                min_per_class=args.fewshot_min,
                seed=args.fewshot_seed,
            )

            train_dataset[i] = Subset(train_dataset[i], idx)
            print(f"[FewShot] task{i}: {len(idx)} / {full_size} samples")


    # 各タスクを順番に処理
    for i in range(task_num):

        if i == 0:
            epoch = args.base_epoch
        else:
            epoch = args.inc_epoch

        # logの名前を設定
        model.set_log_name(log_name + f'/task{i}')

        #
        ckpt_callback_val = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f'./incre_val/{log_name}/',
            filename=f'task{i}_best',
            mode='max')

        # trainerの作成
        trainer = pl.Trainer(gpus=1, precision=32,
                    callbacks=[ckpt_callback_val, ],
                    num_sanity_val_steps=0,
                    accumulate_grad_batches=1,     # Do not change!!!
                    max_epochs=epoch,
                    check_val_every_n_epoch=args.check_v,
                    enable_progress_bar=True
                    )


        # dataloaderの作成
        train_dataloader = DataLoader(train_dataset[i], num_workers=8, batch_size=args.batch_size, shuffle=True)
        gpm_dataloader = DataLoader(train_dataset[i], num_workers=8, batch_size=args.gpm_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset[i], num_workers=8, batch_size=args.batch_size, shuffle=False)

        # model の訓練
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        
        model.load_state_dict(load_state_dict(trainer.checkpoint_callback.best_model_path, location='cuda'), strict=False)

        # gpmの基底計算を実行
        if args.gpm in ["on", "collect"]:
            trainer.test(model, dataloaders=gpm_dataloader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CDAD")

    parser.add_argument("--resume_path", default='./models/base.ckpt')

    parser.add_argument("--data_path", default="./data/mvtec_anomaly_detection", type=str)

    parser.add_argument("--setting", default=1, type=int)

    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--batch_size", default=6, type=int)

    parser.add_argument("--gpm_batch_size", default=1, type=int)

    parser.add_argument("--learning_rate", default=1e-5, type=float)

    parser.add_argument("--base_epoch", default=1, type=int)    # ベースタスクの学習エポック数

    parser.add_argument("--inc_epoch", default=3, type=int)     # 追加タスクの学習エポック数

    parser.add_argument("--config_path", default="models/cdad_mvtec.yaml", type=str)    # configファイルまでのパス

    parser.add_argument("--gpm", choices=["on", "collect", "off"], default="on")        # gpmによる勾配直交を行うか

    parser.add_argument("--check_v", default=1, type=int)

    # few-shot用
    parser.add_argument("--inc_sample_ratio", default=0.5, type=float)     # few-shot時の追加タスクのサンプル割合
    parser.add_argument("--fewshot_min", default=10, type=int)            # few-shotサンプル数の加減
    parser.add_argument("--fewshot_seed", default=0, type=int)             # few-shotサンプル選択時のseed値

    args = parser.parse_args()

    main(args)





