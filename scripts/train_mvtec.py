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


def main(args):
    setup_seed(args.seed)

    # log名の決定
    log_name = f'mvtec_setting{args.setting}'

    # modelの作成
    model = create_model('models/cdad_mvtec.yaml').cpu()

    # stable difusionの事前学習済みパラメータを読み込み
    weights = torch.load(args.resume_path)
    select_weights = {key: weights[key] for key in weights if not 'control_model' in key}
    model.load_state_dict(select_weights, strict=False)

    # 学習率の決定
    model.learning_rate = args.learning_rate

    # データセットの作成
    train_dataset, task_num = MVTecDataset_cad('train', args.data_path, args.setting)
    test_dataset, _ = MVTecDataset_cad('test', args.data_path, args.setting)

    # 各タスクを順番に処理
    for i in range(task_num):

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
                    max_epochs=args.max_epoch,
                    check_val_every_n_epoch=args.check_v,
                    enable_progress_bar=False
                    )


        # dataloaderの作成
        train_dataloader = DataLoader(train_dataset[i], num_workers=8, batch_size=args.batch_size, shuffle=True)
        gpm_dataloader = DataLoader(train_dataset[i], num_workers=8, batch_size=args.gpm_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset[i], num_workers=8, batch_size=args.batch_size, shuffle=False)

        # model の訓練
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        
        model.load_state_dict(load_state_dict(trainer.checkpoint_callback.best_model_path, location='cuda'), strict=False)

        # test is used to process gradient projection
        trainer.test(model, dataloaders=gpm_dataloader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CDAD")

    parser.add_argument("--resume_path", default='./models/base.ckpt')

    parser.add_argument("--data_path", default="./data/mvtec_anomaly_detection", type=str)

    parser.add_argument("--setting", default=1, type=int)

    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--batch_size", default=12, type=int)

    parser.add_argument("--gpm_batch_size", default=1, type=int)

    parser.add_argument("--learning_rate", default=1e-5, type=float)

    parser.add_argument("--max_epoch", default=500, type=int)    # ベースタスクの学習エポック数

    parser.add_argument("--inc_epoch", default=100, type=int)    # 追加タスクの学習エポック数

    parser.add_argument("--config_path", default="models/cdad_mvtec.yaml", type=str)    # configファイルまでのパス

    parser.add_argument("--check_v", default=25, type=int)

    args = parser.parse_args()

    main(args)





