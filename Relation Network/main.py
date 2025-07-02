# --------------- main.py ---------------
# 训练主入口 / Main entry for training
import os
import torch
import random
import numpy as np

from config import TrainingConfig
from models import ModelManager
from trainer import FewShotTrainer
import dataloader


def main():
    """
    主函数，执行训练流程。
    Main function, executes the training process.
    """
    config = TrainingConfig()
    # 创建模型保存目录（如不存在）/ Create model save directory if not exists
    os.makedirs("./checkpoints", exist_ok=True)

    # 初始化模型 / Initialize models
    feature_encoder, relation_network = ModelManager.initialize_models(config)
    optimizers, schedulers = ModelManager.setup_optimizers(feature_encoder, relation_network, config)

    # 准备数据 / Prepare data
    # NOTE: 可在此自定义数据集路径
    # NOTE: You can customize the dataset path here
    train_path = f"./your_data_path/{config.mode}/train"  # 建议修改为你自己的路径 / Please change to your own path
    test_path = f"./your_data_path/{config.mode}/test"   # 建议修改为你自己的路径 / Please change to your own path
    train_files, test_files = dataloader.get_xrf55_files(train_path), dataloader.get_xrf55_files(test_path)

    # 启动训练 / Start training
    trainer = FewShotTrainer(
        config=config,
        models=(feature_encoder, relation_network),
        optimizers=optimizers,
        schedulers=schedulers
    )
    trainer.run_training(train_files, test_files)


if __name__ == '__main__':
    main()