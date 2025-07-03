# --------------- predict.py ---------------
# 预测脚本 / Prediction script
import os

import torch

import dataloader
from config import TrainingConfig
from models import ModelManager
from trainer import FewShotTrainer


def main():
    """
    主函数，执行预测流程。
    Main function, executes the prediction process.
    """
    config = TrainingConfig()
    
    # 初始化模型 / Initialize models
    feature_encoder, relation_network = ModelManager.initialize_models(config)
    optimizers, schedulers = ModelManager.setup_optimizers(feature_encoder, relation_network, config)

    # 如果有预训练权重则加载 / Load pretrained weights if available
    if os.path.exists(config.model_paths["checkpoint-p"]):
        checkpoint = torch.load(config.model_paths["checkpoint-p"], weights_only=False)
        feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
        relation_network.load_state_dict(checkpoint['relation_network_state_dict'])

    # 准备数据 / Prepare data
    # NOTE: 可在此自定义数据集路径
    # NOTE: You can customize the dataset path here
    predict_path = f"./your_data_path/{config.mode}/test"  # 建议修改为你自己的路径 / Please change to your own path
    predict_files = dataloader.get_xrf55_files(predict_path)

    # 启动预测 / Start prediction
    trainer = FewShotTrainer(
        config=config,
        models=(feature_encoder, relation_network),
        optimizers=optimizers,
        schedulers=schedulers
    )
    trainer.predict(predict_files)

if __name__ == '__main__':
    main()