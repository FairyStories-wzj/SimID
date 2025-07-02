# --------------- config.py ---------------
# 配置文件，定义训练参数和模型保存路径
# Configuration file, defines training parameters and model save paths
import argparse


class TrainingConfig:
    """
    训练配置类，包含所有训练相关参数。
    Training configuration class, contains all training-related parameters.
    """
    def __init__(self):
        parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
        parser.add_argument("-f", "--feature_dim", type=int, default=64,
                            help="特征维度 / Feature dimension")
        parser.add_argument("-r", "--relation_dim", type=int, default=8,
                            help="关系维度 / Relation dimension")
        parser.add_argument("-w", "--class_num", type=int, default=8,
                            help="类别数（n-way）/ Number of classes (n-way)")
        parser.add_argument("-s", "--sample_num_per_class", type=int, default=10,
                            help="每类支持集样本数（k-shot）/ Number of support samples per class (k-shot)")
        parser.add_argument("-b", "--batch_num_per_class", type=int, default=6,
                            help="每类查询集样本数 / Number of query samples per class")
        parser.add_argument("-e", "--episode", type=int, default=2000,
                            help="训练总episode数 / Total number of training episodes")
        parser.add_argument("-re", "--record", type=str, default='best',
                            help="记录模型的标识 / Model record identifier")
        parser.add_argument("-o", "--outduration", type=int, default=30,
                            help="每隔多少episode保存一次模型 / Save model every N episodes")
        parser.add_argument("-t", "--test_episode", type=int, default=100,
                            help="每次验证的episode数 / Number of test episodes per validation")
        parser.add_argument("-l", "--learning_rate", type=float, default=1e-2,
                            help="学习率 / Learning rate")
        parser.add_argument("-g", "--gpu", type=int, default=3,
                            help="使用的GPU编号 / GPU id to use")
        parser.add_argument("-u", "--hidden_unit", type=int, default=10,
                            help="隐藏单元数 / Number of hidden units")
        parser.add_argument("-p", "--predict_episode", type=int, default=100,
                            help="预测时的episode数 / Number of prediction episodes")
        parser.add_argument("-pr", "--train_shot", type=int, default=10,
                            help="预测时使用的shot数 / Number of shots used in prediction")
        parser.add_argument("-m", "--mode", type=str, default='CP',
                            help="数据集模式 / Dataset mode")
        args = parser.parse_args()
        self.__dict__.update(vars(args))

    @property
    def model_paths(self):
        """
        返回模型保存路径。
        Return model save paths.
        注意：请根据实际环境修改为你自己的保存路径！
        Note: Please modify to your own save path according to your environment!
        """
        return {
            # 示例路径，建议替换为你自己的路径，如 './checkpoints/'
            # Example path, please replace with your own, e.g. './checkpoints/'
            "checkpoint": f"./checkpoints/{self.mode}/xrf55_model_{self.class_num}way_{self.sample_num_per_class}shot_{self.record}.pkl",
            "checkpoint-p": f"./checkpoints/{self.mode}/xrf55_model_{self.class_num}way_{self.train_shot}shot_{self.record}.pkl"
        }