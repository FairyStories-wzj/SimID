# Relation Network

## 项目简介 | Project Introduction

本文件夹实现了基于关系网络（Relation Network）的Few-shot学习方法，适用于小样本分类任务，主要用于波形数据的人物-动作识别。
This folder implements a Relation Network-based few-shot learning method, suitable for small-sample classification tasks, mainly for person-action recognition on waveform data.

---

## 运行前需自定义/修改的地方 | What You Need to Customize/Modify Before Running

- **数据路径**：所有涉及数据集路径的地方（如`main.py`, `predict.py`, `database_test.py`, `test.py`等），均已用`./your_data_path/`作为占位符。请根据你的实际数据存放位置，修改为你自己的数据路径。
  - Data Path: All dataset paths (e.g., in `main.py`, `predict.py`, `database_test.py`, `test.py`, etc.) use `./your_data_path/` as a placeholder. Please change it to your actual data path.
- **模型保存路径**：`config.py`中的模型保存路径默认是`./checkpoints/`，如有需要可自定义。
  - Model Save Path: The model save path in `config.py` defaults to `./checkpoints/`. You can customize it if needed.
- **参数设置**：所有训练、预测参数均可通过命令行传递，详见`config.py`，如类别数、shot数、学习率、GPU编号等。
  - Parameter Settings: All training and prediction parameters can be passed via command line, see `config.py` for details (e.g., class number, shot number, learning rate, GPU id, etc.).
- **依赖环境**：请确保已安装所有依赖包，建议使用`requirements.txt`统一安装。
  - Dependencies: Please make sure all dependencies are installed. It is recommended to use `requirements.txt` for unified installation.

---

## 依赖环境 | Dependencies

- Python 3.7+

建议使用`requirements.txt`统一安装依赖。
It is recommended to use `requirements.txt` to install dependencies.

---

## 主要文件说明 | Main File Descriptions

- `main.py`：训练主入口，执行训练流程。
  Main entry for training, executes the training process.
- `predict.py`：预测脚本，加载模型并对测试集进行预测。
  Prediction script, loads model and predicts on test set.
- `config.py`：训练参数和模型路径配置。
  Training parameters and model path configuration.
- `dataloader.py`：数据加载与任务生成，支持Few-shot任务采样。
  Data loading and task generation, supports few-shot task sampling.
- `models.py`：模型管理器，负责模型初始化、优化器和调度器设置。
  Model manager, responsible for model/optimizer/scheduler setup.
- `trainer.py`：Few-shot训练、测试与预测核心逻辑。
  Core logic for few-shot training, testing, and prediction.
- `RelationNetworkWithResNet.py`：关系网络结构定义。
  Relation network structure definition.
- `SEResNet.py`：SE-ResNet特征提取网络。
  SE-ResNet feature extraction network.
- `experienment_script.py`：批量实验脚本，自动多次运行预测并统计结果。
  Batch experiment script, runs prediction multiple times and summarizes results.

---

## 数据路径说明 | Data Path Instructions

- 所有涉及数据集路径的地方（如`main.py`, `predict.py`），均已用`./your_data_path/`作为占位符。
- 请根据你的实际数据存放位置，修改为你自己的数据路径。
- All dataset paths (e.g., in `main.py`, `predict.py`) use `./your_data_path/` as a placeholder. Please change it to your actual data path.

---

## 运行方法 | How to Run

### 训练 | Training
```bash
python main.py [optional_parameter]
```

### 预测 | Prediction
```bash
python predict.py [optional_parameter]
```

### 批量实验 | Batch Experiment
```bash
python experienment_script.py
```

---

## 参数自定义 | Parameter Customization

所有训练、预测参数均可通过命令行传递，详见`config.py`。
All training and prediction parameters can be passed via command line, see `config.py` for details.

---

## 联系方式 | Contact
如有问题请联系项目维护者。
If you have any questions, please contact the project maintainer.
