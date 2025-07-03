# --------------- trainer.py ---------------
# FewShotTrainer: few-shot学习训练器
# FewShotTrainer: Few-shot learning trainer
import os
import time

import numpy as np
import torch
from torch import nn

import dataloader


class FewShotTrainer:
    """
    few-shot学习训练器，负责训练、测试和预测流程。
    Few-shot learning trainer, responsible for training, testing, and prediction processes.
    """
    def __init__(self, config, models, optimizers, schedulers):
        """
        初始化训练器。
        Initialize the trainer.
        Args:
            config: 配置对象 / config object
            models: (feature_encoder, relation_network) 特征编码器和关系网络 / tuple of feature encoder and relation network
            optimizers: 优化器字典 / dictionary of optimizers
            schedulers: 学习率调度器字典 / dictionary of schedulers
        """
        self.config = config
        self.feature_encoder, self.relation_network = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.best_accuracy = 0.8  # 最佳准确率阈值 / Best accuracy threshold

    def _process_batch(self, support_set, query_set, batch_num_per_class, class_num):
        """
        处理一个batch，提取特征并生成关系对。
        Process a batch: extract features and generate relation pairs.
        Args:
            support_set: 支持集 / support set
            query_set: 查询集 / query set
            batch_num_per_class: 每类的batch数 / batch size per class
            class_num: 类别数 / number of classes
        Returns:
            关系分数 / relation scores
        """
        # Feature extraction
        # if self.config.sample_num_per_class > 1:
        #     support_set = support_set.view(class_num, self.config.sample_num_per_class, 270, 1000).sum(dim=1).view(class_num, 270, 1000)
        sample_features = self.feature_encoder(support_set.cuda(self.config.gpu))
        if self.config.sample_num_per_class > 1:
            sample_features = sample_features.view(
                class_num,
                self.config.sample_num_per_class,
                self.config.feature_dim,
                -1
            ).mean(1).squeeze(1)
        # sample_features = sample_features.view(-1, self.config.feature_dim, 8)
        query_features = self.feature_encoder(query_set.cuda(self.config.gpu))

        # Create relation pairs
        sample_features_ext = sample_features.unsqueeze(0).repeat(
            batch_num_per_class * class_num, 1, 1, 1)
        query_features_ext = query_features.unsqueeze(0).repeat(
            class_num, 1, 1, 1).transpose(0, 1)

        relation_pairs = torch.cat((sample_features_ext, query_features_ext), 2)
        relation_pairs = relation_pairs.view(-1, self.config.feature_dim * 2, 8)
        return self.relation_network(relation_pairs).view(-1, class_num)

    def _train_episode(self, task):
        """
        训练一个episode。
        Train one episode.
        Args:
            task: few-shot任务对象 / few-shot task object
        Returns:
            损失值和本episode耗时 / loss value and episode duration
        """
        episode_start_time = time.time()

        # Prepare data
        sample_loader = dataloader.get_data_loader(task, self.config.sample_num_per_class, 'support', shuffle=False)
        batch_loader = dataloader.get_data_loader(task, self.config.batch_num_per_class, 'query', shuffle=True)
        samples, _ = next(iter(sample_loader))
        batches, batch_labels = next(iter(batch_loader))

        # 更新config中的class_num以匹配task中的实际人数
        # Update class_num in config to match actual number of persons in task
        self.config.class_num = task.num_persons

        # Forward pass
        relations = self._process_batch(samples, batches, self.config.batch_num_per_class, self.config.class_num)

        # Calculate loss
        mse = nn.MSELoss().cuda(self.config.gpu)
        one_hot_labels = torch.zeros(self.config.batch_num_per_class * self.config.class_num,
                                              self.config.class_num).scatter_(1, batch_labels.view(-1, 1), 1).cuda(self.config.gpu)
        loss = mse(relations, one_hot_labels)

        # Backward pass
        for model in [self.feature_encoder, self.relation_network]:
            model.zero_grad()
        loss.backward()

        # 梯度裁剪 / Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

        # 更新参数（显式调用每个优化器和调度器）
        # Update parameters (explicitly call each optimizer and scheduler)
        self.optimizers["feature"].step()  # 更新特征编码器参数 / update feature encoder
        self.optimizers["relation"].step()  # 更新关系网络参数 / update relation network
        self.schedulers["feature"].step()  # 更新特征编码器学习率 / update feature encoder lr
        self.schedulers["relation"].step()  # 更新关系网络学习率 / update relation network lr

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time

        return loss.item(), episode_duration

    def _test_episode(self, task):
        """
        测试一个episode。
        Test one episode.
        Args:
            task: few-shot任务对象 / few-shot task object
        Returns:
            正确数和损失值 / number of correct predictions and loss value
        """
        sample_loader = dataloader.get_data_loader(task, self.config.sample_num_per_class, 'support', shuffle=False)
        test_loader = dataloader.get_data_loader(task, 1, 'query', shuffle=True)
        samples, _ = next(iter(sample_loader))
        test_data, test_labels = next(iter(test_loader))

        # 更新config中的class_num以匹配task中的实际人数
        self.config.class_num = task.num_persons

        # Forward pass
        relations = self._process_batch(samples, test_data, 1, task.num_persons)

        # Calculate loss
        mse = nn.MSELoss().cuda(self.config.gpu)
        one_hot_labels = torch.zeros(1 * task.num_persons, task.num_persons).scatter_(1, test_labels.view(-1, 1), 1).cuda(
            self.config.gpu)
        loss = mse(relations, one_hot_labels)

        # Calculate accuracy
        _, predict_labels = torch.max(relations.data, 1)
        rewards = [1 if predict_labels[j] == test_labels[j] else 0
                   for j in range(task.num_persons * 1)]
        return np.sum(rewards), loss.item()

    def _predict_episode(self, task):
        """
        预测一个episode。
        Predict one episode.
        Args:
            task: few-shot任务对象 / few-shot task object
        Returns:
            正确数、损失值和耗时 / number of correct predictions, loss value, and duration
        """
        predict_start_time = time.time()

        sample_loader = dataloader.get_data_loader(task, self.config.sample_num_per_class, 'support', shuffle=False)
        test_loader = dataloader.get_data_loader(task, 1, 'query', shuffle=True)
        samples, _ = next(iter(sample_loader))
        test_data, test_labels = next(iter(test_loader))

        # 更新config中的class_num以匹配task中的实际人数
        self.config.class_num = task.num_persons

        # Forward pass
        relations = self._process_batch(samples, test_data, 1, task.num_persons)

        # Calculate loss
        mse = nn.MSELoss().cuda(self.config.gpu)
        one_hot_labels = torch.zeros(1 * task.num_persons, task.num_persons).scatter_(1, test_labels.view(-1, 1),
                                                                                      1).cuda(self.config.gpu)
        predict_loss = mse(relations, one_hot_labels)

        rewards = []
        # Calculate accuracy
        _, predict_labels = torch.max(relations.data, 1)
        for i in range(task.num_persons * 1):
            # print(f"Predicted: {predict_labels[i]}, Actual: {test_labels[i]}")
            rewards.append(1 if predict_labels[i] == test_labels[i] else 0)

        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time

        return np.sum(rewards), predict_loss.item(), predict_duration

    def run_training(self, train_files, test_files):
        """
        训练主循环。
        Main training loop.
        Args:
            train_files: 训练文件列表 / list of training files
            test_files: 测试文件列表 / list of test files
        """
        self.feature_encoder.train()
        self.relation_network.train()
        start_episode = 0
        if os.path.exists(self.config.model_paths['checkpoint']):
            start_episode = self._load_checkpoint()

        # 修改：跟踪总训练时间，但不每30个episode重置
        # Track total training time, but do not reset every 30 episodes
        total_training_time = 0
        training_start_time = time.time()  # 记录整个训练开始时间 / record training start time

        for episode in range(start_episode, self.config.episode):
            task = dataloader.Xrf55Task(
                train_files,
                self.config.class_num,
                self.config.sample_num_per_class,
                self.config.batch_num_per_class
            )

            # Training step
            loss, episode_time = self._train_episode(task)
            total_training_time += episode_time

            if (episode + 1) % self.config.outduration == 0:
                self.config.record = episode + 1
                self._save_models(episode)
                
                # 计算从开始到当前episode的平均时间
                # Calculate average time per episode from start
                elapsed_episodes = episode + 1 - start_episode
                avg_episode_time = total_training_time / elapsed_episodes
                
                print(f"Episode: {episode + 1}, Loss: {loss:.8f}, Avg time per episode: {avg_episode_time:.4f}s, Total time: {total_training_time:.2f}s")
                

            # Validation and model saving
            if (episode + 1) % self.config.outduration == 0:
                # 切换到评估模式 / switch to eval mode
                self.feature_encoder.eval()  
                self.relation_network.eval()  
                
                total_rewards = 0
                total_test_loss = 0
                
                with torch.no_grad():  # 关闭梯度计算 / disable grad
                    for _ in range(self.config.test_episode):
                        task = dataloader.Xrf55Task(
                        test_files,
                        self.config.class_num,
                        self.config.sample_num_per_class,
                        1
                    )
                        rewards, test_loss = self._test_episode(task)
                        total_rewards += rewards
                        total_test_loss += test_loss
                    
                    
                # 必须恢复两个模型的训练模式！/ must restore train mode for both models!
                self.feature_encoder.train()  
                self.relation_network.train()
                
                accuracy = total_rewards / (task.num_persons * 1 * self.config.test_episode)
                avg_test_loss = total_test_loss / self.config.test_episode
                print(f"Episode: {episode + 1}, Accuracy: {accuracy:.4f}, Test Loss: {avg_test_loss:.8f}")
                if accuracy >= self.best_accuracy:
                    self.config.record = 'best'
                    self._save_models(episode)
                    self.best_accuracy = accuracy

    def _save_models(self, episode):
        """
        保存模型和优化器状态。
        Save model and optimizer states.
        Args:
            episode: 当前episode编号 / current episode number
        """
        checkpoint = {
            'episode': episode,
            'feature_encoder_state_dict': self.feature_encoder.state_dict(),
            'relation_network_state_dict': self.relation_network.state_dict(),
            'optimizer_feature_state_dict': self.optimizers['feature'].state_dict(),
            'optimizer_relation_state_dict': self.optimizers['relation'].state_dict(),
            'scheduler_feature_state_dict': self.schedulers['feature'].state_dict(),
            'scheduler_relation_state_dict': self.schedulers['relation'].state_dict(),
            'best_accuracy': self.best_accuracy
        }
        torch.save(checkpoint, self.config.model_paths['checkpoint'])
        print(f"Checkpoint saved at {self.config.model_paths['checkpoint']}")

    def _load_checkpoint(self, sure_to_load=False):
        """
        加载模型和优化器状态。
        Load model and optimizer states.
        Args:
            sure_to_load: 是否确认加载 / whether to confirm loading
        Returns:
            起始episode编号 / starting episode number
        """
        if not sure_to_load:
            sure_to_load = input("A checkpoint already exists. Do you want to load it? (y/n): ")
        if sure_to_load.lower() == 'y':
            checkpoint = torch.load(self.config.model_paths['checkpoint'])
            self.feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
            self.relation_network.load_state_dict(checkpoint['relation_network_state_dict'])
            self.optimizers['feature'].load_state_dict(checkpoint['optimizer_feature_state_dict'])
            self.optimizers['relation'].load_state_dict(checkpoint['optimizer_relation_state_dict'])
            self.schedulers['feature'].load_state_dict(checkpoint['scheduler_feature_state_dict'])
            self.schedulers['relation'].load_state_dict(checkpoint['scheduler_relation_state_dict'])
            self.best_accuracy = checkpoint['best_accuracy']
            start_episode = checkpoint['episode']
            print(f"Checkpoint loaded from {self.config.model_paths['checkpoint']}")
            return start_episode
        else: return 0

    def predict(self, predict_files):
        """
        预测主循环。
        Main prediction loop.
        Args:
            predict_files: 预测文件列表 / list of files for prediction
        """
        self.feature_encoder.eval()
        self.relation_network.eval()

        total_rewards = 0
        total_predict_loss = 0
        total_predict_time = 0

        task = None
        for episode in range(self.config.predict_episode):
            task = dataloader.Xrf55Task(
                predict_files,
                self.config.class_num,
                self.config.sample_num_per_class,
                1
            )
            rewards, predict_loss, predict_time = self._predict_episode(task)
            total_rewards += rewards
            total_predict_loss += predict_loss
            total_predict_time += predict_time

        accuracy = total_rewards / (task.num_persons * 1 * self.config.predict_episode)
        avg_predict_loss = total_predict_loss / self.config.predict_episode
        avg_predict_time = total_predict_time / (self.config.predict_episode * task.num_persons)

        print(f"Accuracy: {accuracy:.4f}, Predict Loss: {avg_predict_loss:.8f}")
        print(f"Total prediction time: {total_predict_time:.2f}s, Average time per prediction: {avg_predict_time:.4f}s")