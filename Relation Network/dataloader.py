import os
import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Xrf55Task(object):
    """
    用于生成基于人物和动作的few-shot学习任务，确保每个任务中人物不重复。
    Used to generate few-shot learning tasks based on persons and actions, ensuring no duplicate persons in each task.
    """

    def __init__(self, all_files, num_persons, shots_per_person, query_per_person):
        """
        初始化任务。
        Initialize the task.
        Args:
            all_files: 所有波形文件路径的列表 / list of all waveform file paths
            num_persons: 每个任务包含的人数(n-way) / number of persons per task (n-way)
            shots_per_person: 每人训练样本数(k-shot) / number of training samples per person (k-shot)
            query_per_person: 每人测试样本数 / number of test samples per person
        """
        self.all_files = all_files
        self.num_persons = num_persons
        self.shots_per_person = shots_per_person
        self.query_per_person = query_per_person

        # 使用pandas批量处理 / Use pandas for batch processing
        df = pd.DataFrame({'path': all_files})
        # 提取人物ID和动作ID / Extract person_id and action_id
        df['person_id'] = df['path'].str.split('/').str[-1].str.split('_').str[0]
        df['action_id'] = df['path'].str.split('/').str[-1].str.split('_').str[1]
        
        # 首先按动作分组 / Group by action first
        action_groups = df.groupby('action_id')
        action_person_samples = {}
        
        # 随机选择一个动作 / Randomly select an action
        available_actions = list(action_groups.groups.keys())
        selected_action = random.choice(available_actions)
        action_df = action_groups.get_group(selected_action)
        
        # 对选中的动作，按人物分组 / For the selected action, group by person
        person_groups = action_df.groupby('person_id')
        
        # 筛选出有足够样本的人物 / Select persons with enough samples
        valid_persons = {}
        for person, group in person_groups:
            files = group['path'].tolist()
            if len(files) >= (shots_per_person + query_per_person):
                valid_persons[person] = files
        
        if len(valid_persons) < num_persons:
            print(f"动作 {selected_action} 下可用的不同人物数量不足: 需要{num_persons}个，只 有{len(valid_persons)}个可用")
            self.num_persons = len(valid_persons)
            num_persons = self.num_persons
            print(f"将n-way设置为{self.num_persons}")
        
        # 随机选择指定数量的人物 / Randomly select the specified number of persons
        selected_persons = random.sample(list(valid_persons.keys()), num_persons)
        # 只使用人物ID作为标签 / Use person_id as label
        self.person_to_label = {person: idx for idx, person in enumerate(selected_persons)}

        # 为每个选中的人物分配训练和测试样本 / Assign train and test samples for each selected person
        self.train_waves = []
        self.test_waves = []
        self.train_labels = []
        self.test_labels = []

        for person in selected_persons:
            samples = valid_persons[person]
            # 随机打乱样本 / Shuffle samples
            random.shuffle(samples)

            # 分配训练和测试样本 / Assign train and test samples
            train_samples = samples[:shots_per_person]
            test_samples = samples[shots_per_person:shots_per_person + query_per_person]

            self.train_waves.extend(train_samples)
            self.test_waves.extend(test_samples)

            # 添加对应的标签（只使用人物ID）/ Add corresponding labels (use person_id only)
            label = self.person_to_label[person]
            self.train_labels.extend([label] * len(train_samples))
            self.test_labels.extend([label] * len(test_samples))


class XRF55FewShot(Dataset):
    """
    Few-shot学习数据集。
    Few-shot learning dataset.
    """

    def __init__(self, task, split='support'):
        """
        初始化few-shot数据集。
        Initialize the few-shot dataset.
        Args:
            task: Xrf55Task对象 / Xrf55Task object
            split: 'support'为支持集，'query'为查询集 / 'support' for support set, 'query' for query set
        """
        super(XRF55FewShot, self).__init__()
        self.task = task
        self.split = split
        self.waves = self.task.train_waves if self.split == 'support' else self.task.test_waves
        self.labels = self.task.train_labels if self.split == 'support' else self.task.test_labels
        
        # 检查路径有效性 / Check path validity
        for wave_path in self.waves:
            if not os.path.exists(wave_path):
                raise FileNotFoundError(f"找不到文件: {wave_path}")
        
        # 初始化空缓存 / Initialize empty cache
        self.cache = {}

    def __len__(self):
        """
        返回样本数量。
        Return the number of samples.
        """
        return len(self.waves)

    def __getitem__(self, idx):
        """
        获取指定索引的数据和标签。
        Get the data and label at the specified index.
        Args:
            idx: 索引 / index
        Returns:
            (wave, label): 波形数据和标签 / waveform data and label
        """
        wave_path = self.waves[idx]
        
        # 检查缓存中是否存在该数据 / Check if data exists in cache
        if wave_path in self.cache:
            wave_data = self.cache[wave_path]
        else:
            try:
                # 如果不在缓存中，则加载数据 / Load data if not in cache
                wave_data = np.load(wave_path)
                
                # 检查数据尺寸是否匹配 / Check if data shape matches
                if wave_data.size != 270 * 1000:
                    raise ValueError(f"文件 {wave_path} 的数据尺寸不匹配: 需要270 * 1000={270 * 1000}, 实际为{wave_data.size}")
                
                # 将加载的数据加入缓存 / Add loaded data to cache
                self.cache[wave_path] = wave_data
            except Exception as e:
                print(f"加载文件 {wave_path} 时出错: {e}")
                raise ValueError(f"索引 {idx} 对应的数据加载失败") from e
        
        wave = torch.from_numpy(wave_data).to(torch.float32).view(270, 1000).to(device='cuda')
        label = self.labels[idx]
        return wave, label

class ClassBalancedSampler:
    """
    确保每个batch包含均衡的类别样本。
    Ensure each batch contains balanced class samples.
    """

    def __init__(self, num_per_class, num_classes, num_samples, shuffle=False):
        """
        初始化采样器。
        Initialize the sampler.
        Args:
            num_per_class: 每类样本数 / number of samples per class
            num_classes: 类别数 / number of classes
            num_samples: 每类总样本数 / total samples per class
            shuffle: 是否打乱 / whether to shuffle
        """
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.shuffle = shuffle

    def __iter__(self):
        """
        返回采样索引的迭代器。
        Return an iterator of sample indices.
        """
        if self.shuffle:
            batch = [[i + j * self.num_samples for i in torch.randperm(self.num_samples)[:self.num_per_class]]
                     for j in range(self.num_classes)]
        else:
            batch = [[i + j * self.num_samples for i in range(self.num_samples)[:self.num_per_class]]
                     for j in range(self.num_classes)]

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        """
        返回batch数量（始终为1）。
        Return the number of batches (always 1).
        """
        return 1

def get_xrf55_files(data_path, mode='train'):
    """
    获取所有.npy文件并按类别（文件名前两位）组织，直接返回所有文件路径。
    Get all .npy files and organize by class (first two chars of filename), return all file paths directly.
    Args:
        data_path: 数据目录 / data directory
        mode: 模式（未使用）/ mode (unused)
    Returns:
        all_files: 所有文件路径列表 / list of all file paths
    """
    class_files = {}

    # 按类别收集文件 / Collect files by class
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                class_id = file.split('_')[0]
                action_id = file.split('_')[1]
                
                if not (class_id == '12' and action_id == '05'):
                    file_path = os.path.join(root, file)
                    if class_id not in class_files:
                        class_files[class_id] = []
                    class_files[class_id].append(file_path)

    # 获取所有文件路径 / Get all file paths
    all_files = [f for files in class_files.values() for f in files]

    print(f"总类别数: {len(class_files)}")
    print(f"总文件数: {len(all_files)}")

    return all_files


def get_data_loader(task, num_per_class=1, split='support', shuffle=False):
    """
    创建数据加载器。
    Create a data loader.
    Args:
        task: Xrf55Task对象 / Xrf55Task object
        num_per_class: 每类样本数 / number of samples per class
        split: 'support'或'query' / 'support' or 'query'
        shuffle: 是否打乱 / whether to shuffle
    Returns:
        DataLoader对象 / DataLoader object
    """
    dataset = XRF55FewShot(task, split=split)

    if split == 'support':
        sampler = ClassBalancedSampler(
            num_per_class=num_per_class,
            num_classes=task.num_persons,
            num_samples=task.shots_per_person,
            shuffle=shuffle
        )
    else:
        sampler = ClassBalancedSampler(
            num_per_class=num_per_class,
            num_classes=task.num_persons,
            num_samples=task.query_per_person,
            shuffle=shuffle
        )

    loader = DataLoader(
        dataset,
        batch_size=num_per_class * task.num_persons,
        sampler=sampler
    )
    return loader