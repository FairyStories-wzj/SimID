"""
the dataset loader for prototypical network and XRF55 dataset
"""
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import Dataset

def loadToMemEnhanced(data_path, dataset_name):
    """
    :param data_path: the path of the dataset
    :param dataset_name: the name of the dataset ("training" or "test")
    :return datas: a 3-dimensional list, data[i][j][k] represents category i, action j, and sample k
    :return:
        num_class: the number of  categories
        num_action: the number of actions
        minimal_sample: at least how many samples for each action of each category
    """
    print(f"loading {dataset_name} set...")

    # extract from files
    people, actions = [], []
    npy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    for file in npy_files:
        person = file.split('_')[0]
        action = file.split('_')[1]
        people.append(person)
        actions.append(action)
    people, actions = list(set(people)), list(set(actions))
    # renumber is needed because the indexes do not start from 0 and are not necessarily continuous
    num_class, num_action = len(people), len(actions)

    datas = [[[] for _ in range(num_action)] for _ in range(num_class)]
    for file in npy_files:
        # print("loading：", file)
        file_path = os.path.join(data_path, file)
        array = np.load(file_path)
        tensor = torch.from_numpy(array).to(torch.float32).view(270, 1000)
        if torch.isnan(tensor).any():
            "Input contains nan!"
            continue
        person = file.split('_')[0]
        action = file.split('_')[1]
        datas[people.index(person)][actions.index(action)].append(tensor)

    minimal_sample = min(len(datas[person_id][action_id])
                         for person_id in range(num_class)
                         for action_id in range(num_action))

    print(f"{dataset_name} is loaded，with {num_class} categories, {num_action} actions，at least {minimal_sample} samples.")
    return datas, num_class, num_action, minimal_sample


class Xrf55TrainEnhanced(Dataset):
    def __init__(self, data_path, shot, batch_size, dataset_name):
        self.datas, self.num_class, self.num_action, minimal_sample = loadToMemEnhanced(data_path, dataset_name)

        self.batch_size = batch_size

        if shot > minimal_sample - 1:
            print(f"warning：there are less than {shot} samples in the dataset，the shot of {dataset_name} has been modified to {minimal_sample - 1}")
        self.shot = min(shot, minimal_sample - 1)

        self.selectedAction, self.support, self.query = None, None, None

    def __len__(self):
        return sys.maxsize

    def selectAndSplit(self):
        """
        select an action, and split the training support set and training query set for it
        """
        self.selectedAction = random.choice(range(self.num_action))

        self.support = [[] for _ in range(self.num_class)]
        self.query = [[] for _ in range(self.num_class)]

        for person_id in range(self.num_class):
            self.support[person_id] = random.sample(self.datas[person_id][self.selectedAction], self.shot)
            self.query[person_id] = [wave
                                     for wave in self.datas[person_id][self.selectedAction]
                                     if not any(torch.equal(wave, w)
                                                for w in self.support[person_id])]

        self.support = [t for sublist in self.support for t in sublist]  # 展平

    def getSupport(self):
        return torch.stack(self.support)

    def __getitem__(self, item):
        """
        select a sample and its label from the training query set
        :param item: Parameters automatically passed in by class torch.utils.data.Dataset
        :return: a query sample and its label
        """
        if item % self.batch_size == 0:  # split the training set at the beginning of each batch
            self.selectAndSplit()

        label = random.choice(range(self.num_class))
        wave = random.choice(self.query[label])
        return wave, label


class Xrf55Predict(Dataset):
    def __init__(self, data_path, shot, times, dataset_name):
        super().__init__()
        self.datas, self.num_class, self.num_action, minimal_sample = loadToMemEnhanced(data_path, dataset_name)

        self.shot, self.times = shot, times
        if self.shot > minimal_sample - 1:
            print(f"warning：there are less than {shot} samples in the dataset，the shot of {dataset_name} has been modified to {minimal_sample - 1}")
            self.shot = minimal_sample - 1

        self.selectedAction, self.support, self.query = None, None, None

    def __len__(self):
        return self.times

    def selectAndSplit(self):
        """
        select an action, and split the test support set and test query set for it
        """
        self.selectedAction = random.choice(range(self.num_action))

        self.support = [[] for _ in range(self.num_class)]
        self.query = [[] for _ in range(self.num_class)]

        for person_id in range(self.num_class):
            self.support[person_id] = random.sample(self.datas[person_id][self.selectedAction], self.shot)
            self.query[person_id] = [wave
                                     for wave in self.datas[person_id][self.selectedAction]
                                     if not any(torch.equal(wave, w)
                                                for w in self.support[person_id])]

        self.support = [t for sublist in self.support for t in sublist]

    def getSupport(self):
        return torch.stack(self.support)

    def __getitem__(self, item):
        if item % self.times == 0:
            self.selectAndSplit()

        label = random.choice(range(self.num_class))
        wave = random.choice(self.query[label])
        return wave, label
