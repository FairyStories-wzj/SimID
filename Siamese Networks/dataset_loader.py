"""
the dataset loader for siamese networks and XRF55 dataset
"""
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def loadToMemEnhanced(data_path, dataset_name):
    print(f"loading {dataset_name} set...")

    people, actions = [], []
    npy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    for file in npy_files:
        # the XRF55 dataset name the samples by "XX_YY_ZZ.npy", where the XX represents the user, YY represents action, and ZZ represents try
        person = file.split('_')[0]
        action = file.split('_')[1]
        people.append(person)
        actions.append(action)
    people, actions = list(set(people)), list(set(actions))
    num_class, num_action = len(people), len(actions)

    # datas[i][j][k] represents the sample for person i, action j, and try k
    datas = [[[] for _ in range(num_action)] for _ in range(num_class)]
    for file in npy_files:
        # print("loading：", file)
        file_path = os.path.join(data_path, file)
        array = np.load(file_path)
        tensor = torch.from_numpy(array).to(torch.float32).view(270, 1000)  # each sample in XRF55 dataset is a 270*1000 waveform
        if torch.isnan(tensor).any():
            "Input contains nan!"
            continue
        person = file.split('_')[0]
        # please note that datas[i][j][k]'s i may not necessarily match the XX in the sample file names; it depends on the order in which you read the files.
        action = file.split('_')[1]
        # same for datas[i][j][k]'s j and YY, or data[i][j][k]'s k and ZZ
        datas[people.index(person)][actions.index(action)].append(tensor)

    print(f"{dataset_name} set is loaded，with {num_class} categories and {num_action} actions")
    return datas, num_class, num_action


class Xrf55TrainEnhanced(Dataset):
    """
    the data loader for training set
    the training strategy of Siamese Networks is not the same as the Prototypical Network
    """
    def __init__(self, data_path):
        super(Xrf55TrainEnhanced, self).__init__()
        self.datas, self.num_class, self.num_action = loadToMemEnhanced(data_path, "training")

    def __len__(self):
        """
        You can consider the length of train_set to be infinite,
        so you just need to fill in a very large integer here,
        at least larger than MAX_ITER in train.py
        """
        return sys.maxsize

    def __getitem__(self, index):
        """
        function：get data from the training set
        input：an index
             An odd index indicates selecting waveforms from the same type (positive sample)
             while an even index indicates selecting waveforms from different types (negative sample)
        output：1. two waveforms selected from the training set, as the input for the Siamese Networks
            2. a label, with the format of a [(1)] tensor,
               label=0.0 means the positive sample, label=1.0 means the negative sample
        """
        # positive sample
        if index % 2 == 1:
            label = 0.0
            idx1 = random.randint(0, self.num_class - 1)
            action = random.randint(0, self.num_action - 1)  # the a_{train}
            wave1, wave2 = random.sample(self.datas[idx1][action], 2)  # use random.sample to ensure they are different
        # negative sample
        else:
            label = 1.0
            idx1, idx2 = random.sample(range(0, self.num_class - 1), 2)
            action = random.randint(0, self.num_action - 1)  # the a_{train}
            wave1 = random.choice(self.datas[idx1][action])
            wave2 = random.choice(self.datas[idx2][action])

        return wave1, wave2, torch.from_numpy(np.array([label], dtype=np.float32))


class Xrf55Predict(Dataset):
    """
    the dataloader for test set
    """
    def __init__(self, data_path):
        super(Xrf55Predict, self).__init__()
        self.datas, self.num_class, self.num_action = loadToMemEnhanced(data_path, "test")
        self.candidate = None  # the query
        self.candidateClass = None  # the category of the query or u_{test}
        self.candidateAction = None  #  the action of the query or a_{test}
        self.candidateIndex = None  # the index of query in data[u_{test}][a_{test}][]


    def getCandidate(self):
        """
        select a query
        """
        self.candidateClass = random.randint(0, self.num_class - 1)
        self.candidateAction = random.randint(0, self.num_action - 1)

        self.candidateIndex, self.candidate = (
            random.choice(list(enumerate(self.datas[self.candidateClass][self.candidateAction]))))
        return self.candidate, self.candidateClass

    def getSupporter(self, num_per_class):
        """
        input:
            num_per_class: the n_{test}, or the shot
        output：
            supporters: a [num_per_class*|U_{test}|, 270, 1000] tensor, representing the test support set
            classes: 一a [num_per_class*|U_{test}|] list, representing the u of each sample in the support set, respectively
        """
        supporters = []
        classes = []
        for class_i in range(self.num_class):
            filtered_data = self.datas[class_i][self.candidateAction].copy()
            if class_i == self.candidateClass:  # the query and the test support set shall not override
                filtered_data.pop(self.candidateIndex)

            supporter = random.sample(filtered_data, min(num_per_class, len(filtered_data)))
            supporters.extend(supporter)
            classes.extend([class_i] * min(num_per_class, len(filtered_data)))

        return torch.stack(supporters), classes