"""
Test the Prototypical Network
Please note: Since the testing is random, you may not be able to reproduce results that are exactly the same as those in the paper,
but you should be able to achieve results that are fairly close.
"""
import os
import time
import torch
import numpy as np
from sklearn.manifold import TSNE
from dataset import Xrf55Predict
from torch.utils.data import DataLoader
from prototypical_body import PrototypicalNetwork

# 路径
TEST_PATH = "E:\\Xrf55\\CACP\\test"  # path to the test set
CHECKPOINT_PATH = "E:\\Python Project\\SimID\\models\\test"  # path to where the checkpoints are saved
# all the files end with '.pt' under the CHECKPOINT_PATH would be evaluated
PREDICT_TIME = 5000  # the rounds of test
HEAD = "SENet"  # make sure that you choose the same feature encoder as the training process
DIS_F = "Sim"  # The similarity computation method, you can choose Sim/L2
SHOT = 1  # the n_{test}, or the n-shot in test
CUDA = True
GPU_IDS = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
print("use device:", GPU_IDS, "to test.")

predict_set = Xrf55Predict(TEST_PATH, SHOT, 1, "test")
predict_loader = DataLoader(predict_set, batch_size=1, num_workers=0)

net = PrototypicalNetwork(dis_f=DIS_F, head=HEAD)

if CUDA:
    device = torch.device("cuda:" + GPU_IDS)
else:
    device = "cpu"

net = net.to(device)

ts = TSNE(n_components=2, perplexity=3)

accuracies = []
for ck_point in os.listdir(CHECKPOINT_PATH):
    if ck_point.endswith(".pt"):
        print("testing：", ck_point)

        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, ck_point))
        net.load_state_dict(checkpoint['net_state_dict'])

        right, error = 0, 0

        predict_times = []

        for action in range(PREDICT_TIME):
            for _, (waves, labels) in enumerate(predict_loader):
                support = predict_set.getSupport()

                if CUDA:
                    waves, labels, support = waves.cuda(), labels.cuda(), support.cuda()

                time_start = time.time()
                features = torch.cat((support, waves), dim=0)
                output = net.forward(features, predict_set.num_class, predict_set.shot).data.cpu().numpy()

                pred = np.argmax(output[0])

                time_elapsed = time.time() - time_start
                predict_times.append(time_elapsed)

                answer = labels[0].item()

                # print(f"ground truth：{answer} predicted answer：{pred} time lapsed：{time_elapsed}", end=' ')
                if pred == answer:
                    right += 1
                    # print("  correct")
                else:
                    error += 1
                    # print("wrong")

        precision = right / (right + error)
        accuracies.append(precision)
        print("evaluation over, accuracy: ", precision)
        print("average time: ", np.mean(predict_times))
        print("standard deviation", np.std(predict_times))

print("average over all checkpoints:", np.mean(accuracies))
print("the best checkpoint:", np.max(accuracies))
