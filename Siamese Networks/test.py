"""
Test the Siamese Networks
Please note: Since the testing is random, you may not be able to reproduce results that are exactly the same as those in the paper,
but you should be able to achieve results that are fairly close.
"""
import os
import time
import random
import numpy as np
import torch

from dataset_loader import Xrf55Predict
from siamese_body import SiameseNetwork

TEST_PATH = "E:\\Xrf55\\CACP\\test"  # path to the test set
CHECKPOINT_PATH = "E:\\Python Project\\SimID\\models\\CACP"  # path to where the checkpoints are saved
# all the files end with '.pt' under the CHECKPOINT_PATH would be evaluated
PREDICT_TIME = 5000  # the rounds of test
HEAD = 'SENet'  # make sure that you choose the same feature encoder as the training process
SHOTS = 1  # the n_{test}, or the n-shot in test
CUDA = True
GPU_IDS = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
print("use gpu:", GPU_IDS, "to predict.")

predict_loader = Xrf55Predict(TEST_PATH)

# load the model
net = SiameseNetwork(head=HEAD)

if CUDA:
    device = torch.device("cuda:" + GPU_IDS)
else:
    device = "cpu"
net = net.to(device)

accuracies = []
# ready for test
for ck_point in os.listdir(CHECKPOINT_PATH):
    if ck_point.endswith(".pt"):
        print("Evaluating：", ck_point)

        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, ck_point))
        net.load_state_dict(checkpoint['net_state_dict'])
        # record the results
        correct = 0
        error = 0
        predicting_times = []

        for predict_id in range(1, PREDICT_TIME):
            time_start = time.time()
            # select the test support set and the query
            candidate, candidate_class = predict_loader.getCandidate()
            supporters, classes = predict_loader.getSupporter(SHOTS)
            if CUDA:
                candidate = candidate.cuda()
                supporters = supporters.cuda()
            # copy the query to a tensor with the same size as the test support set
            candidate = torch.stack([candidate] * len(classes), dim=0)

            output = net.forward(supporters, candidate).data.cpu().numpy()
            output = np.squeeze(output)  # the last dimension is not needed

            # get the average similarity score of each user, and the argmax would be the predicting result
            score = {}
            count = {}
            for i in range(len(output)):
                if not score.get(classes[i]):
                    score[classes[i]] = output[i]
                    count[classes[i]] = 1
                else:
                    score[classes[i]] += output[i]
                    count[classes[i]] += 1
            predict_res = -1
            predict_score = 100000000
            for class_i in score.keys():
                avg_score = score[class_i] / count[class_i]
                if avg_score < predict_score:
                    predict_score = avg_score
                    predict_res = class_i

            time_elapsed = time.time() - time_start
            predicting_times.append(time_elapsed)

            # print(f"[{predict_id}]  predicting result：{predict_res}  ground truth：{candidate_class}  time lapsed：{time_elapsed}", end='')
            error_flag = False
            if predict_res == candidate_class:
                # print("   correct")
                correct += 1
            else:
                # print("  wrong")
                error_flag = True
                error += 1

        accuracy = correct / (correct + error)
        print("evaluation over, accuracy:", accuracy)
        print("average time:", np.mean(predicting_times))
        accuracies.append(accuracy)

print("average over all checkpoints:", np.mean(accuracies))
print("the best checkpoint:", np.max(accuracies))