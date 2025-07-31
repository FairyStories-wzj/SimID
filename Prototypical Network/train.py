"""
Prototypical Network + different feature encoders
"""
import os
import time
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Xrf55TrainEnhanced
from prototypical_body import PrototypicalNetwork


# Addresses
TRAIN_PATH = "E:\\Xrf55\\CACP\\train"  # path to the training set
CHECKPOINT_PATH = "E:\\Python Project\\SimID\\models\\test"  # the place to save the checkpoint
# When a checkpoint named "model-checkpoint-last.pt" exists under CHECKPOINT_PATH, the model will continue training from this checkpoint.
FIGURE_PATH = "E:\\Python Project\\SimID\\models\\test"  # the place to save the training curve figure

# Network architecture
DIS_F = 'Sim'  # The similarity computation method, you can choose Sim/L2
HEAD = 'SENet'  # The feature encoder, you can choose ResNet/DenseNet/SENet/SENet18

# Hyperparameters
SHOT = 4  # The hyperparameter "n_train", which is the number of samples each category in the training support set
LR = 0.1  # The initial learning rate
LR_DECAY_ENABLED = False  # Turn on this if you want the learning rate to decay
LR_DECAY = 0.8  # The factor that the learning rate decays exponentially
LR_DECAY_EVERY = 100  # How often (in terms of training epochs) the learning rate decays
LR_LIMIT = 1e-6  # The lower bound of the learning rate
BATCH_SIZE = 128
MAX_ITER = 20000  # max iteration for training

# Configs
SHOW_EVERY = 1  # how often (in terms of training epochs) the training loss and elapsed time are displayed during training
DRAW_EVERY = 10  # how often (in terms of training epochs) the training curve is sampled
SAVE_EVERY = 100  # how often (in terms of training epochs) the checkpoint 'model-checkpoint-last.pt' is saved
BACKUP_EVERY = 1000  # how often (in terms of training epochs) the checkpoint 'model-checkpoint-iterxxx.pt' is saved

# GPU
CUDA = True
GPU_IDS = "0"  # warning: it is not guaranteed that the model can run in a multi-GPU environment

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
print("use gpu:", GPU_IDS, "to train.")

# load the traning set
train_set = Xrf55TrainEnhanced(TRAIN_PATH, SHOT, BATCH_SIZE, "training")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # you shall not set the parameter "shuffle" to True

# initiate the model and loss function
loss_fn = torch.nn.NLLLoss()
net = PrototypicalNetwork(dis_f=DIS_F, head=HEAD)

# move the model to the device
device = torch.device("cuda:" + GPU_IDS if torch.cuda.is_available() else "cpu")
loss_fn = loss_fn.to(device)
net = net.to(device)

# define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
optimizer.zero_grad()

# variables used to record the training process
loss = []
loss_x = []
loss_val = 0
last_iter = 0

# load the checkpoint
if os.path.exists(os.path.join(CHECKPOINT_PATH, "model-last-checkpoint.pt")):
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "model-last-checkpoint.pt"), weights_only=True)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    LR = checkpoint['lr']
    loss = checkpoint['loss']
    loss_x = checkpoint['loss_x']
    last_iter = checkpoint['last_iter']


def draw_iter():
    # paint the training curve figure
    plt.figure(figsize=(10, 5))

    plt.plot(loss_x, loss, label="Train Loss", color='blue', linewidth=1)

    plt.title("Training Loss and Precision Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, "TrainingCurve.png"))


def save_iter(iter_id, checkpoint_name):
    torch.save({
        'iter_id': iter_id,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'loss_x': loss_x,
        'lr': LR,
        'last_iter': iter_id
    }, os.path.join(CHECKPOINT_PATH, checkpoint_name))


net.train()
time_start = time.time()
training_times = []

for iter_id, (waves, labels) in enumerate(train_loader, last_iter + 1):
    if iter_id > MAX_ITER:
        break

    support = train_set.getSupport()
    if CUDA:
        waves, labels, support = waves.cuda(), labels.cuda(), support.cuda()

    features = torch.cat((support, waves), dim=0)
    output = net.forward(features, train_set.num_class, train_set.shot)

    iter_loss = loss_fn(output, labels)
    loss_val += iter_loss.item()

    iter_loss.backward()

    optimizer.step()

    time_elapsed = time.time() - time_start
    training_times.append(time_elapsed)

    if iter_id % SHOW_EVERY == 0:
        print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            iter_id, loss_val / SHOW_EVERY, time_elapsed))
        loss.append(loss_val / SHOW_EVERY)
        loss_x.append(iter_id)
        loss_val = 0

    if iter_id % DRAW_EVERY == 0:
        draw_iter()

    if iter_id % SAVE_EVERY == 0:
        save_iter(iter_id, "model-last-checkpoint.pt")

    if iter_id % BACKUP_EVERY == 0:
        save_iter(iter_id, f"model-checkpoint-iter{iter_id}.pt")

    # Modify the learning rate
    if LR_DECAY_ENABLED and iter_id % LR_DECAY_EVERY == 0 and LR > LR_LIMIT:
        LR = LR * LR_DECAY
        print("The LR now:", LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR

    plt.close('all')

    time_start = time.time()

print("average time：", np.mean(training_times))


# D:\ANACONDA\envs\Prototypical-Network\Lib\site-packages\torch\lib