"""
Siamese Networks + different feature encoders
"""
import os
import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from siamese_body import SiameseNetwork
from torch.utils.data import DataLoader
from dataset_loader import Xrf55TrainEnhanced

random.seed(42)

# Addresses
TRAIN_PATH = "E:\\Xrf55\\CACP\\train"  # path to the training set
CHECKPOINT_PATH = "E:\\Python Project\\SimID\\models\\CACP"  # the place to save the checkpoint
# When a checkpoint named "model-checkpoint-last.pt" exists under CHECKPOINT_PATH, the model will continue training from this checkpoint.
FIGURE_PATH = "E:\\Python Project\\SimID\\models\\CACP"  # the place to save the training curve figure

# Network architecture
HEAD = 'SENet'  # feature encoder, you can choose ResNet/DenseNet/SENet/SENet18

# Hyperparameters
BATCH_SIZE = 128
LR = 0.1
MAX_ITER = 15000  # max iteration for training

# Configs
SHOW_EVERY = 1  # how often (in terms of training epochs) the training loss and elapsed time are displayed during training
DRAW_EVERY = 100  # how often (in terms of training epochs) the training curve is sampled
SAVE_EVERY = 100  # how often (in terms of training epochs) the checkpoint 'model-checkpoint-last.pt' is saved
BACKUP_EVERY = 1000  # how often (in terms of training epochs) the checkpoint 'model-checkpoint-iterxxx.pt' is saved

# GPU
CUDA = True
GPU_IDS = "0"  # warning: it is not guaranteed that the model can run in a multi-GPU environment

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
print("use gpu:", GPU_IDS, "to train.")

# load the training set
train_set = Xrf55TrainEnhanced(TRAIN_PATH)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# you shall not set the parameter "shuffle" to True

# initiate the model and loss function
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
net = SiameseNetwork(head=HEAD)

# move the model to the device
device = torch.device("cuda:" + GPU_IDS if torch.cuda.is_available() else "cpu")
net = net.to(device)

# define the optimizer
optimizer = torch.optim.Adam(list(net.parameters()), lr=LR)
optimizer.zero_grad()

# variables used to record the training process
train_loss = []
train_loss_x = []
loss_val = 0
last_iter = 1

# load the checkpoint
if os.path.exists(os.path.join(CHECKPOINT_PATH, "model-checkpoint-last.pt")):
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "model-checkpoint-last.pt"))
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    LR = checkpoint['lr']
    train_loss = checkpoint['train_loss']
    train_loss_x = checkpoint['train_loss_x']
    last_iter = checkpoint['last_iter']

# ready for training
net.train()

training_times = []
time_start = time.time()


def save_iter(iter_id, checkpoint_name):
    checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
    torch.save({
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_loss_x': train_loss_x,
        'lr': LR,
        'last_iter': iter_id
    }, checkpoint_path)


def draw_iter():
    # paint the training curve figure
    plt.figure(figsize=(10, 5))

    plt.plot(train_loss_x, train_loss, label="Train Loss", color='blue', linewidth=1)

    plt.title("Training Loss and Precision Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, 'TrainingCurve.png'))


for iter_id, (wave1, wave2, label) in enumerate(train_loader, last_iter + 1):
    if iter_id > MAX_ITER:
        break
    if CUDA:
        wave1, wave2, label = wave1.cuda(), wave2.cuda(), label.cuda()

    # forward and back forward
    optimizer.zero_grad()
    output_train = net.forward(wave1, wave2)
    loss = loss_fn(output_train, label)
    loss_val += loss.item()
    loss.backward()
    optimizer.step()

    # loss count
    if iter_id % SHOW_EVERY == 0:
        print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            iter_id, loss_val / SHOW_EVERY, time.time() - time_start))
        training_times.append(time.time() - time_start)
        train_loss.append(loss_val / SHOW_EVERY)
        train_loss_x.append(iter_id)
        loss_val = 0

    if iter_id % DRAW_EVERY == 0:
        draw_iter()

    if iter_id % BACKUP_EVERY == 0:
        save_iter(iter_id, f"model-checkpoint-iter{iter_id}.pt")

    if iter_id % SAVE_EVERY == 0:
        save_iter(iter_id, "model-checkpoint-last.pt")

    plt.close('all')

    time_start = time.time()

print("average time: ", np.mean(training_times))