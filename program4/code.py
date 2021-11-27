import argparse
import os

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


'''Model of three questions'''
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.fc = nn.Linear(784, 10) # B, 784 -> B, 10

    def forward(self, x):
        x = self.fc(x)
        return x

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        x = self.fc(x)  # B, 784 -> B, 10
        return x

class ConvModel(nn.Module):

    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 ,64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linearfc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 28, 28).unsqueeze(1) # B, 784 -> B, 1, 28, 28
        x = self.conv1(x) # B, 32, 14, 14
        x = self.conv2(x) # B, 64, 7, 7
        x = x.view(B, -1)
        x = self.linearfc(x) #B, 64*7*7, 10
        return x

'''Dataset'''
class MNISTDataset(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        self.mnist = scio.loadmat('./mnist.mat')
        if self.train:
            self.data = self.mnist['train_X']  # 60000, 784
            self.label = self.mnist['train_Y'].reshape(self.data.shape[0])  # 60000
        else:
            self.data = self.mnist['test_X']  # 10000, 784
            self.label = self.mnist['test_Y'].reshape(self.data.shape[0])  # 10000

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]

        image = torch.from_numpy(image.astype(np.float32))
        label = int(label)
        return image, label

    def __len__(self):
        return self.data.shape[0]

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='',  help='model name')
parser.add_argument('--method', type=str, default='conv', choices=['softmax', 'linear', 'conv'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()
#print(args)

writer = SummaryWriter(log_dir=f'./events/{args.name}', comment=args.name)

train_dataset = MNISTDataset(train=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                         shuffle=True, drop_last=True)

test_dataset = MNISTDataset(train=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                        shuffle=False, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.method == 'softmax':
    net = SoftmaxModel().to(device)
elif args.method == 'linear':
    net = LinearModel().to(device)
elif args.method == 'conv':
    net = ConvModel().to(device)

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss().to(device)
net = net.to(device)
steps = 0
best_acc = 0

print("Training", args.name, "on", device)

for epoch in range(args.nepoch):

    '''begin training'''
    total_num = 0
    total_correct = 0
    total_train_loss = 0
    num_batch = len(trainloader)
    net.train()

    print('Training Epoch [%d/%d]' % (epoch, args.nepoch))

    for idx, (image, label) in enumerate(trainloader, 0):
        steps += 1
        image, label = Variable(image), Variable(label)
        image, label = image.cuda(device), label.cuda(device)

        optimizer.zero_grad()
        pred = net(image)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, pred = pred.max(1)
        total_num += label.size(0)
        total_correct += pred.eq(label).sum().item()

        if idx % 100 == 0:
            accuracy = 100. * total_correct / total_num
            print('Train[%d: %d/%d] loss: %.6f accuracy: %.6f' % (epoch, idx, num_batch, total_train_loss / (idx + 1), accuracy))
            writer.add_scalar('train/acc', accuracy, steps)
            writer.add_scalar('train/loss', loss.item(), steps)
            writer.flush()

    '''begin evaluating'''
    total_num = 0
    total_correct = 0
    total_test_loss = 0
    net.eval()

    print('Evaluating Epoch [%d/%d]' % (epoch, args.nepoch))

    for idx, (image, label) in enumerate(testloader, 0):
        with torch.no_grad():
            image, label = image.to(device), label.to(device)
            pred = net(image)

        _, pred = pred.max(1)
        total_num += label.size(0)
        total_correct += pred.eq(label).sum().item()

    test_acc = 100. * total_correct / total_num
    if test_acc > best_acc:
        best_acc = test_acc

    print('test accuracy: %.6f, best accuracy: %.6f' % (test_acc, best_acc))
    writer.add_scalar('test/acc', test_acc, epoch)
    writer.add_scalar('test/best_acc', best_acc, epoch)
    writer.flush()