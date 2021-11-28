import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
from utils import plot_prc_figure , plot_roc_figure
import copy


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

def plot_acc_loss(train_acc, test_acc, args):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()   # 共享x轴
 
    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("train-acc")
    par1.set_ylabel("test-acc")
 
    # plot curves
    p1, = host.plot(range(len(train_acc)), train_acc, label="train_acc")
    p2, = par1.plot(range(len(test_acc)), test_acc, label="test_acc")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.savefig("%s.png"%args.name)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='',  help='model name')
parser.add_argument('--method', type=str, default='conv', choices=['softmax', 'linear', 'conv'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()
#print("%s.png"%args.name)

#writer = SummaryWriter(log_dir=f'./events/{args.name}', comment=args.name)

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
train_acc_for_plt = np.zeros(args.nepoch)
test_acc_for_plt = np.zeros(args.nepoch)

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
            """ print('Train[%d: %d/%d] loss: %.6f accuracy: %.6f' % (epoch, idx, num_batch, total_train_loss / (idx + 1), accuracy)) """
            train_acc_for_plt[epoch] = accuracy

    '''begin evaluating'''
    total_num = 0
    total_correct = 0
    total_test_loss = 0
    net.eval()

    
    """     print('Evaluating Epoch [%d/%d]' % (epoch, args.nepoch)) """

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

    """ print('test accuracy: %.6f, best accuracy: %.6f' % (test_acc, best_acc)) """
    test_acc_for_plt[epoch] = test_acc

'''save acc plot'''
plot_acc_loss(train_acc_for_plt, test_acc_for_plt, args)

'''Evaluate the model'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
net.eval()
preds = []
trues = []
outs = []
with torch.no_grad(): # 测试集不更新梯度
    for i, data in enumerate(testloader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        origin_outputs = copy.deepcopy(outputs)
        #print(origin_outputs)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels)
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        trues.extend(labels.detach().cpu().numpy())
        outs.extend(origin_outputs.detach().cpu().numpy())
#print(outs[0].shape, len(outs))
# Accuracy
for i in range(10):
    print('Accuracy of %d: %6f%%' %(i, 100 * class_correct[i] / class_total[i]))
# Precision
print('precision_micro: %.6f' %precision_score(trues, preds, average = 'micro'))
print('precision_macro: %.6f' %precision_score(trues, preds, average = 'macro'))
# Recall
print('recall_micro: %.6f' %recall_score(trues, preds, average='micro'))
print('recall_macro: %.6f' %recall_score(trues, preds, average='macro'))
# F1-measure
print('F1_micro: %.6f' %f1_score(trues, preds, average='micro'))
print('F1_macro: %.6f' %f1_score(trues, preds, average='macro'))
# Matthew's correlation coefficient
print('MCC:%.6f' % matthews_corrcoef(trues, preds))

print(classification_report(trues, preds))


# auROC
plot_roc_figure(trues, outs, args.name)
# auPRC
plot_prc_figure(trues, outs, args.name)