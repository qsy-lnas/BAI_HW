import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
import scipy.io as scio
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from scipy.linalg import expm
from tqdm import tqdm

def plot_acc_loss(loss):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    #par1 = host.twinx()   # 共享x轴
 
    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("test-loss")
    #par1.set_ylabel("test-accuracy")
 
    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    #p2, = par1.plot(range(len(acc)), acc, label="accuracy")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    #par1.axis["right"].label.set_color(p2.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()

def shuffle(X, Y, seed=2019011455):
    """
    随机打乱原始数据
    """
    random.seed(seed)
    index = [i for i in range(X.shape[0])]
    random.shuffle(index)
    return X[index], Y[index]

def get_zero_and_one(X, Y):
    """
    从给定数据中抽取数字0和数字1的样本
    """
    index_1 = Y==0
    index_8 = Y==1
    index = index_1 + index_8
    return X[index], Y[index]
    
def load_data(data_dir="./", data_file="mnist.mat"):
    # 加载数据，划分数据集
    data = scio.loadmat(os.path.join(data_dir, data_file))
    train_X, test_X = data['train_X'], data['test_X']
    train_Y, test_Y = data['train_Y'].reshape(train_X.shape[0]), data['test_Y'].reshape(test_X.shape[0])
    
    # 从训练数据中抽取数字 0 和数字 1 的样本，并打乱
    train_X, train_Y = get_zero_and_one(train_X, train_Y)
    train_X, train_Y = shuffle(train_X, train_Y)
    train_Y = (train_Y==1).astype(np.float32)  # 1->True 0->false
    # 从测试数据中抽取数字 0 和数字 1 的样本，并打乱
    test_X, test_Y = get_zero_and_one(test_X, test_Y)
    test_X, test_Y = shuffle(test_X, test_Y)
    test_Y = (test_Y==1).astype(np.float32)
    print("原始图片共有%d张，其中数字1的图片有%d张。" % (test_X.shape[0], sum(test_Y==1)))
    return train_X, train_Y, test_X, test_Y
        
def ext_feature(train_X, test_X, threshold = 200):
    """
    抽取图像的白色像素点比例作为特征
    """
    train_feature = np.sum(train_X > threshold, axis=1)/784
    test_feature = np.sum(test_X > threshold, axis=1)/784
    return train_feature, test_feature

def train(w, b, X, Y, alpha=0.1, epochs=50, batchsize=32):
    """
    YOUR CODE HERE
    """
    loss = np.zeros(epochs)
    with tqdm(total = epochs) as t:
        for i in range(epochs):
            '''split the dataset'''
            random.seed(i)
            index = [i for i in range(X.shape[0])]
            random.shuffle(index)
            X = X[index]
            Y = Y[index]
            '''train'''
            loss_sum = 0
            for k in range(X.shape[0] // batchsize):
                X_ = X[k * batchsize : (k + 1) * batchsize]
                Y_ = Y[k * batchsize : (k + 1) * batchsize]
                Z = np.dot(w, X_) + b
                H = np.power(np.e, Z) / (1 + np.power(np.e, Z))
                Deltab  = np.mean(H - Y_)
                Deltaw = np.mean(np.dot(H - Y_, X_))   
                loss_sum += np.mean(-Y_ * np.log(H) - (1 - Y_) * np.log(1 - H))             
                if k == X.shape[0] // batchsize - 1:
                    
                    loss[i] = loss_sum / (k + 1)
                    t.set_postfix(loss = loss_sum / (k + 1))
                    t.update(1)
                w = w - Deltaw * alpha
                b = b - Deltab * alpha
                
    return w, b, loss

def test(w, b, X, Y, threshold):
    """
    YOUR CODE HERE
    """
    print("w = %.4f, b = %.4f" % (w, b))
    Z = np.dot(w, X) + b
    index_z_1 = (Z > threshold)
    index_z_0 = (Z < threshold)
    index_y_1 = (Y == 1)
    index_y_0 = (Y == 0)
    TP = np.sum(index_z_1 * index_y_1) / Z.shape[0]
    FP = np.sum(index_z_1 * index_y_0) / Z.shape[0]
    FN = np.sum(index_z_0 * index_y_1) / Z.shape[0]
    TN = np.sum(index_z_0 * index_y_0) / Z.shape[0]
    print(TP, FP, FN, TN)
    # Accuracy
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("ACC = %.4f" % ACC)
    # Balances error rate
    BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP))
    print("BER = %.4f" % BER)
    # Matthew's correlation coefficient
    MCC = (TP * TN - FP * FN) / np.power((TP + FP) * (FP + TN) * (TN + FN) * (FN + TP), 0.5)
    print("MCC = %.4f" % MCC)
    # Sensitivity
    Sensitivity = TP / (TP + FN)
    print("Sensitivity = %.4f" % Sensitivity)
    # Specificity
    Specificity = TN / (TN + FP)
    print("Specificity = %.4f" % Specificity)
    # Recall
    Recall = TP / (TP + FN)
    print("Recall = %.4f" % Recall)
    # Precision
    Precision = TP / (TP + FP)
    print("Precision = %.4f" % Precision)
    # F1-measure
    F1 = 2 * Precision * Recall / (Precision + Recall)
    print("F1 = %.4f" % F1)
    # auROC
    auROC = metrics.roc_auc_score(Y, Z)
    print("auROC = %.4f" % auROC)
    # auPRC
    auPRC = metrics.average_precision_score(Y, Z)
    print("auPRC = %.4f" % auPRC)

def train_sklearn(X, Y, X_test, epochs):
    #print(X.shape, Y.shape, X_test)
    skmodel = LogisticRegression(penalty = 'none', max_iter = epochs)
    skmodel.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    Z = skmodel.predict(X_test.reshape(-1, 1))
    return Z

def test_sklearn(Z, Y, threshold):
    print("---------result for sklearn-----------")
    index_z_1 = (Z > threshold)
    index_z_0 = (Z < threshold)
    index_y_1 = (Y == 1)
    index_y_0 = (Y == 0)
    TP = np.sum(index_z_1 * index_y_1) / Z.shape[0]
    FP = np.sum(index_z_1 * index_y_0) / Z.shape[0]
    FN = np.sum(index_z_0 * index_y_1) / Z.shape[0]
    TN = np.sum(index_z_0 * index_y_0) / Z.shape[0]
    print(TP, FP, FN, TN)
    # Accuracy
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("ACC = %.4f" % ACC)
    # Balances error rate
    BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP))
    print("BER = %.4f" % BER)
    # Matthew's correlation coefficient
    MCC = (TP * TN - FP * FN) / np.power((TP + FP) * (FP + TN) * (TN + FN) * (FN + TP), 0.5)
    print("MCC = %.4f" % MCC)
    # Sensitivity
    Sensitivity = TP / (TP + FN)
    print("Sensitivity = %.4f" % Sensitivity)
    # Specificity
    Specificity = TN / (TN + FP)
    print("Specificity = %.4f" % Specificity)
    # Recall
    Recall = TP / (TP + FN)
    print("Recall = %.4f" % Recall)
    # Precision
    Precision = TP / (TP + FP)
    print("Precision = %.4f" % Precision)
    # F1-measure
    F1 = 2 * Precision * Recall / (Precision + Recall)
    print("F1 = %.4f" % F1)
    # auROC
    auROC = metrics.roc_auc_score(Y, Z)
    print("auROC = %.4f" % auROC)
    # auPRC
    auPRC = metrics.average_precision_score(Y, Z)
    print("auPRC = %.4f" % auPRC)

if __name__ == "__main__":
    seed = 2019011455
    #指定种子
    np.random.seed(seed)
    # 加载数据
    train_X, train_Y, test_X, test_Y = load_data(data_dir="program3/")
    # 抽取特征
    train_feature, test_feature = ext_feature(train_X, test_X)
    
    # 初始化参数
    w = np.random.randn()
    b = np.random.randn()
    #print(w, b)
    alpha = 0.05
    epochs = 100
    batchsize = 16
    test_threshold = 0.5
    # train
    w, b, loss = train(w, b, train_feature, train_Y, alpha, epochs, batchsize)
    plot_acc_loss(loss)
    Z = train_sklearn(train_feature, train_Y, test_feature, epochs)
    #print(Z.shape)
    #test
    test(w, b, test_feature, test_Y, test_threshold)
    test_sklearn(Z, test_Y, test_threshold)
    