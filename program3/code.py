import scipy.io as scio
import random
import numpy as np
import os
from tqdm import tqdm

from scipy.linalg import expm


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
    with tqdm(total = epochs) as t:
        for i in range(epochs):
            '''split the dataset'''
            random.seed(i)
            index = [i for i in range(X.shape[0])]
            random.shuffle(index)
            X = X[index]
            Y = Y[index]
            '''train'''
            for k in range(X.shape[0] // batchsize):
                X_ = X[k * batchsize : (k + 1) * batchsize]
                Y_ = Y[k * batchsize : (k + 1) * batchsize]
                Z = np.dot(w, X_) + b
                H = np.power(np.e, Z) / (1 + np.power(np.e, Z))
                Deltab  = np.mean(H - Y_)
                Deltaw = np.mean(np.dot(H - Y_, X_))
                if k == X.shape[0] // batchsize - 1:
                    loss = np.mean(-Y_ * np.log(H) - (1 - Y_) * np.log(1 - H))
                    t.set_postfix(loss = loss)
                    t.update(1)
                w = w - Deltaw * alpha
                b = b - Deltab * alpha
                
    return w, b

def test(w, b, X, Y):
    """
    YOUR CODE HERE
    """
    Z = np.dot(w, X) + b
    index_z_1 = (Z == 1)
    index_z_0 = (Z == 0)
    index_y_1 = (Y == 1)
    index_y_0 = (Y == 0)
    TP = np.sum(index_z_1 * index_y_1)
    FP = np.sum(index_z_1 * index_y_0)
    FN = np.sum(index_z_0 * index_y_1)
    TN = np.sum(index_z_0 * index_y_0)
    # Accuracy
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("ACC = %.4f"%ACC)
    # Balances error rate
    BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP))
    print("BER = %.4f"%BER)
    # Matthew's correlation coefficient
    MCC = (TP * TN - FP * FN) / (\
        (TP + FP) * (FP + TN) * (TN + FN) * (FN + TP)) ** 0.5
    print("MCC = %.4f"%MCC)
    # 



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
    alpha = 0.1
    epochs = 50
    batchsize = 32
    
    # train
    w, b = train(w, b, train_feature, train_Y, alpha, epochs, batchsize)

    #test
    test(w, b, test_feature, test_Y)

    