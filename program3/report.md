<h2  align = "center" >人工智能基础第三次编程<br> 实验报告 </h2>

<h6 align = "center">自96 曲世远 2019011455</h6>

### 1.作业要求

本次编程作业要求使用手写数字图片中的白色像素数量作为每个数据点的特征值，进行Logistic回归计算，并利用梯度下降算法得到较优的模型，并对模型进行评价。

### 2.理论计算

*推导使用随机梯度下降法求解**一元Logistic回归**的过程：*



### 2.算法实现

**训练部分：**

```python
def train(w, b, X, Y, alpha=0.1, epochs=50, batchsize=32):
    """
    Input random parameters w, b and features:X labels:Y\\
    Set the parameters alpha as learning rate, epochs and batchsize\\
    return trained w, b and recording the train loss and accuracy

    """
    loss = np.zeros(epochs)
    acc = np.zeros(epochs)
    with tqdm(total = epochs) as t:
        for i in range(epochs):
            '''split the dataset'''
            random.seed(i)
            index = [i for i in range(X.shape[0])]
            random.shuffle(index)
            X = X[index]
            Y = Y[index]
            '''train'''
            acc_sum = 0
            loss_sum = 0
            for k in range(X.shape[0] // batchsize):
                X_ = X[k * batchsize : (k + 1) * batchsize]
                Y_ = Y[k * batchsize : (k + 1) * batchsize]
                Z = np.dot(w, X_) + b
                Z_ = Z > 0.5
                Y__ = Y_ == 1
                acc_sum += (np.sum(Z_ * Y__) + np.sum(~Z_ * ~Y__)) / batchsize
                H = np.power(np.e, Z) / (1 + np.power(np.e, Z))
                Deltab  = np.mean(H - Y_)
                Deltaw = np.mean(np.dot(H - Y_, X_))   
                loss_sum += np.mean(-Y_ * np.log(H) - (1 - Y_) * np.log(1 - H))             
                if k == X.shape[0] // batchsize - 1:
                    acc[i] = acc_sum / (k + 1)
                    loss[i] = loss_sum / (k + 1)
                    t.set_postfix(loss = loss_sum / (k + 1))
                    t.update(1)
                w = w - Deltaw * alpha
                b = b - Deltab * alpha
                
    return w, b, loss, acc
```

以上函数为训练函数，主要就是在batch内使用上一部分推导得到的梯度对参数$w, b$进行梯度下降运算，同时记录训练过程中的loss与accuracy。

```python
def test(w, b, X, Y, threshold):
    """
    Use trained parameters w, b with testfeature:X, test_labels:Y to evaluate the model
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
    # Balances error rate
    BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP))
    # Matthew's correlation coefficient
    MCC = (TP * TN - FP * FN) / np.power((TP + FP) * (FP + TN) * (TN + FN) * (FN + TP), 0.5)
    # Sensitivity
    Sensitivity = TP / (TP + FN)
    # Specificity
    Specificity = TN / (TN + FP)
    # Recall
    Recall = TP / (TP + FN)
    # Precision
    Precision = TP / (TP + FP)
    # F1-measure
    F1 = 2 * Precision * Recall / (Precision + Recall)
    # auROC
    auROC = metrics.roc_auc_score(Y, Z)
    # auPRC
    auPRC = metrics.average_precision_score(Y, Z)
```

以上代码为我的评价函数，主要是针对模型给出的预测结果与真实标签做对比并依据诸多参数给出评价。在$epochs = 100, alpha = 0.05, bathcsize = 16$时，同时与sklearn的效果进行对比如下：

| 评价指标 | My Model | sk-learn |
| :------: | :------: | :------: |
| Accuracy |          |          |
|          |          |          |
|          |          |          |
|          |          |          |
|          |          |          |
|          |          |          |
|          |          |          |
|          |          |          |
|          |          |          |
|          |          |          |

