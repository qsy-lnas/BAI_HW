from os import name
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, average_precision_score


# 绘制混淆矩阵
def plot_confusion_matrix(m, title='Confusion Matrix', name=name):
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.clf()
    plt.imshow(m, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(('./result/%s_confusion_mat.png' % name), format='png')


# 绘制ROC曲线
def plot_roc_figure(target, score, name): 
    # 分类标签转吐热编码
    target = label_binarize(target, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    score = np.asarray(score)
    #print(target.shape, score.shape)
    n_class = target.shape[1]

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for idx in range(n_class):
        fpr[idx], tpr[idx], _ = roc_curve(target[:, idx], score[:, idx])
        roc_auc[idx] = auc(fpr[idx], tpr[idx])

    # 记录ROC曲线和auROC的值
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制各类别的roc曲线
    line_w=2
    plt.clf()
    plt.figure()

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'springgreen', 'lightcoral', 'violet', 'gold', 'lightpink', 'sandybrown', 'mediumturquoise']
    for idx, color in zip(range(n_class), colors):
        plt.plot(fpr[idx], tpr[idx], color=color, lw=line_w,
                label='class{0}:ROC curve (area = {1:0.3f})'
                ''.format(idx, roc_auc[idx]))

    plt.plot([0, 1], [0, 1], 'k--', lw=line_w)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - MNIST')
    plt.legend(loc="lower right")
    plt.savefig(('%s_roc.png' % name), format='png')


# 绘制PRC曲线
def plot_prc_figure(target, output, name): 
    # 分类标签转吐热编码
    target = label_binarize(target, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    output = np.asarray(output)
    n_class = target.shape[1]

    # 计算每一类的ROC
    prc = dict()
    rec = dict()
    ave_prc = dict()
    for idx in range(n_class):
        prc[idx], rec[idx], _ = precision_recall_curve(target[:, idx], output[:, idx])
        ave_prc[idx] = average_precision_score(target[:, idx], output[:, idx])

    # 记录PRC曲线和auPRC的值
    prc["micro"], rec["micro"], _ = precision_recall_curve(target.ravel(), output.ravel())
    ave_prc["micro"] = (prc["micro"], rec["micro"])

    # 绘制各类别的prc曲线
    line_w=2
    plt.clf()
    plt.figure()

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'springgreen', 'lightcoral', 'violet', 'gold', 'lightpink', 'sandybrown', 'mediumturquoise']
    for idx, color in zip(range(n_class), colors):
        plt.plot(rec[idx], prc[idx], color=color, lw=line_w,
                label='class{0}:AP (score = {1:0.3f})'
                ''.format(idx, ave_prc[idx]))

    plt.plot([0, 1], [0, 1], 'k--', lw=line_w)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('precision recall curve - MNIST')
    plt.legend(loc="lower right")
    plt.savefig(('%s_prc.png' % name), format='png')
