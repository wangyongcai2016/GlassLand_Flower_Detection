#绘制混淆矩阵
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签


def plot_confusion_matrix(cm, classes,savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 9), dpi=300)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes)+1)
    x, y = np.meshgrid(ind_array, ind_array)#生成坐标矩阵
    diags = np.diag(cm)#对角TP值
    TP_FNs, TP_FPs = [], []
    for x_val, y_val in zip(x.flatten(), y.flatten()):#并行遍历
        max_index = len(classes)
        if x_val != max_index and y_val != max_index:#绘制混淆矩阵各格数值
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, c, color='black', fontsize=15, va='center', ha='center')
        elif x_val == max_index and y_val != max_index:#绘制最右列即各数据类别的查全率
            TP = diags[y_val]
            TP_FN = cm.sum(axis=1)[y_val]
            recall = TP / (TP_FN)
            if recall != 0.0 and recall > 0.01:
                recall = str('%.2f'%(recall*100,))+'%'
            elif recall == 0.0:
                recall = '0'
            TP_FNs.append(TP_FN)
            plt.text(x_val, y_val, str(TP_FN)+'\n'+str(recall)+'%', color='black', va='center', ha='center')
        elif x_val != max_index and y_val == max_index:#绘制最下行即各数据类别的查准率
            TP = diags[x_val]
            TP_FP = cm.sum(axis=0)[x_val]
            precision = TP / (TP_FP)
            if precision != 0.0 and precision > 0.01:
                precision = str('%.2f'%(precision*100,))+'%'
            elif precision == 0.0:
                precision = '0'
            TP_FPs.append(TP_FP)
            plt.text(x_val, y_val, str(TP_FP)+'\n'+str(precision), color='black', va='center', ha='center')
    cm = np.insert(cm,max_index,TP_FNs,1)
    cm = np.insert(cm,max_index,np.append(TP_FPs,SUM),0)
    plt.text(max_index, max_index, str(SUM)+'\n'+str('%.2f'%(PRECISION*100,))+'%', color='red', va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    #plt.gcf().subplots_adjust(bottom=0.15)
    # show confusion matrix
    plt.savefig(savename, format='JPG',dpi=300)
    plt.show()

if __name__ == '__main__':
    metric = pd.read_excel(r"E:\科研\草地植物识别\detect\ALL.xlsx",sheet_name=1)
    #classes = metric.columns.tolist()[1:]
    classes = [u"北芸香",u"蒺藜",u"蒙古韭",u"兔唇花",u"细叶韭",u"背景"]
    matrix = metric.iloc[:,1:].values
    FP = sum(matrix.sum(axis=0)) - sum(np.diag(matrix)) #假正样本数
    FN = sum(matrix.sum(axis=1)) - sum(np.diag(matrix)) #假负样本数
    TP = sum(np.diag(matrix)) #真正样本数
    TN = sum(matrix.sum().flatten()) - (FP + FN + TP) #真负样本数
    SUM = TP+FP
    PRECISION = TP / (TP+FP)  # 查准率，又名准确率
    RECALL = TP / (TP+FN)  # 查全率，又名召回率
    plot_confusion_matrix(matrix,classes,"YOLOv7-E6E_ConfusionMatrix.jpg")


