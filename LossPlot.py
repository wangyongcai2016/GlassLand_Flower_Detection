#绘制Precision、Recall、mAP@0.5、F1-Score
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

if __name__ == '__main__':
    inpath = r"E:\科研\草地植物识别\Fig"
    loss = pd.read_excel(os.path.join(inpath,"clcloss.xlsx"))
    loss_array = loss.values
    plt.plot(loss_array[:,0],loss_array[:,1],label="YOLOv7")
    plt.plot(loss_array[:,0],loss_array[:,2], label="YOLOv7-X")
    plt.plot(loss_array[:,0],loss_array[:,3],label="YOLOv7-E6E")
    plt.grid(True)
    plt.legend(["YOLOv7","YOLOv7-X","YOLOv7-E6E"],loc="lower right",)
    plt.legend(shadow=True, fancybox=True)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.show()
    plt.savefig("Loss.jpg",dpi=300)
