#绘制Precision、Recall、mAP@0.5、F1-Score
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

if __name__ == '__main__':
    inpath = r"E:\科研\草地植物识别\Fig"
    mAP = pd.read_excel(os.path.join(inpath,"mAP@0.5.xlsx"))
    Precision = pd.read_excel(os.path.join(inpath,"precision.xlsx"))
    Recall = pd.read_excel(os.path.join(inpath,"Recall.xlsx"))
    F1_score = pd.read_excel(os.path.join(inpath,"F1-Score.xlsx"))

    mAP_array = mAP.values
    precision_array = Precision.values
    Recall_array = Recall.values
    F1_score_array = F1_score.values
    fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=False,
                            sharey=False,constrained_layout=True)
    axs[0,0].plot(mAP_array[:,0],mAP_array[:,1],label="YOLOv7")
    axs[0,0].plot(mAP_array[:,0],mAP_array[:,2],label="YOLOv7-X")
    axs[0,0].plot(mAP_array[:,0],mAP_array[:,3],label="YOLOv7-E6E")
    axs[0,0].grid(True)
    axs[0,0].set_title("mAP@0.5")
    axs[0,0].set_xlabel("Epochs")
    axs[0,0].set_ylabel("mAP@0.5")
    axs[0,0].legend(["YOLOv7","YOLOv7-X","YOLOv7-E6E"],loc="lower right",)
    axs[0,0].legend(shadow=True, fancybox=True)

    axs[0,1].plot(precision_array[:,0],precision_array[:,1],label="YOLOv7")
    axs[0,1].plot(precision_array[:,0],precision_array[:,2],label="YOLOv7-X")
    axs[0,1].plot(precision_array[:,0],precision_array[:,3],label="YOLOv7-E6E")
    axs[0,1].grid(True)
    axs[0,1].set_title("Precision")
    axs[0,1].set_xlabel("Epochs")
    axs[0,1].set_ylabel("Precision")
    axs[0,1].legend(["YOLOv7","YOLOv7-X","YOLOv7-E6E"],loc="lower right",)
    axs[0,1].legend(shadow=True, fancybox=True)

    axs[1,0].plot(Recall_array[:,0],Recall_array[:,1],label="YOLOv7")
    axs[1,0].plot(Recall_array[:,0],Recall_array[:,2],label="YOLOv7-X")
    axs[1,0].plot(Recall_array[:,0],Recall_array[:,3],label="YOLOv7-E6E")
    axs[1,0].grid(True)
    axs[1,0].set_title("Recall")
    axs[1,0].set_xlabel("Epochs")
    axs[1,0].set_ylabel("Recall")
    axs[1,0].legend(["YOLOv7","YOLOv7-X","YOLOv7-E6E"],loc="lower right",)
    axs[1,0].legend(shadow=True, fancybox=True)

    axs[1,1].plot(F1_score_array[:,0],F1_score_array[:,1],label="YOLOv7")
    axs[1,1].plot(F1_score_array[:,0],F1_score_array[:,2],label="YOLOv7-X")
    axs[1,1].plot(F1_score_array[:,0],F1_score_array[:,3],label="YOLOv7-E6E")
    axs[1,1].grid(True)
    axs[1,1].set_title("F1-score")
    axs[1,1].set_xlabel("Epochs")
    axs[1,1].set_ylabel("F1-score")
    axs[1,1].legend(["YOLOv7","YOLOv7-X","YOLOv7-E6E"],loc="lower right",)
    axs[1,1].legend(shadow=True, fancybox=True)

    plt.legend(loc="lower right")
    #fig.savefig('Metric.jpg', bbox_inches='tight')
    plt.savefig("Metric.jpg",dpi=300)
    plt.show()

'''
    ax1 = plt.subplot(221)
    ax1.plot(mAP_array[:,0],mAP_array[:,1],label="YOLOv7")
    ax1.plot(mAP_array[:,0],mAP_array[:,2],label="YOLOv7-X")
    ax1.plot(mAP_array[:,0],mAP_array[:,2],label="YOLOv7-E6E")
    ax1.grid(True)
    ax1.set_title("mAP@0.5")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("mAP@0.5 values")
    ax1.legend(loc="lower right")
    ax1.legend(shadow=True, fancybox=True)

    ax2 = plt.subplot(222)
    ax2.plot(precision_array[:,0],precision_array[:,1:])
    ax2.grid(True)
    ax2.set_title("Precision")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Precision")

    ax3 = plt.subplot(223)
    ax3.plot(Recall_array[:,0],Recall_array[:,1:])
    ax3.grid(True)
    ax3.set_title("Recall")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Recall")

    ax4 = plt.subplot(224)
    ax4.plot(F1_score_array[:,0],F1_score_array[:,1:])
    ax4.grid(True)
    ax4.set_title("F1-score")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("F1-score")
    plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,wspace=0.2,hspace=0.3)
    plt.show()
'''
