####多个混淆矩阵合并
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    inpath = r"E:\科研\草地植物识别\detect"
    f1 = r"E:\科研\草地植物识别\detect\EDS-01-01\EDS-01-01.xlsx"
    data = np.zeros((6,6))
    for root,subdir,files in os.walk(inpath):
        for f in files:
            if f.endswith(".xlsx"):
                fname = os.path.join(root,f)
                df = pd.read_excel(fname)
                da = df.iloc[6:,2:].values
                data = np.add(data,da)
                #print("." * 10 + fname + "." * 10)
                #print(da)
                #print(da.shape)
    outdf = pd.DataFrame(data,columns=["BaiYunXiang","JiLi","MengGuJiu","TuChunHua","XiYeJiu","background"],index=["BaiYunXiang","JiLi","MengGuJiu","TuChunHua","XiYeJiu","background"])
    with pd.ExcelWriter(os.path.join(inpath,"ALL.xlsx")) as writer:
        outdf.to_excel(writer)

