import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import matplotlib.pyplot as plt

# 打开并读取原始图像中像素颜色值，转换为三维数组
imOrigin = Image.open('颜色压缩测试图像.jpg')
dataOrigin = np.array(imOrigin)
# 然后再转换为二维数组，-1表示自动计算该维度的大小
data = dataOrigin.reshape(-1,3)

# 使用KMeans算法把所有像素的颜色值划分为4类
kmeansPredicter = KMeans(n_clusters=4)
#kmeansPredicter = MiniBatchKMeans(n_clusters=4)
kmeansPredicter.fit(data)

# 使用每个像素所属类的中心值替换该像素的颜色
# temp中存放每个数据所属类的标签
temp = kmeansPredicter.labels_
dataNew = kmeansPredicter.cluster_centers_[temp]
dataNew.shape = dataOrigin.shape
dataNew = np.uint8(dataNew)
plt.imshow(dataNew)
plt.imsave('结果图像.jpg', dataNew)
plt.show()
