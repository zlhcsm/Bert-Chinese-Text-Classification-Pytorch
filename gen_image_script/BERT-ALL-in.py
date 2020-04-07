import numpy as np
import matplotlib.pyplot as plt
#coding:utf-8
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['savefig.dpi'] = 300 # 图片像素
plt.rcParams['figure.dpi'] = 300 # 分辨率

#有中文出现的情况，需要u'内容'

classes = {"孕产次", "孕周", "胎位", "高危因素", "结果", "阿氏评分"}

# 设置图片的长宽比
plt.figure(figsize=(8, 4))

# 所有数据汇总
TotalCNN = [0.8464, 0.8833, 0.8970]
TotalDNN = [0.7913, 0.8560, 0.8841]
TotalBert = [0.9108, 0.9299, 0.9442]
x4 = np.arange(0, 3, 1)
x = np.arange(0, 6, 1)

# 一个窗口，多个图，多条数据
# 将窗口分成2行1列，在第1个作图，并设置背景色
sub4 = plt.subplot(111)

# 设置名称
sub4.set_title("三组实验模型整体效果对比")

# 设置网格线
sub4.grid(axis="y")

# 讲坐标轴设置为不可见
sub4.spines['right'].set_color('none')
sub4.spines['top'].set_color('none')
sub4.spines['left'].set_color('none')

# 设置y坐标轴的范围
sub4.set_ylim((0.6, 1))

# 设置y坐标的标签
sub4.set_ylabel("TotalF1-score")

# 设置x坐标轴的label
plt.xticks(x4, ['6000', '18000', '30000'])

# 往子图里边添加线条
sub4.plot(x4, TotalCNN,marker='o', label="TextCNN", linestyle=':')
sub4.plot(x4, TotalDNN, marker='o',label="TextRCNN", linestyle='-.')
sub4.plot(x4, TotalBert, marker='o',label="Bert-CNN")

# 设置图例
sub4.legend(loc='lower right',shadow=False, fontsize=10)

# 图像展示
plt.show()
