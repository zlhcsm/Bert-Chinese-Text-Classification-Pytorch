import numpy as np
import matplotlib.pyplot as plt
#coding:utf-8
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

classes = {"孕产次", "孕周", "胎位", "高危因素", "结果", "阿氏评分"}

# 使用CNN网络跑出来的结果
CNN1000 = [0.7940, 0.8832, 0.7586, 0.9216, 0.8235, 0.8976]
CNN3000 = [0.8632, 0.9052, 0.8284, 0.9428, 0.8606, 0.8972]
CNN5000 = [0.8852, 0.9118, 0.8433, 0.9499, 0.8813, 0.9106]

# 使用DCNN网络跑出来的结果
DNN1000 = [0.7633, 0.8750, 0.6816, 0.7932, 0.7407, 0.8942]
DNN3000 = [0.8374, 0.8867, 0.7845, 0.9362, 0.9362, 0.8809]
DNN5000 = [0.8772, 0.8948, 0.8226, 0.9452, 0.8502, 0.9145]

# 使用BERT网络跑出来的结果
BERT1000 = [0.9036, 0.9394, 0.8708, 0.9548, 0.8705, 0.9259]
BERT3000 = [0.9260, 0.9260, 0.8825, 0.9712, 0.9192, 0.9370]
BERT5000 = [0.9363, 0.9529, 0.9529, 0.9717, 0.9405, 0.9520]

# 设置图片的长宽比
plt.figure(figsize=(10, 6))

# 所有数据汇总
TotalCNN = [0.8464, 0.8833, 0.8970]
TotalDNN = [0.7913, 0.8560, 0.8841]
TotalBert = [0.9108, 0.9299, 0.9442]
x4 = np.arange(0, 3, 1)
x = np.arange(0, 6, 1)

# 一个窗口，多个图，多条数据
# 将窗口分成2行1列，在第1个作图，并设置背景色
sub1 = plt.subplot(221)
sub2 = plt.subplot(222)   # 将窗口分成2行1列，在第2个作图
sub3 = plt.subplot(223)
sub4 = plt.subplot(224)

# 设置名称
sub1.set_title("A组实验模型结果对比")
sub2.set_title("B组实验模型结果对比")
sub3.set_title("C组实验模型结果对比")
sub4.set_title("三组实验模型整体效果对比")

# 设置网格线
sub1.grid(axis="y")
sub2.grid(axis="y")
sub3.grid(axis="y")
sub4.grid(axis="y")

# 讲坐标轴设置为不可见
sub1.spines['right'].set_color('none')
sub1.spines['top'].set_color('none')
sub1.spines['left'].set_color('none')
sub1.spines['bottom'].set_color('none')

sub2.spines['right'].set_color('none')
sub2.spines['top'].set_color('none')
sub2.spines['left'].set_color('none')
sub2.spines['bottom'].set_color('none')

sub3.spines['right'].set_color('none')
sub3.spines['top'].set_color('none')
sub3.spines['left'].set_color('none')
sub3.spines['bottom'].set_color('none')

sub4.spines['right'].set_color('none')
sub4.spines['top'].set_color('none')
sub4.spines['left'].set_color('none')
sub4.spines['bottom'].set_color('none')

# 设置y坐标轴的范围
sub1.set_ylim((0.4, 1))
sub2.set_ylim((0.4, 1))
sub3.set_ylim((0.4, 1))
sub4.set_ylim((0.4, 1))

# 设置y坐标的标签
sub1.set_ylabel("F1-score")
sub2.set_ylabel("F1-score")
sub3.set_ylabel("F1-score")
sub4.set_ylabel("TotalF1-score")


# setting the x_label value
sub1.set_xticks(x, classes)
sub2.set_xticks(x, classes)
sub3.set_xticks(x, classes)

plt.xticks(x4, ['A组', 'B组', 'C组'])

# 往子图里边添加线条
sub1.plot(x, CNN1000,marker='o', label="TextCNN", linestyle=':')
sub1.plot(x, DNN1000,marker='o',label='TextRCNN', linestyle='-.')
sub1.plot(x, BERT1000,marker='o', label='Bert')

sub2.plot(x, CNN3000, marker='o',label="TextCNN", linestyle=':')
sub2.plot(x, DNN3000, marker='o',label="TextRCNN", linestyle='-.')
sub2.plot(x, BERT3000,marker='o', label="Bert")

sub3.plot(x, CNN5000,marker='o', label="TextCNN", linestyle=':')
sub3.plot(x, DNN5000, marker='o',label="TextRCNN", linestyle='-.')
sub3.plot(x, BERT5000,marker='o', label="Bert")


sub4.plot(x4, TotalCNN,marker='o', label="TextCNN", linestyle=':')
sub4.plot(x4, TotalDNN, marker='o',label="TextRCNN", linestyle='-.')
sub4.plot(x4, TotalBert, marker='o',label="Bert")

# 设置图例
sub1.legend(loc='lower center',shadow=False, fontsize=10)
sub2.legend(loc='lower center',shadow=False, fontsize=10)
sub3.legend(loc='lower center',shadow=False, fontsize=10)
sub4.legend(loc='lower center',shadow=False, fontsize=10)

plt.legend
# 图像展示
plt.show()
