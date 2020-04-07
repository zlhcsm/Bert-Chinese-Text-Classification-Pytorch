# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
#coding:utf-8
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 4.0) # 设置figure_size尺寸

labels = ["孕产次", "孕周", "胎位", "高危因素", "结果", "阿氏评分"]
x = np.arange(len(labels))
# 1000
a = [0.7940, 0.8832, 0.7586, 0.9216, 0.8235, 0.8976]
b = [0.7633, 0.8750, 0.6816, 0.7932, 0.7407, 0.8942]
c = [0.9036, 0.9394, 0.8708, 0.9548, 0.8705, 0.9259]

# # 3000
# a = [0.8632, 0.9052, 0.8284, 0.9428, 0.8606, 0.8972]
# b = [0.8374, 0.8867, 0.7845, 0.9362, 0.9362, 0.8809]
# c = [0.9260, 0.9260, 0.8825, 0.9712, 0.9192, 0.9370]

# # 5000
# a = [0.8852, 0.9118, 0.8433, 0.9499, 0.8813, 0.9106]
# b = [0.8772, 0.8948, 0.8226, 0.9452, 0.8502, 0.9145]
# c = [0.9363, 0.9529, 0.9529, 0.9717, 0.9405, 0.9520]

total_width, n = 0.6, 3
width = total_width / n
x = x - (total_width - width) / 2

fig, ax = plt.subplots()    # 定义一些子图

rects1 = ax.bar(x, a,  width=width, label='TextCNN')
rects2 = plt.bar(x + width + 0.05, b, width=width, label='TextRCNN')
rects3 = plt.bar(x + 2 * width + 0.1, c, width=width, label='BERT-CNN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-Scores')
ax.set_xticks(x + width +0.05)
ax.set_xticklabels(labels)
ax.grid(axis="y")
ax.set_axisbelow(True)

# 将图例设置为图外下边的中心位置
ax.legend(loc=8, bbox_to_anchor=(0.5, -0.2), ncol=3)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)



plt.ylim(0.6, 1.0)
plt.show()
