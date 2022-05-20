# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#fmri=pd.read_csv('G:\研究生\shuai\充电安全识别\程序\CNNdata\\迭代100——filters=16数据4值拼接.csv')
#fmri=pd.read_csv('G:\研究生\shuai\充电安全识别\程序\CNNdata\\迭代100——filters=32数据4值拼接.csv')
#fmri=pd.read_csv('G:\研究生\shuai\充电安全识别\程序\CNNdata\\迭代100——filters=48数据4值拼接.csv')

fmri=pd.read_csv('G:\研究生\shuai\充电安全识别\程序\CNNdata\\迭代100——filters=64数据4值拼接.csv')

plt.style.use({'figure.figsize':(6, 4)})
sns.set(rc={'font.sans-serif':'SimHei','lines.linewidth':3})
sns.set(style="whitegrid",font='Helvetica')

s1=sns.lineplot(x="16 kernels with 70 iterations",y="Loss/Accuracy", ci="sd", hue="Indicator",style="Indicator",  data=fmri)

plt.title('Loss and Accuracy of training and validation',y=0.99,x=0.51,fontdict={'weight':'bold','size': 14})
plt.xlabel('64 kernels and 100 iterations',fontdict={'size': 12})
plt.ylabel('Loss/Accuracy', fontdict={'size': 12})
plt.tight_layout()
plt.show()

plt=s1.get_figure()
plt.savefig('G:\研究生\shuai\充电安全识别\程序\CNNdata\\CNN迭代100—filters=64数据置信区间图.pdf',format='pdf', dpi=600)
plt.savefig('G:\研究生\shuai\充电安全识别\程序\CNNdata\\CNN迭代100—filters=64数据置信区间图.svg',format='svg', dpi=600)
plt.savefig('G:\研究生\shuai\充电安全识别\程序\CNNdata\\CNN迭代100—filters=64数据置信区间图.png',format='png', dpi=600)




