import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
fmri=pd.read_csv('G:\研究生\shuai\充电安全识别\程序\CNNdata\\迭代100——filters=64数据4值拼接.csv')
#plt.style.use('_mpl-gallery')
#fmri.event
# make data
np.random.seed(1)
x = np.linspace(0, 8, 16)
y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

# plot
fig, ax = plt.subplots()

ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/2, linewidth=2)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()