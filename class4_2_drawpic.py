import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv('F:\\YF文件\\NC-paper\\论文图\\出图\\程序\\current\\test1.csv',engine = 'python')
data1=data.values
(row,col)=data1.shape


# Nsteps length arrays empirical means and standard deviations of both
# populations over time
y1=data.iloc[:,1].values
y2=data.iloc[:,2].values
#len1=len(np.argwhere(y1))
len1=len(y1)
t=np.arange(0,len1,1)
y1=y1[0:len1]
y2=y2[0:len1]

# plot it!
fig, [[ax,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2, 3,  figsize=(6, 4))
#fig, [ax,ax1,ax2,ax3,ax4,ax5] = plt.subplots(1, 6,  figsize=(6, 6))
ax.plot(t, y1, lw=1)
ax.plot(t, -y2, lw=1)
ax.fill_between(t, 0, y1, facecolor='C2', alpha=0.4)
ax.fill_between(t, 0,-y2, facecolor='C1', alpha=0.4)
#ax.set_xlabel('collected interval')
ax.set_ylabel('current (A)')
ax.grid()

y1=data.iloc[:,3].values
y2=data.iloc[:,4].values
t=np.arange(0,len(y1),1)

ax1.plot(t, y1, lw=1)
ax1.plot(t, -y2, lw=1)
ax1.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax1.fill_between(t, 0,-y2, facecolor='C1', alpha=0.4)

#ax1.set_xlabel('collected interval')
#ax1.set_ylabel('current (A)')
ax1.grid()

y1=data.iloc[:,5].values
y2=data.iloc[:,6].values

ax2.plot(t, y1, lw=1, label='abnormal')
ax2.plot(t, -y2, lw=1, label='normal')
ax2.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax2.fill_between(t, 0,-y2, facecolor='C1', alpha=0.4)

#ax1.set_xlabel('collected interval')
#ax1.set_ylabel('current (A)')
ax2.grid()

y1=data.iloc[:,7].values
y2=data.iloc[:,8].values

ax3.plot(t, y1, lw=1)
ax3.plot(t, -y2, lw=1)
ax3.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax3.fill_between(t, 0,-y2, facecolor='C1', alpha=0.4)

ax3.set_xlabel('intervals')
ax3.set_ylabel('current (A)')
ax3.grid()

y1=data.iloc[:,9].values
y2=data.iloc[:,10].values

ax4.plot(t, y1, lw=1)
ax4.plot(t, -y2, lw=1)
ax4.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax4.fill_between(t, 0,-y2, facecolor='C1', alpha=0.4)

ax4.set_xlabel('intervals')
#ax4.set_ylabel('current (A)')
ax4.grid()

y1=data.iloc[:,11].values
y2=data.iloc[:,12].values

ax5.plot(t, y1, lw=1 )
ax5.plot(t, -y2, lw=1)
ax5.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5.fill_between(t, 0,-y2, facecolor='C1', alpha=0.4)

ax5.set_xlabel('intervals')
#ax5.set_ylabel('current (A)')
ax5.grid()

fig.tight_layout()
plt.show()
# fig.savefig('.\Class2.svg', format='svg',dpi=600)
# fig.savefig('.\Class2.png', format='png',dpi=600)