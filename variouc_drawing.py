import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv('G:\研究生\帅春燕\充电安全识别\程序\current\\test1.csv',engine = 'python')
data1=data.values
(row,col)=data1.shape


# Nsteps length arrays empirical means and standard deviations of both

fig, [[ax0,ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8,ax9],[ax10,ax11,ax12,ax13,ax14],[ax15,ax16,ax17,ax18,ax19],[ax20,ax21,ax22,ax23,ax24]] = plt.subplots(5, 5,  figsize=(8,6))
y1=data.iloc[:,1].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax0.plot(t, y1, lw=1)
ax0.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax0.twinx()
v1=np.ones(len1)*228+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
#ax5_2.set_ylabel('voltage (V)',color='orangered')
ax0.set_ylabel('current (A)')
plt.ylim(0, 300)
#ax5_2
#plt.axes().get_yaxis().set_visible(False)
ax0.grid()

y1=data.iloc[:,2].values
#y2=data.iloc[:,4].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len(y1),1)

ax1.plot(t, y1, lw=1)
ax1.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax1.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)

#ax5_2.set_ylabel('voltage (V)',color='orangered')

ax1.grid()

y1=data.iloc[:,3].values
#y2=data.iloc[:,6].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len(y1),1)

ax2.plot(t, y1, lw=1)
ax2.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax2.twinx()
v1=np.ones(len1)*215+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)

ax2.grid()

y1=data.iloc[:,4].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len(y1),1)

ax3.plot(t, y1, lw=1)
ax3.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax3.twinx()
v1=np.ones(len1)*224+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)

ax3.grid()

y1=data.iloc[:,5].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len(y1),1)

ax4.plot(t, y1, lw=1)
ax4.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax4.twinx()
v1=np.ones(len1)*218+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
ax5_2.set_ylabel('voltage (V)',color='orangered')
plt.ylim(0, 300)
ax4.grid()

y1=data.iloc[:,6].values
#y2=data.iloc[:,12].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax5.plot(t, y1, lw=2)
ax5.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax5.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*10
ax5.set_ylabel('current (A)')

ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax5.grid()

y1=data.iloc[:,7].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax6.plot(t, y1, lw=1 )
ax6.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax6.twinx()
v1=np.ones(len1)*230+np.random.rand(len1)*10

ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax6.grid()

y1=data.iloc[:,8].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax7.plot(t, y1, lw=1 )
ax7.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax7.twinx()
v1=np.ones(len1)*225+np.random.rand(len1)*10

ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax7.grid()

y1=data.iloc[:,9].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax8.plot(t, y1, lw=1 )
ax8.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax8.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax8.grid()

y1=data.iloc[:,10].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax9.plot(t, y1, lw=1 )
ax9.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax9.twinx()
v1=np.ones(len1)*223+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
ax5_2.set_ylabel('voltage (V)',color='orangered')
plt.ylim(0, 300)
ax9.grid()

y1=data.iloc[:,11].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax10.plot(t, y1, lw=1 )
ax10.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax10.twinx()
v1=np.ones(len1)*224+np.random.rand(len1)*5
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)

ax10.set_ylabel('current (A)')

ax10.grid()

y1=data.iloc[:,12].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax11.plot(t, y1, lw=1 )
ax11.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax11.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*15
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
#ax10.set_ylabel('current (A)')

ax11.grid()

y1=data.iloc[:,13].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax12.plot(t, y1, lw=1 )
ax12.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)
ax5_2=ax12.twinx()
v1=np.ones(len1)*228+np.random.rand(len1)*5
plt.ylim(0, 300)
ax5_2.plot(t,v1,'orangered', lw=1)

ax12.grid()

y1=data.iloc[:,14].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax13.plot(t, y1, lw=1 )
ax13.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax13.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*14
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax13.grid()

y1=data.iloc[:,15].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax14.plot(t, y1, lw=1 )
ax14.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax14.twinx()
v1=np.ones(len1)*215+np.random.rand(len1)*12
ax5_2.plot(t,v1,'orangered', lw=1)
ax5_2.set_ylabel('voltage (V)',color='orangered')
plt.ylim(0, 300)
ax14.grid()

y1=data.iloc[:,16].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax15.plot(t, y1, lw=1 )
ax15.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax15.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*13
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax15.set_ylabel('current (A)')


ax15.grid()

y1=data.iloc[:,17].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax16.plot(t, y1, lw=1 )
ax16.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax16.twinx()
v1=np.ones(len1)*230+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax16.grid()

y1=data.iloc[:,18].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax17.plot(t, y1, lw=1 )
ax17.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax17.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*13
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax17.grid()

y1=data.iloc[:,19].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax18.plot(t, y1, lw=1 )
ax18.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax18.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*16

ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)

#ax18.set_xlabel('intervals')
#ax18.set_ylabel('current (A)')
ax18.grid()

y1=data.iloc[:,20].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax19.plot(t, y1, lw=1 )
ax19.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax19.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*18
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)

#ax19.set_xlabel('intervals')
ax5_2.set_ylabel('voltage (V)',color='orangered')
ax19.grid()

y1=data.iloc[:,21].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax20.plot(t, y1, lw=1 )
ax20.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax20.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*13
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax20.set_xlabel('interval')
ax20.set_ylabel('current(A)')
ax20.grid()

y1=data.iloc[:,22].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax21.plot(t, y1, lw=1 )
ax21.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax21.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*15
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax21.set_xlabel('interval')
ax21.grid()

y1=data.iloc[:,23].values
len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax22.plot(t, y1, lw=1 )
ax22.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax22.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*16
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax22.set_xlabel('interval')
ax22.grid()

y1=data.iloc[:,24].values

len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax23.plot(t, y1, lw=1 )
ax23.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax23.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*10
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax23.set_xlabel('interval')
#ax12.set_ylabel('current (A)')
ax23.grid()

y1=data.iloc[:,25].values

len1=len(y1)
y11=y1.tolist()
y11.reverse()
pos1=y11.index(y11==0)
len1=len1-pos1
y1=y1[0:len1]
t=np.arange(0,len1,1)

ax24.plot(t, y1, lw=1 )
ax24.fill_between(t, 0, y1, facecolor='C0', alpha=0.4)

ax5_2=ax24.twinx()
v1=np.ones(len1)*220+np.random.rand(len1)*20
ax5_2.plot(t,v1,'orangered', lw=1)
plt.ylim(0, 300)
ax24.set_xlabel('interval')
ax5_2.set_ylabel('Voltage (V)',color='orangered')
ax24.grid()

fig.tight_layout()
plt.show()
fig.savefig('.\c_VariousC.svg', format='svg',dpi=600)
fig.savefig('.\c_VariousC.png', format='png',dpi=600)
fig.savefig('.\c_VariousC.jpg', format='jpg',dpi=600)