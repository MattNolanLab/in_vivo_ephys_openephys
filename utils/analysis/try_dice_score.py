#%%
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt

#%%
x = np.arange(0,1,0.1)
y = np.arange(0,1,0.1)

a,b = np.meshgrid(x,y)

score = 2*a*b/(a+b)

fig,ax = plt.subplots(1,2,figsize = (8,3))
ax[0].set_title('Harmonic mean')
sns.heatmap(score,ax=ax[0])

score2 = a*b
ax[1].set_title('arithmetic mean')
sns.heatmap(score2, ax=ax[1])


# %%
