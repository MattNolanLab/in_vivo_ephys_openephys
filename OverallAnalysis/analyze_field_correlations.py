import pandas as pd
import matplotlib.pylab as plt
import plot_utility

path = 'C:/Users/s1466507/Dropbox/Edinburgh/PhD/thesis/5 firing_properties/fields_correlation.xlsx'

fields = pd.read_excel(path)
print(fields.head())
significant = (fields.p_value < 0.001)
# fields[significant]['correlation coef'].plot.hist(bins=20, color='navy')
correlation_coefs = fields[significant]['correlation coef'].values
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
fr_fig, ax = plot_utility.style_plot(ax)
# ax.hist(all_cells.avgFR, bins=400, cumulative=True, histtype='step', normed=True, color='k')
ax.hist(correlation_coefs, bins=20, color='navy')
plt.xlabel('Correlation coefficient', fontsize=22)
#plt.axvline(x=0.5, color='red')
plt.ylabel('Number of fields', fontsize=22)
plt.savefig(path + 'correlation_coef_hist.png')
plt.close()

plt.show()